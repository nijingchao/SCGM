import numpy as np
import torch
import os
import argparse
from sklearn.utils import shuffle
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from scgm_g.scgm_resnet import resnet50
from utils.utils import get_training_dataloader_breeds, get_validation_dataloader_breeds, get_test_dataloader_breeds, adjust_learning_rate_cos
from time import time
from vis import vis_tsne_multiclass_means_new
from sinkhornknopp import optimize_l_sk
# import resource

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='arguments for training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--hiddim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--mlp', action='store_false', help='use mlp head')
    parser.add_argument('--num-cycles', default=10, type=int, help='number of cycles for cosine learning rate schedule')
    parser.add_argument('--tau', default=0.1, type=float, help='variance of subclass')
    parser.add_argument('--alpha', default=0.5, type=float, help='regularization parameter on scgm likelihood')
    parser.add_argument('--lmd', default=25.0, type=float, help='parameter for sinkhorn-knopp algorithm')
    parser.add_argument('--beta', default=1.0, type=float, help='regularization parameter on self-distillation (detault: 1.0 for regular training, set 0.5 for self-distillation)')
    parser.add_argument('--kd-t', default=4.0, type=float, help='temperature of self-distillation')
    parser.add_argument('--n-subclass', default=100, type=int, help='the number of subclasses')
    parser.add_argument('--n-iter-estep', default=5, type=int, help='the number of iterations for performing e-step')
    parser.add_argument('--n-class', default=17, type=int, help='the number of superclasses, e.g., 17 for living17, 26 for nonliving26, 13 for entity13, 30 for entity30')
    parser.add_argument('--dataset', default='living17', choices=['living17', 'nonliving26', 'entity13', 'entity30'])

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_cuda = True
    info_dir = os.path.join(args.data, 'BREEDS/')
    data_dir = os.path.join(args.data, 'Data', 'CLS-LOC/')

    breeds_training_loader = get_training_dataloader_breeds(
        ds_name=args.dataset,
        info_dir=info_dir,
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        twocrops=False)

    breeds_validation_loader = get_validation_dataloader_breeds(
        ds_name=args.dataset,
        info_dir=info_dir,
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True)

    n_tr = len(breeds_training_loader.dataset)
    n_va = len(breeds_validation_loader.dataset)
    iter_per_epoch_tr = len(breeds_training_loader)
    iter_per_epoch_va = len(breeds_validation_loader)

    print('dataset={:s}'.format(args.dataset))
    print('training: size={:d},'.format(n_tr),
          'iter per epoch={:d} |'.format(iter_per_epoch_tr),
          'validation: size={:d},'.format(n_va),
          'iter per epoch={:d}'.format(iter_per_epoch_va))

    # model
    # ---
    net = resnet50(num_classes=args.n_class, num_subclasses=args.n_subclass, kd_t=args.kd_t, hiddim=args.hiddim, with_mlp=args.mlp)

    if set_cuda is True:
        net.to(device)
        net = torch.nn.DataParallel(net)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # training
    # ---
    # opt_times = ((np.linspace(0, 1, nopts) ** 2)[::-1] * epochs).tolist()
    # opt_times[0] = opt_times[0] + 1
    # print('opt_times:', opt_times)

    ls_tr_all = []
    ls1_tr_all = []
    ls2_tr_all = []
    ls3_tr_all = []

    total_time = 0

    for epoch in range(1, (args.epochs + 1)):
        t0 = time()

        adjust_learning_rate_cos(opt, args.lr, (epoch - 1), args.epochs, args.num_cycles)

        print('epoch={:d}'.format(epoch),
              'learning rate={:.3f}'.format(opt.param_groups[0]['lr']))

        # training e step
        # ---
        net.train()

        if epoch % args.n_iter_estep == 1:
            # if epoch >= opt_times[-1]:

            # _ = opt_times.pop()
            prob_tr = []
            batch_idx = []

            with torch.no_grad():
                for (images, _, labels_coarse, selected) in breeds_training_loader:

                    selected = selected.detach().cpu().numpy()
                    labels_coarse = labels_coarse.detach().cpu().numpy()
                    labels_coarse = labels_coarse.astype(np.int64)

                    batch_y = np.zeros([len(labels_coarse), args.n_class])
                    batch_y[np.arange(len(labels_coarse)), labels_coarse] = 1
                    batch_y = torch.tensor(batch_y, dtype=torch.float32)  # (n, num_class)

                    if set_cuda:
                        images = images.to(device)
                        batch_y = batch_y.to(device)

                    outputs = net(images)
                    outputs = net.module.embed(outputs)
                    batch_prob_y_x, batch_prob_y_z, batch_prob_z_x = net.module.forward_to_prob(outputs, batch_y, args.tau)
                    prob_tr.append(batch_prob_y_x.detach().detach().cpu().numpy())

                    batch_idx.append(selected)

                prob_tr = np.concatenate(prob_tr, axis=0)  # (n, k)
                batch_idx = np.concatenate(batch_idx, axis=0)

                # run sinkhorn-knopp
                # ---
                q, argmax_q = optimize_l_sk(prob_tr, args.lmd)
                q_new = np.zeros((n_tr, args.n_subclass))  # (n, k)
                q_new[batch_idx, argmax_q] = 1

        # training m step
        # ---
        ls_tr = 0
        ls1_tr = 0
        ls2_tr = 0
        ls3_tr = 0
        ls_div1_tr = 0
        ls_div2_tr = 0
        ls_div3_tr = 0
        cnt = 0
        correct = 0
        x_tr_embed = []
        y_tr_embed = []
        y_tr_embed_coarse = []
        y_pred_tr_embed = []

        for (images, labels, labels_coarse, selected) in breeds_training_loader:

            selected = selected.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            labels_coarse = labels_coarse.detach().cpu().numpy()
            labels = labels.astype(np.int64)
            labels_coarse = labels_coarse.astype(np.int64)

            batch_y = np.zeros([len(labels_coarse), args.n_class])
            batch_y[np.arange(len(labels_coarse)), labels_coarse] = 1
            batch_y = torch.tensor(batch_y, dtype=torch.float32)  # (n, num_class)

            batch_q = q_new[selected, :]
            batch_q = torch.tensor(batch_q, dtype=torch.float32)  # (n, k)

            if set_cuda:
                images = images.to(device)
                batch_y = batch_y.to(device)
                batch_q = batch_q.to(device)

            outputs = net(images)
            outputs = net.module.embed(outputs)
            ls, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3 = net.module.loss(outputs, batch_q, batch_y, args.tau, args.alpha, logit_t3=None, beta3=args.beta)

            opt.zero_grad()
            ls.backward()
            opt.step()

            ls_tr += ls.data
            ls1_tr += ls1.data
            ls2_tr += ls2.data
            ls3_tr += ls3.data
            ls_div1_tr += ls_div1
            ls_div2_tr += ls_div2
            ls_div3_tr += ls_div3

            prob_y_x, prob_z_x, prob_y_z = net.module.pred(outputs, args.tau)

            outputs = outputs.detach().cpu().numpy()
            prob_y_x = prob_y_x.detach().cpu().numpy()
            prob_z_x = prob_z_x.detach().cpu().numpy()

            correct += (prob_y_x.argmax(1) == labels_coarse).sum()
            cnt += len(labels_coarse)
            # print('number={:d}'.format(cnt))

            x_tr_embed.append(outputs)
            y_tr_embed.append(labels)
            y_tr_embed_coarse.append(labels_coarse)
            y_pred_tr_embed.append(prob_z_x)

        acc_tr = correct / cnt
        ls_tr = ls_tr.cpu().numpy() / iter_per_epoch_tr
        ls1_tr = ls1_tr.cpu().numpy() / iter_per_epoch_tr
        ls2_tr = ls2_tr.cpu().numpy() / iter_per_epoch_tr
        ls3_tr = ls3_tr.cpu().numpy() / iter_per_epoch_tr
        ls_div1_tr = ls_div1_tr / iter_per_epoch_tr
        ls_div2_tr = ls_div2_tr / iter_per_epoch_tr
        ls_div3_tr = ls_div3_tr / iter_per_epoch_tr
        x_tr_embed = np.concatenate(x_tr_embed, axis=0)
        y_tr_embed = np.concatenate(y_tr_embed, axis=0)
        y_tr_embed_coarse = np.concatenate(y_tr_embed_coarse, axis=0)
        y_pred_tr_embed = np.concatenate(y_pred_tr_embed, axis=0)  # (n, k)

        ls_tr_all.append(ls_tr)
        ls1_tr_all.append(ls1_tr)
        ls2_tr_all.append(ls2_tr)
        ls3_tr_all.append(ls3_tr)

        nmi_score_tr = normalized_mutual_info_score(y_tr_embed, y_pred_tr_embed.argmax(1), average_method='arithmetic')
        acc_score_tr = homogeneity_score(y_tr_embed, y_pred_tr_embed.argmax(1))

        epoch_time = time() - t0
        total_time += epoch_time

        # validation
        # ---
        net.eval()

        cnt = 0
        correct = 0
        x_va_embed = []
        y_va_embed = []
        y_va_embed_coarse = []
        y_pred_va_embed = []

        with torch.no_grad():
            for (images, labels, labels_coarse, selected) in breeds_validation_loader:

                labels = labels.detach().cpu().numpy()
                labels_coarse = labels_coarse.detach().cpu().numpy()
                labels = labels.astype(np.int64)
                labels_coarse = labels_coarse.astype(np.int64)

                if set_cuda:
                    images = images.to(device)

                outputs = net(images)
                outputs = net.module.embed(outputs)
                prob_y_x, prob_z_x, prob_y_z = net.module.pred(outputs, args.tau)

                outputs = outputs.detach().cpu().numpy()
                prob_y_x = prob_y_x.detach().cpu().numpy()
                prob_z_x = prob_z_x.detach().cpu().numpy()

                correct += (prob_y_x.argmax(1) == labels_coarse).sum()
                cnt += len(labels_coarse)
                # print('number={:d}'.format(cnt))

                x_va_embed.append(outputs)
                y_va_embed.append(labels)
                y_va_embed_coarse.append(labels_coarse)
                y_pred_va_embed.append(prob_z_x)

            acc_va = correct / cnt
            x_va_embed = np.concatenate(x_va_embed, axis=0)
            y_va_embed = np.concatenate(y_va_embed, axis=0)
            y_va_embed_coarse = np.concatenate(y_va_embed_coarse, axis=0)
            y_pred_va_embed = np.concatenate(y_pred_va_embed, axis=0)  # (n, k)

            nmi_score_va = normalized_mutual_info_score(y_va_embed, y_pred_va_embed.argmax(1), average_method='arithmetic')
            acc_score_va = homogeneity_score(y_va_embed, y_pred_va_embed.argmax(1))

        print('training: epoch={:d}'.format(epoch),
              'loss={:.5f}'.format(ls_tr),
              'loss1={:.5f}'.format(ls1_tr),
              'loss2={:.5f}'.format(ls2_tr),
              'loss3={:.5f}'.format(ls3_tr),
              'loss_div1={:.5f}'.format(ls_div1_tr),
              'loss_div2={:.5f}'.format(ls_div2_tr),
              'loss_div3={:.5f}'.format(ls_div3_tr),
              'acc={:.5f}'.format(acc_tr),
              'purity={:.5f}'.format(acc_score_tr),
              'nmi={:.5f}'.format(nmi_score_tr),
              '| validation: acc={:.5f}'.format(acc_va),
              'purity={:.5f}'.format(acc_score_va),
              'nmi={:.5f}'.format(nmi_score_va),
              'time={:.5f}'.format(time() - t0))

    print('total training time={:.5f}'.format(total_time))

    # save model
    # ---
    model_path = 'pretrain_model/scgm_g_' + args.dataset + '.pth'
    torch.save(net.module.state_dict(), model_path)

    # # vis training embedding
    # mu_z_tr = net.module.mu_z.data.detach().cpu().numpy()
    # mu_y_tr = net.module.mu_y.data.detach().cpu().numpy()
    # mu_z_tr = mu_z_tr / ((mu_z_tr ** 2).sum(1) ** 0.5).reshape(-1, 1)
    # mu_y_tr = mu_y_tr / ((mu_y_tr ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # x_embed_vis, y_embed_vis = shuffle(x_tr_embed, y_tr_embed_coarse)
    # x_embed_vis = x_embed_vis[:2000, :]
    # y_embed_vis = y_embed_vis[:2000]
    # x_embed_vis = x_embed_vis / ((x_embed_vis ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # destpath = 'fig/tsne_scgm_g_' + args.dataset + '_tr.png'
    # vis_tsne_multiclass_means_new(x_embed_vis, y_embed_vis, mu_z_tr, mu_y_tr, destpath, y_pred=None, destpath_correct=None)
    #
    # # vis validation embedding
    # # ---
    # x_embed_vis, y_embed_vis = shuffle(x_va_embed, y_va_embed_coarse)
    # x_embed_vis = x_embed_vis[:2000, :]
    # y_embed_vis = y_embed_vis[:2000]
    # x_embed_vis = x_embed_vis / ((x_embed_vis ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # destpath = 'fig/tsne_scgm_g_' + args.dataset + '_va.png'
    # vis_tsne_multiclass_means_new(x_embed_vis, y_embed_vis, mu_z_tr, mu_y_tr, destpath, y_pred=None, destpath_correct=None)


if __name__ == '__main__':
    main()
