import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
from sklearn.utils import shuffle
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from scgm_a.model_generator import MoEncoderGenerator
from utils.utils import get_training_dataloader_breeds, get_validation_dataloader_breeds, get_test_dataloader_breeds, adjust_learning_rate_cos, write_values
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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--hiddim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--queue-k', default=65536, type=int, help='queue size; number of negative keys per class (default: 65536)')
    parser.add_argument('--encoder-m', default=0.999, type=float, help='momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--mlp', action='store_false', help='use mlp head')
    parser.add_argument('--num-cycles', default=10, type=int, help='number of cycles for cosine learning rate schedule')
    parser.add_argument('--queue', type=str, default='multi', choices=['single', 'multi', 'none'], help='type of queue (default: multi)')
    parser.add_argument('--metric', type=str, default='angular', choices=['norm', 'angular'], help='the metric to apply before calculating contrastive dot products')
    parser.add_argument('--calc-types', nargs='*', type=str, default=['cls', 'cst_by_class', 'cst_by_subclass', 'cst_two_class'],
                        choices=['cls', 'cst_by_class', 'cst_all', 'cst_all_by_cls', 'cst_by_subclass', 'cst_by_subclass_angnorm', 'cst_two_class'],
                        help='a list of loss calculators to be used for training')
    parser.add_argument('--head-type', type=str, default='seq_em')
    parser.add_argument('--cls-t', default=1.0, type=float, help='temperature of classification loss function')
    parser.add_argument('--cst-t', default=0.2, type=float, help='temperature of contrastive loss function')
    parser.add_argument('--tau1', default=0.1, type=float, help='variance of subclass')
    parser.add_argument('--tau2', default=1.0, type=float, help='variance of superclass')
    parser.add_argument('--alpha', default=0.5, type=float, help='regularization parameter on scgm likelihood')
    parser.add_argument('--lmd', default=25.0, type=float, help='parameter for sinkhorn-knopp algorithm')
    parser.add_argument('--n-subclass', default=100, type=int, help='the number of subclasses')
    parser.add_argument('--n-iter-estep', default=5, type=int, help='the number of iterations for performing e-step')
    parser.add_argument('--head-norm', action='store_false', help='normalization on the output of the last layer')
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
        twocrops=True)

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
    model = MoEncoderGenerator().generate_momentum_model(arch=args.arch,
                                                         head_type=args.head_type,
                                                         dim=args.hiddim,
                                                         K=args.queue_k,
                                                         m=args.encoder_m,
                                                         T=[args.cls_t, args.cst_t, args.tau1, args.tau2],
                                                         mlp=args.mlp,
                                                         num_classes=args.n_class,
                                                         num_subclasses=args.n_subclass,
                                                         norm=args.head_norm,
                                                         queue_type=args.queue,
                                                         metric=args.metric,
                                                         calc_types=args.calc_types)

    if set_cuda is True:
        model.to(device)
        model = torch.nn.DataParallel(model)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # training
    # ---
    ls_tr_all = []
    ls1_tr_all = []
    ls2_tr_all = []
    ls3_tr_all = []
    ls4_tr_all = []

    total_time = 0

    for epoch in range(1, (args.epochs + 1)):
        t0 = time()

        adjust_learning_rate_cos(opt, args.lr, (epoch - 1), args.epochs, args.num_cycles)

        print('epoch={:d}'.format(epoch),
              'learning rate={:.3f}'.format(opt.param_groups[0]['lr']))

        # training e step
        # ---
        model.train()

        if epoch % args.n_iter_estep == 1:
            prob_tr = []
            batch_idx = []

            with torch.no_grad():
                for (images, _, labels_coarse, selected) in breeds_training_loader:

                    images_q = images[0]
                    selected = selected.detach().cpu().numpy()

                    if set_cuda:
                        images_q = images_q.to(device)
                        labels_coarse = labels_coarse.to(device)

                    outputs, _ = model.module.encoder_q(images_q)
                    batch_prob_y_x, _, _ = model.module.forward_to_prob(outputs, labels_coarse, args.tau1)
                    prob_tr.append(batch_prob_y_x.detach().detach().cpu().numpy())

                    batch_idx.append(selected)

                prob_tr = np.concatenate(prob_tr, axis=0)  # (n, k)
                batch_idx = np.concatenate(batch_idx, axis=0)

                # run sinkhorn-knopp
                # ---
                _, argmax_q = optimize_l_sk(prob_tr, args.lmd)
                argmax_q_new = np.zeros(n_tr)  # (n, k)
                argmax_q_new[batch_idx] = argmax_q

        # training m step
        # ---
        ls_tr = 0
        ls1_tr = 0
        ls2_tr = 0
        ls3_tr = 0
        ls4_tr = 0
        cnt = 0
        correct_1 = 0
        correct_2 = 0
        x_tr_embed = []
        y_tr_embed_coarse = []
        y_tr_embed = []
        y_pred_tr_embed = []

        for (images, labels, labels_coarse, selected) in breeds_training_loader:

            images_q = images[0]
            images_k = images[1]

            selected = selected.detach().cpu().numpy()
            batch_argmax_q = argmax_q_new[selected]
            batch_argmax_q = torch.tensor(batch_argmax_q, dtype=torch.int64)

            if set_cuda:
                images_q = images_q.to(device)
                images_k = images_k.to(device)
                labels_coarse = labels_coarse.to(device)
                batch_argmax_q = batch_argmax_q.to(device)

            logits_and_labels = model(im_q=images_q, im_k=images_k, cls_labels=labels_coarse, subcls_labels=batch_argmax_q)

            ls1 = F.cross_entropy(logits_and_labels[0][0], logits_and_labels[0][1], weight=None)
            ls2 = F.cross_entropy(logits_and_labels[1][0], logits_and_labels[1][1], weight=None)
            ls3 = F.cross_entropy(logits_and_labels[2][0], logits_and_labels[2][1], weight=None)
            ls4 = F.cross_entropy(logits_and_labels[3][0], logits_and_labels[3][1], weight=None)
            ls = ls1 + ls2 + args.alpha * (ls3 + ls4)

            opt.zero_grad()
            ls.backward()
            opt.step()

            ls_tr += ls.data
            ls1_tr += ls1.data
            ls2_tr += ls2.data
            ls3_tr += ls3.data
            ls4_tr += ls4.data

            batch_embed, batch_pred_coarse_1 = model.module.encoder_q(images_q)
            batch_pred_coarse_1 = torch.softmax(batch_pred_coarse_1, dim=1)
            batch_pred_coarse_2, batch_pred, _ = model.module.pred(batch_embed, args.tau1)

            batch_embed = batch_embed.detach().cpu().numpy()
            batch_pred_coarse_1 = batch_pred_coarse_1.detach().cpu().numpy()
            batch_pred_coarse_2 = batch_pred_coarse_2.detach().cpu().numpy()
            batch_pred = batch_pred.detach().cpu().numpy()

            labels_coarse = labels_coarse.detach().cpu().numpy()
            labels_coarse = labels_coarse.astype(np.int64)
            labels = labels.detach().cpu().numpy()
            labels = labels.astype(np.int64)

            correct_1 += (batch_pred_coarse_1.argmax(1) == labels_coarse).sum()
            correct_2 += (batch_pred_coarse_2.argmax(1) == labels_coarse).sum()
            cnt += len(labels_coarse)

            x_tr_embed.append(batch_embed)
            y_tr_embed_coarse.append(labels_coarse)
            y_tr_embed.append(labels)
            y_pred_tr_embed.append(batch_pred)

        acc_tr_1 = correct_1 / cnt
        acc_tr_2 = correct_2 / cnt
        ls_tr = ls_tr.cpu().numpy() / iter_per_epoch_tr
        ls1_tr = ls1_tr.cpu().numpy() / iter_per_epoch_tr
        ls2_tr = ls2_tr.cpu().numpy() / iter_per_epoch_tr
        ls3_tr = ls3_tr.cpu().numpy() / iter_per_epoch_tr
        ls4_tr = ls4_tr.cpu().numpy() / iter_per_epoch_tr
        x_tr_embed = np.concatenate(x_tr_embed, axis=0)
        y_tr_embed_coarse = np.concatenate(y_tr_embed_coarse, axis=0)
        y_tr_embed = np.concatenate(y_tr_embed, axis=0)
        y_pred_tr_embed = np.concatenate(y_pred_tr_embed, axis=0)  # (n, k)

        ls_tr_all.append(ls_tr)
        ls1_tr_all.append(ls1_tr)
        ls2_tr_all.append(ls2_tr)
        ls3_tr_all.append(ls3_tr)
        ls4_tr_all.append(ls4_tr)

        nmi_score_tr = normalized_mutual_info_score(y_tr_embed, y_pred_tr_embed.argmax(1), average_method='arithmetic')
        acc_score_tr = homogeneity_score(y_tr_embed, y_pred_tr_embed.argmax(1))

        epoch_time = time() - t0
        total_time += epoch_time

        # validation
        # ---
        model.eval()
        cnt = 0
        correct_1 = 0
        correct_2 = 0
        x_va_embed = []
        y_va_embed_coarse = []
        y_va_embed = []
        y_pred_va_embed = []

        with torch.no_grad():
            for (images, labels, labels_coarse, selected) in breeds_validation_loader:

                images_q = images

                if set_cuda:
                    images_q = images_q.to(device)

                batch_embed, batch_pred_coarse_1 = model.module.encoder_q(images_q)
                batch_pred_coarse_1 = torch.softmax(batch_pred_coarse_1, dim=1)
                batch_pred_coarse_2, batch_pred, _ = model.module.pred(batch_embed, args.tau1)

                batch_embed = batch_embed.detach().cpu().numpy()
                batch_pred_coarse_1 = batch_pred_coarse_1.detach().cpu().numpy()
                batch_pred_coarse_2 = batch_pred_coarse_2.detach().cpu().numpy()
                batch_pred = batch_pred.detach().cpu().numpy()

                labels_coarse = labels_coarse.detach().cpu().numpy()
                labels_coarse = labels_coarse.astype(np.int64)
                labels = labels.detach().cpu().numpy()
                labels = labels.astype(np.int64)

                correct_1 += (batch_pred_coarse_1.argmax(1) == labels_coarse).sum()
                correct_2 += (batch_pred_coarse_2.argmax(1) == labels_coarse).sum()
                cnt += len(labels_coarse)

                x_va_embed.append(batch_embed)
                y_va_embed_coarse.append(labels_coarse)

                x_va_embed.append(batch_embed)
                y_va_embed_coarse.append(labels_coarse)
                y_va_embed.append(labels)
                y_pred_va_embed.append(batch_pred)

            acc_va_1 = correct_1 / cnt
            acc_va_2 = correct_2 / cnt
            x_va_embed = np.concatenate(x_va_embed, axis=0)
            y_va_embed_coarse = np.concatenate(y_va_embed_coarse, axis=0)
            y_va_embed = np.concatenate(y_va_embed, axis=0)
            y_pred_va_embed = np.concatenate(y_pred_va_embed, axis=0)  # (n, k)

            nmi_score_va = normalized_mutual_info_score(y_va_embed, y_pred_va_embed.argmax(1), average_method='arithmetic')
            acc_score_va = homogeneity_score(y_va_embed, y_pred_va_embed.argmax(1))

        print('training: epoch={:d}'.format(epoch),
              'loss={:.5f}'.format(ls_tr),
              'loss1={:.5f}'.format(ls1_tr),
              'loss2={:.5f}'.format(ls2_tr),
              'loss3={:.5f}'.format(ls3_tr),
              'loss4={:.5f}'.format(ls4_tr),
              'acc1={:.5f}'.format(acc_tr_1),
              'acc2={:.5f}'.format(acc_tr_2),
              'purity={:.5f}'.format(acc_score_tr),
              'nmi={:.5f}'.format(nmi_score_tr),
              '| validation: acc1={:.5f}'.format(acc_va_1),
              'acc2={:.5f}'.format(acc_va_2),
              'purity={:.5f}'.format(acc_score_va),
              'nmi={:.5f}'.format(nmi_score_va),
              'time={:.5f}'.format(time() - t0))

    print('total training time={:.5f}'.format(total_time))

    # save model
    # ---
    model_path = 'pretrain_model/scgm_a_' + args.dataset + ' .pth'
    torch.save(model.module.state_dict(), model_path)

    # # vis training embedding
    # mu_z_tr = model.module.encoder_q.fc.mu_z.data.detach().cpu().numpy()
    # mu_y_tr = model.module.encoder_q.fc.mu_y.data.detach().cpu().numpy()
    # mu_z_tr = mu_z_tr / ((mu_z_tr ** 2).sum(1) ** 0.5).reshape(-1, 1)
    # mu_y_tr = mu_y_tr / ((mu_y_tr ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # x_embed_vis, y_embed_vis = shuffle(x_tr_embed, y_tr_embed_coarse)
    # x_embed_vis = x_embed_vis[:2000, :]
    # y_embed_vis = y_embed_vis[:2000]
    # x_embed_vis = x_embed_vis / ((x_embed_vis ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # destpath = '../fig/tsne_scgm_a_' + args.dataset + '_tr.png'
    # vis_tsne_multiclass_means_new(x_embed_vis, y_embed_vis, mu_z_tr, mu_y_tr, destpath, y_pred=None, destpath_correct=None)
    #
    # # vis validation embedding
    # # ---
    # x_embed_vis, y_embed_vis = shuffle(x_va_embed, y_va_embed_coarse)
    # x_embed_vis = x_embed_vis[:2000, :]
    # y_embed_vis = y_embed_vis[:2000]
    # x_embed_vis = x_embed_vis / ((x_embed_vis ** 2).sum(1) ** 0.5).reshape(-1, 1)
    #
    # destpath = '../fig/tsne_scgm_a_' + args.dataset + '_va.png'
    # vis_tsne_multiclass_means_new(x_embed_vis, y_embed_vis, mu_z_tr, mu_y_tr, destpath, y_pred=None, destpath_correct=None)


if __name__ == '__main__':
    main()
