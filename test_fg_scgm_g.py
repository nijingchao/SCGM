import numpy as np
import torch
import torch.nn.functional as F
import os
import argparse
from sklearn import metrics
from time import time
from scgm_g.scgm_resnet import resnet50
# from utils.model_toolkit import identity_layer
from utils.utils import get_test_dataloader_breeds
from eval.eval_performance import classify, mean_confidence_interval
# import resource

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    parser = argparse.ArgumentParser(description='arguments for training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--n-test-runs', default=1000, type=int, help='the number of test runs')
    parser.add_argument('--n-shots', default=1, type=int)
    parser.add_argument('--n-queries', default=15, type=int)
    parser.add_argument('--n-aug-support-samples', default=5, type=int, help='the number of support samples after augmentation')
    parser.add_argument('--feat-norm', action='store_true', help='normalization on features')
    parser.add_argument('--classifier', default='LR', choices=['LR', 'SGDLR', 'KNN'])
    parser.add_argument('--hiddim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--mlp', action='store_false', help='use mlp head')
    parser.add_argument('--kd-t', default=4.0, type=float, help='temperature of self-distillation')
    parser.add_argument('--n-subclass', default=100, type=int, help='the number of subclasses')
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

    breeds_test_loader = get_test_dataloader_breeds(
        ds_name=args.dataset,
        info_dir=info_dir,
        data_dir=data_dir,
        n_test_runs=args.n_test_runs,
        n_ways=10000,
        n_shots=args.n_shots,
        n_queries=args.n_queries,
        n_aug_support_samples=args.n_aug_support_samples,
        fg=True,
        batch_size=1,
        num_workers=0)

    # load model
    # ---
    net = resnet50(num_classes=args.n_class, num_subclasses=args.n_subclass, kd_t=args.kd_t, hiddim=args.hiddim, with_mlp=args.mlp)
    weights_path = 'pretrain_model/scgm_g_' + args.dataset + '.pth'
    net.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
    # net.fc_enc = identity_layer()

    if set_cuda is True:
        net.to(device)
        # net = torch.nn.DataParallel(net)

    net.eval()

    # evaluation
    # ---
    with torch.no_grad():
        acc = []
        t0 = time()

        for (run_idx, batch_data) in enumerate(breeds_test_loader):

            support_xs, support_ys, query_xs, query_ys = batch_data
            support_xs = support_xs[0]
            support_ys = support_ys[0]
            query_xs = query_xs[0]
            query_ys = query_ys[0]

            # load support set embeddings
            # ---
            support_feats = []

            if len(support_ys) > args.batch_size:
                loop_range = range(0, (len(support_ys) - args.batch_size), args.batch_size)
            else:
                loop_range = [0]

            for i in loop_range:
                if (len(support_ys) - i) < 2 * args.batch_size:
                    batchsz_iter = len(support_ys) - i
                else:
                    batchsz_iter = args.batch_size

                batch_support_xs = support_xs[i:(i + batchsz_iter)]
                if set_cuda is True:
                    batch_support_xs = batch_support_xs.to(device)

                batch_support_xs = net(batch_support_xs)
                # batch_support_xs = net.module.embed(batch_support_xs)
                batch_support_xs = net.embed(batch_support_xs)
                if args.feat_norm is True:
                    batch_support_xs = F.normalize(batch_support_xs, p=2, dim=1)

                support_feats.append(batch_support_xs.detach().cpu().numpy())

            support_feats = np.concatenate(support_feats, axis=0)

            # load query set embeddings
            # ---
            query_feats = []

            if len(query_ys) > args.batch_size:
                loop_range = range(0, (len(query_ys) - args.batch_size), args.batch_size)
            else:
                loop_range = [0]

            for i in loop_range:
                if (len(query_ys) - i) < 2 * args.batch_size:
                    batchsz_iter = len(query_ys) - i
                else:
                    batchsz_iter = args.batch_size

                batch_query_xs = query_xs[i:(i + batchsz_iter)]
                if set_cuda is True:
                    batch_query_xs = batch_query_xs.to(device)

                batch_query_xs = net(batch_query_xs)
                # batch_query_xs = net.module.embed(batch_query_xs)
                batch_query_xs = net.embed(batch_query_xs)
                if args.feat_norm is True:
                    batch_query_xs = F.normalize(batch_query_xs, p=2, dim=1)

                query_feats.append(batch_query_xs.detach().cpu().numpy())

            query_feats = np.concatenate(query_feats, axis=0)

            # classification
            # ---
            clf = classify(args.classifier, support_feats, support_ys)
            support_preds = clf.predict(support_feats)
            query_preds = clf.predict(query_feats)

            # evaluation
            # ---
            acc_tr = metrics.accuracy_score(support_ys, support_preds)
            acc_te = metrics.accuracy_score(query_ys, query_preds)
            acc.append(acc_te)

            print('[{:d}'.format(run_idx),
                  '/ {:d}]'.format(args.n_test_runs),
                  'training: acc = {:.5f}'.format(acc_tr * 100),
                  '| testing: acc = {:.5f}'.format(acc_te * 100))

            del clf

        acc_mn, acc_std = mean_confidence_interval(acc)
        print('accuracy={:.5f}'.format(acc_mn * 100),
              'std={:.5f}'.format(acc_std * 100),
              'time={:.5f}'.format(time() - t0))


if __name__ == '__main__':
    main()
