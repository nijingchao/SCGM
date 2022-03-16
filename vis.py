import numpy as np
from utils.tsne import tsne
import matplotlib as mpl
# mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt


def vis_tsne_multiclass(x, y, destpath):
    x_embed = tsne(x, no_dims=2, max_iter=500)
    y2idx = {}
    idx2color = {}
    y_new = []
    cnt = 0
    for i in y:
        if i not in y2idx:
            y2idx[i] = cnt
            idx2color[cnt] = np.random.choice(range(256), size=3) / 256
            cnt = cnt + 1

        y_new.append(y2idx[i])

    y_new = np.array(y_new)

    plt.figure()
    for i in range(cnt):
        plt.scatter(x_embed[y_new == i, 0], x_embed[y_new == i, 1], edgecolors='none', marker='o', facecolors=idx2color[i], s=5)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')
    return None


def vis_tsne_multiclass_means(x, y, mu, destpath, y_pred=None, destpath_correct=None):
    n = x.shape[0]
    x = np.concatenate([x, mu], axis=0)
    x_embed = tsne(x, no_dims=2, max_iter=500)
    mu_embed = x_embed[n:, :]
    x_embed = x_embed[:n, :]

    y2idx = {}
    idx2color = {}
    y_new = []
    cnt = 0
    for i in y:
        if i not in y2idx:
            y2idx[i] = cnt
            idx2color[cnt] = np.random.choice(range(256), size=3) / 256
            cnt = cnt + 1

        y_new.append(y2idx[i])

    y_new = np.array(y_new)

    plt.figure()
    for i in range(cnt):
        plt.scatter(x_embed[y_new == i, 0], x_embed[y_new == i, 1], edgecolors='none', marker='o', facecolors=idx2color[i], s=5)

    plt.scatter(mu_embed[:, 0], mu_embed[:, 1], edgecolors='tab:orange', marker='+', facecolors='tab:orange', s=30)
    # plt.scatter(mu_embed[:, 0], mu_embed[:, 1], edgecolors='k', marker='*', facecolors='tab:purple', s=30)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')

    if y_pred is not None and destpath_correct is not None:
        correct_label = y == y_pred
        plt.figure()
        plt.scatter(x_embed[correct_label, 0], x_embed[correct_label, 1], edgecolors='none', marker='o', facecolors='tab:blue', s=5)
        plt.scatter(x_embed[(~correct_label), 0], x_embed[(~correct_label), 1], edgecolors='none', marker='o', facecolors='tab:red', s=5)
        plt.scatter(mu_embed[:, 0], mu_embed[:, 1], edgecolors='tab:orange', marker='+', facecolors='tab:orange', s=30)
        plt.xticks([])
        plt.yticks([])
        # plt.title('tsne visualization', fontdict={'fontsize': 25})
        plt.savefig(destpath_correct, bbox_inches='tight', dpi=600)
        plt.close('all')

    return None


def vis_tsne_multiclass_means_new(x, y, mu_z, mu_y, destpath, y_pred=None, destpath_correct=None):
    n = x.shape[0]
    k = mu_z.shape[0]
    x = np.concatenate([x, mu_z, mu_y], axis=0)
    x_embed = tsne(x, no_dims=2, max_iter=500)
    mu_z_embed = x_embed[n:(n + k), :]
    mu_y_embed = x_embed[(n + k):, :]
    x_embed = x_embed[:n, :]

    y2idx = {}
    idx2color = {}
    y_new = []
    cnt = 0
    for i in y:
        if i not in y2idx:
            y2idx[i] = cnt
            idx2color[cnt] = np.random.choice(range(256), size=3) / 256
            cnt = cnt + 1

        y_new.append(y2idx[i])

    y_new = np.array(y_new)

    plt.figure()
    for i in range(cnt):
        plt.scatter(x_embed[y_new == i, 0], x_embed[y_new == i, 1], edgecolors='none', marker='o', facecolors=idx2color[i], s=5)

    plt.scatter(mu_z_embed[:, 0], mu_z_embed[:, 1], edgecolors='tab:orange', marker='+', facecolors='tab:orange', s=30)
    plt.scatter(mu_y_embed[:, 0], mu_y_embed[:, 1], edgecolors='k', marker='*', facecolors='tab:purple', s=30)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')

    if y_pred is not None and destpath_correct is not None:
        correct_label = y == y_pred
        plt.figure()
        plt.scatter(x_embed[correct_label, 0], x_embed[correct_label, 1], edgecolors='none', marker='o', facecolors='tab:blue', s=5)
        plt.scatter(x_embed[(~correct_label), 0], x_embed[(~correct_label), 1], edgecolors='none', marker='o', facecolors='tab:red', s=5)
        plt.scatter(mu_z_embed[:, 0], mu_z_embed[:, 1], edgecolors='tab:orange', marker='+', facecolors='tab:orange', s=30)
        plt.scatter(mu_y_embed[:, 0], mu_y_embed[:, 1], edgecolors='k', marker='*', facecolors='tab:purple', s=30)
        plt.xticks([])
        plt.yticks([])
        # plt.title('tsne visualization', fontdict={'fontsize': 25})
        plt.savefig(destpath_correct, bbox_inches='tight', dpi=600)
        plt.close('all')

    return None


def draw_tsne_multiclass(x, y, num_class, destpath):
    x_embed = tsne(x, no_dims=2, max_iter=500)

    y2color = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
               'tab:cyan', 'darkgreen', 'lightgreen', 'goldenrod', 'peru', 'tan', 'slategrey', 'teal', 'lightsteelblue']

    plt.figure()
    for i in range(num_class):
        plt.scatter(x_embed[y == i, 0], x_embed[y == i, 1], edgecolors='none', marker='o', facecolors=y2color[i], s=15, alpha=0.7)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')

    return None


def draw_tsne_multiclass_means(x, y, mu, num_class, destpath):
    n = x.shape[0]
    x = np.concatenate([x, mu], axis=0)
    x_embed = tsne(x, no_dims=2, max_iter=500)
    mu_embed = x_embed[n:, :]
    x_embed = x_embed[:n, :]

    y2color = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
               'tab:cyan', 'darkgreen', 'lightgreen', 'goldenrod', 'peru', 'tan', 'slategrey', 'teal', 'lightsteelblue']

    plt.figure()
    for i in range(num_class):
        plt.scatter(x_embed[y == i, 0], x_embed[y == i, 1], edgecolors='none', marker='o', facecolors=y2color[i], s=15, alpha=0.7)

    plt.scatter(mu_embed[:, 0], mu_embed[:, 1], edgecolors='k', marker='*', facecolors='khaki', s=80)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')

    return None


def draw_tsne_multiclass_means_new(x, y, mu_z, mu_y, num_class, destpath):
    n = x.shape[0]
    k = mu_z.shape[0]
    x = np.concatenate([x, mu_z, mu_y], axis=0)
    x_embed = tsne(x, no_dims=2, max_iter=500)
    mu_z_embed = x_embed[n:(n + k), :]
    mu_y_embed = x_embed[(n + k):, :]
    x_embed = x_embed[:n, :]

    y2color = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
               'tab:cyan', 'darkgreen', 'lightgreen', 'goldenrod', 'peru', 'tan', 'slategrey', 'teal', 'lightsteelblue']

    plt.figure()
    for i in range(num_class):
        plt.scatter(x_embed[y == i, 0], x_embed[y == i, 1], edgecolors='none', marker='o', facecolors=y2color[i], s=15, alpha=0.7)

    plt.scatter(mu_z_embed[:, 0], mu_z_embed[:, 1], marker='+', facecolors='tab:orange', s=80)
    plt.scatter(mu_y_embed[:, 0], mu_y_embed[:, 1], edgecolors='k', marker='*', facecolors='khaki', s=80)

    plt.xticks([])
    plt.yticks([])
    # plt.title('tsne visualization', fontdict={'fontsize': 25})
    plt.savefig(destpath, bbox_inches='tight', dpi=600)
    plt.close('all')

    return None


def eval_reliability_diagram(y_pred, y_true, probs, num_bins, destpath):
    '''
    :param y_pred:
    :param y_true:
    :param probs:
    :param num_bins:
    :param destpath:
    :return:
    '''
    n = len(y_true)
    intv = 1 / num_bins
    intvs = np.arange(0, 1, intv)
    acc = []
    conf = []
    bs = []
    for i in range(num_bins):
        lb = intvs[i]
        ub = lb + intv
        if i == 0:
            idx = np.logical_and(probs >= lb, probs <= ub)
        else:
            idx = np.logical_and(probs > lb, probs <= ub)

        y_pred_i = y_pred[idx]
        y_true_i = y_true[idx]
        probs_i = probs[idx]
        bs_i = idx.sum()

        if bs_i == 0:
            acc.append(0)
            conf.append(0)
            bs.append(0)
        else:
            acc_i = (y_pred_i == y_true_i).sum() / bs_i
            conf_i = probs_i.sum() / bs_i

            acc.append(acc_i)
            conf.append(conf_i)
            bs.append(bs_i)

    acc = np.array(acc)
    conf = np.array(conf)
    bs = np.array(bs)

    ece = (np.abs(acc - conf) * (bs / n)).sum()
    mce = (np.abs(acc - conf)).max()

    plt.figure()
    plt.bar(intvs, acc, width=intv, align='edge', color='b', edgecolor='black')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='-.')
    # plt.xticks(intvs, intvs, rotation=90, ha='center')
    plt.title('reliability diagram')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    # for i, v in enumerate(acc):
    #     plt.text(i - 0.3, v, '%.4f' % v, color='black', fontsize=5)
    plt.savefig(destpath, bbox_inches='tight', dpi=600)

    return ece, mce, acc, conf, bs
