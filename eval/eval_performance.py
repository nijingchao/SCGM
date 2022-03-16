import numpy as np
import scipy
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


# def classify(classifier_name, query_feature, support_features, support_ys):
#     if len(support_features.shape) == 1:
#         support_features = np.expand_dims(support_features, 0)
#     if len(query_feature.shape) == 1:
#         query_feature = np.expand_dims(query_feature, 0)
#     if classifier_name == 'LR':
#         clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
#         # clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
#         clf.fit(support_features, support_ys)
#         return clf.predict(query_feature)
#     elif classifier_name == 'SGDLR':
#         clf = SGDClassifier(loss='log', max_iter=1000)
#         clf.fit(support_features, support_ys)
#         return clf.predict(query_feature)
#     elif classifier_name == 'NN':
#         return NN(support_features, support_ys, query_feature)
#     elif classifier_name == 'Cosine':
#         return Cosine(support_features, support_ys, query_feature)
#     elif classifier_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=5, weights='distance',n_jobs=12)
#         clf.fit(support_features,support_ys)
#         return clf.predict(query_feature)
#     else:
#         raise NotImplementedError('classifier not supported: {}'.format(classifier_name))


def classify(classifier_name, support_features, support_ys):
    if len(support_features.shape) == 1:
        support_features = np.expand_dims(support_features, 0)
    if classifier_name == 'LR':
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        # clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
        clf.fit(support_features, support_ys)
        return clf
    elif classifier_name == 'SGDLR':
        clf = SGDClassifier(loss='log', max_iter=1000)
        clf.fit(support_features, support_ys)
        return clf
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=12)
        clf.fit(support_features,support_ys)
        return clf
    else:
        raise NotImplementedError('classifier not supported: {}'.format(classifier_name))


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred


# def loop_get_features(net, x, y, only_base, is_norm, batch_size):
#     batches = int(len(x) / batch_size) + 1
#     all_features = []
#     all_ys = []
#     for b in range(batches):
#         features, ys = get_features(net, x[b * batch_size:b * batch_size + batch_size], y[b * batch_size:b * batch_size + batch_size], only_base, is_norm)
#         all_features.append(features)
#         all_ys.append(ys)
#     all_features = np.concatenate(all_features)
#     all_ys = np.concatenate(all_ys)
#     return all_features, all_ys
#
#
# def get_features(net, x, y, only_base, is_norm):
#     x = x.cuda()
#
#     if not only_base:
#         x = net(x)
#         if type(x) is tuple:
#             x = x[0]
#         x = x.view(x.size(0), -1)
#     else:
#         x = forward(net, x)
#     if is_norm:
#         x = torch.nn.functional.normalize(x)
#     x = x.detach().cpu().numpy()
#     y = y.view(-1).numpy()
#     return x, y

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h
