import torch
from torch import nn
from torch.nn import Sequential

METRICS = ['norm', 'angular']
CALCULATORS = ['cls', 'cst_by_class', 'cst_all', 'cst_all_by_class']


class CalculatorFactory(object):
    def create_calculator(self, calc_type, metric_type, T=1, num_classes=0):
        if metric_type == 'angular':
            metric = Angular()
        elif metric_type == 'norm':
            metric = Norm()
        else:
            raise NotImplementedError(metric_type)
        if calc_type == 'cls':
            return ClsCalculator(T)
        elif calc_type == 'cst_by_class':
            return ContrastByClassCalculator(T, metric, num_classes)
        elif calc_type == 'cst_all':
            return ContrastAllCalculator(T, metric)
        elif calc_type == 'cst_all_by_class':
            return ContrastAllByClassCalculator(T, metric, num_classes)
        elif calc_type == 'cst_by_subclass':
            return ContrastBySubclassCalculator(T)
        elif calc_type == 'cst_by_subclass_angnorm':
            return ContrastBySubclassAngNormCalculator(T)
        elif calc_type == 'cst_two_class':
            return ContrastTwoClassCalculator(T)
        else:
            raise NotImplementedError(calc_type)


class LogitsLabelsCalculator(object):
    def calculate(self, q_outs):
        raise NotImplementedError


class ClsCalculator(LogitsLabelsCalculator):
    """
    encoder_q returns q_outs: a tuple of (embedding[128], class_preds[#num_classes])
    """

    def __init__(self, T):
        self.T = T

    def calculate(self, q_outs, cls_labels=None, **kwargs):
        if isinstance(q_outs, tuple):
            cls_logits = q_outs[1]
        else:
            cls_logits = q_outs
        return cls_logits, cls_labels


class ContrastAllCalculator(LogitsLabelsCalculator):
    def __init__(self, T, metric):
        self.T = T
        self.metric = metric

    def calculate(self, q_outs, k=None, queue=None, **kwargs):
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        queue = queue[-1, :, :] if len(queue.shape) > 2 else queue

        q = nn.functional.normalize(q)
        k = nn.functional.normalize(k)
        queue = nn.functional.normalize(queue, dim=0)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, queue])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels


class ContrastAllByClassCalculator(LogitsLabelsCalculator):
    def __init__(self, T, metric, num_classes):
        self.T = T
        self.metric = metric
        self.num_classes = num_classes

    def calculate(self, q_outs, k=None, encoder_q=None, queue=None, cls_labels=None, **kwargs):
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        queue = queue[:self.num_classes, :, :]

        class_weights = self._extract_class_weights(encoder_q)
        class_weights_by_label = class_weights[cls_labels]
        q = self.metric.preprocess(q, class_weights=class_weights_by_label)
        k = self.metric.preprocess(k, class_weights=class_weights_by_label)
        queue = self.metric.preprocess(queue, class_weights=class_weights.unsqueeze(2))
        queue = torch.transpose(queue, 0, 1).reshape(q.shape[1], -1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_neg = torch.einsum('nd,dk->nk', q, queue)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels

    @staticmethod
    def _extract_class_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            # ugly hack - sequential fc means it's the mlp, so the actual output layer is fc[2]
            class_weight = encoder_q.fc[2].fc2.weight.clone().detach()
        else:
            class_weight = encoder_q.fc.fc2.weight.clone().detach()
        return class_weight


class ContrastByClassCalculator(LogitsLabelsCalculator):
    def __init__(self, T, metric, num_classes):
        self.T = T
        self.metric = metric
        self.num_classes = num_classes

    def calculate(self, q_outs, k=None, encoder_q=None, queue=None, cls_labels=None, **kwargs):
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        queue = queue[:self.num_classes, :, :]

        class_weights = self._extract_class_weights(encoder_q)
        class_weights_by_label = class_weights[cls_labels]
        q = self.metric.preprocess(q, class_weights=class_weights_by_label, cls_labels=cls_labels, track=True)
        k = self.metric.preprocess(k, class_weights=class_weights_by_label, cls_labels=cls_labels, track=False)
        queue = self.metric.preprocess(queue, class_weights=class_weights.unsqueeze(2), cls_labels=cls_labels,
                                       track=False)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        labels_onehot = torch.zeros((cls_labels.shape[0], self.num_classes)).cuda().scatter(
            1, cls_labels.unsqueeze(1), 1)
        q_onehot = labels_onehot.unsqueeze(-1) * q.unsqueeze(1)
        l_neg = torch.einsum('ncd,cdk->nk', q_onehot, queue)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels

    # @staticmethod
    # def _extract_class_weights(encoder_q):
    #     if isinstance(encoder_q.fc, Sequential):
    #         class_weight = encoder_q.fc[2].fc2.weight.clone().detach()
    #     else:
    #         class_weight = encoder_q.fc.fc2.weight.clone().detach()
    #     return class_weight

    @staticmethod
    def _extract_class_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            class_weight = encoder_q.fc[2].mu_y.clone().detach()
        else:
            class_weight = encoder_q.fc.mu_y.clone().detach()
        return class_weight


class ContrastBySubclassCalculator(LogitsLabelsCalculator):
    def __init__(self, T):
        self.T = T

    def calculate(self, q_outs, encoder_q=None, subcls_labels=None, **kwargs):
        '''
        :param q_outs: (n, d)
        :param encoder_q:
        :param subcls_labels: (n,)
        :param kwargs:
        :return:
        '''
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        subclass_weights = self._extract_subclass_weights(encoder_q)
        q = nn.functional.normalize(q, p=2, dim=1)
        subclass_weights = nn.functional.normalize(subclass_weights, p=2, dim=1)
        logits = q @ (subclass_weights.t())
        logits = logits / self.T
        return logits, subcls_labels

    @staticmethod
    def _extract_subclass_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            subclass_weight = encoder_q.fc[2].mu_z.clone().detach()
        else:
            subclass_weight = encoder_q.fc.mu_z.clone().detach()
        return subclass_weight


class ContrastBySubclassAngNormCalculator(LogitsLabelsCalculator):
    def __init__(self, T):
        self.T = T

    def calculate(self, q_outs, encoder_q=None, cls_labels=None, subcls_labels=None, **kwargs):
        '''
        :param q_outs: (n, d)
        :param encoder_q:
        :param subcls_labels: (n,)
        :param kwargs:
        :return:
        '''
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        class_weights = self._extract_class_weights(encoder_q)
        class_weights = nn.functional.normalize(class_weights, p=2, dim=1)
        subclass_weights = self._extract_subclass_weights(encoder_q)
        subclass_weights = nn.functional.normalize(subclass_weights, p=2, dim=1)
        class_weights = class_weights[cls_labels]
        q = nn.functional.normalize(q, p=2, dim=1)
        q = nn.functional.normalize(q - class_weights, p=2, dim=1)
        logits = subclass_weights.t().unsqueeze(0) - class_weights.unsqueeze(-1)
        logits = nn.functional.normalize(logits, p=2, dim=1)
        logits = (q.unsqueeze(-1) * logits).sum(1)
        logits = logits / self.T
        return logits, subcls_labels

    @staticmethod
    def _extract_subclass_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            subclass_weight = encoder_q.fc[2].mu_z.clone().detach()
        else:
            subclass_weight = encoder_q.fc.mu_z.clone().detach()
        return subclass_weight


class ContrastTwoClassCalculator(LogitsLabelsCalculator):
    def __init__(self, T):
        self.T = T

    def calculate(self, q_outs, encoder_q=None, cls_labels=None, subcls_labels=None, **kwargs):
        '''
        :param q_outs: (n, d)
        :param encoder_q:
        :param cls_labels: (n,)
        :param kwargs:
        :return:
        '''
        class_weights = self._extract_class_weights(encoder_q)
        class_weights = nn.functional.normalize(class_weights, p=2, dim=1)
        subclass_weights = self._extract_subclass_weights(encoder_q)
        subclass_weights = nn.functional.normalize(subclass_weights, p=2, dim=1)
        subclass_weights = subclass_weights[subcls_labels]
        logits = subclass_weights @ (class_weights.t())
        logits = logits / self.T
        return logits, cls_labels

    @staticmethod
    def _extract_class_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            class_weight = encoder_q.fc[2].mu_y.clone().detach()
        else:
            class_weight = encoder_q.fc.mu_y.clone().detach()
        return class_weight

    @staticmethod
    def _extract_subclass_weights(encoder_q):
        if isinstance(encoder_q.fc, Sequential):
            subclass_weight = encoder_q.fc[2].mu_z.clone().detach()
        else:
            subclass_weight = encoder_q.fc.mu_z.clone().detach()
        return subclass_weight


class Metric(nn.Module):
    def preprocess(self, v):
        raise NotImplementedError


class Norm(Metric):
    def preprocess(self, v, **kwargs):
        v = nn.functional.normalize(v)
        return v


class Angular(Metric):
    def preprocess(self, v, class_weights=None, **kwargs):
        v = nn.functional.normalize(nn.functional.normalize(v) - nn.functional.normalize(class_weights))
        return v
