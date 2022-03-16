import io
import os
import pickle
import numpy as np
import torch
from PIL import Image
from learn2learn.vision.datasets import TieredImagenet


class TieredImageNet(TieredImagenet):
    def __init__(self, root, partition="train", transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        tiered_imaganet_path = os.path.join(self.root, 'tiered-imagenet')
        short_partition = 'val' if partition == 'validation' else partition
        labels_path = os.path.join(tiered_imaganet_path, short_partition + '_labels.pkl')
        images_path = os.path.join(tiered_imaganet_path, short_partition + '_images_png.pkl')

        with open(images_path, 'rb') as images_file:
            images = pickle.load(images_file)

        with open(labels_path, 'rb') as labels_file:
            labels = pickle.load(labels_file)
            self.coarse2fine = {}
            for c, f in zip(labels['label_general'], labels['label_specific']):
                if c in self.coarse2fine:
                    if f not in self.coarse2fine[c]:
                        self.coarse2fine[c].append(f)
                else:
                    self.coarse2fine[c] = [f]

            coarse_labels = labels['label_general']
            labels = labels['label_specific']

        self.images = images
        self.labels = labels
        self.coarse_labels = coarse_labels

    @property
    def num_classes(self):
        return len(np.unique(self.labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(io.BytesIO(self.images[index]))
        target = self.labels[index]
        coarse_target = self.coarse_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            coarse_target = self.target_transform(coarse_target)

        return img, target, coarse_target, index

    def __len__(self):
        return len(self.images)


class TieredImageNet_Meta(TieredImageNet):
    def __init__(self, data_root, n_test_runs, n_ways, n_shots, n_queries, n_aug_support_samples=1, partition='train', transform_train=None, transform_test=None, fg=False, fix_seed=True):
        super(TieredImageNet_Meta, self).__init__(root=data_root, partition=partition)
        self.n_test_runs = n_test_runs
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.n_aug_support_samples = n_aug_support_samples
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.fg = fg
        self.fix_seed = fix_seed
        self.coarse2fine_keys = list(self.coarse2fine.keys())

        self.datadic = {}
        for idx in range(len(self.images)):
            if self.labels[idx] not in self.datadic:
                self.datadic[self.labels[idx]] = []

            self.datadic[self.labels[idx]].append(self.images[idx])

    def __getitem__(self, index):
        if self.fix_seed:
            np.random.seed(index)

        if self.fg is True:
            classes = self.coarse2fine[np.random.choice(self.coarse2fine_keys, 1, False)[0]]
        else:
            # classes = list(self.datadic.keys())
            classes = sorted(list(self.datadic.keys()))

        if len(classes) > self.n_ways:
            cls_sampled = np.random.choice(classes, self.n_ways, replace=False)
        else:
            print('n_ways={:d},'.format(self.n_ways),
                  'n_class={:d}'.format(len(classes)))
            cls_sampled = np.array(classes) if not isinstance(classes, np.ndarray) else classes

        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []

        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.datadic[cls], dtype=object)

            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, replace=False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)

            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, replace=False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * self.n_queries)

        support_xs = np.array(support_xs)
        support_ys = np.array(support_ys)
        query_xs = np.array(query_xs)
        query_ys = np.array(query_ys)

        support_xs = support_xs.reshape(-1)
        support_ys = support_ys.reshape(-1)
        query_xs = query_xs.reshape(-1)
        query_ys = query_ys.reshape(-1)

        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, [self.n_aug_support_samples])
            support_ys = np.tile(support_ys, self.n_aug_support_samples)

        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.transform_train(self._load_png_byte(x[0])), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.transform_test(self._load_png_byte(x[0])), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def _load_png_byte(self, bytes):
        return Image.open(io.BytesIO(bytes))

    def __len__(self):
        return self.n_test_runs
