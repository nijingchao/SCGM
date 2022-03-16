import os
import os
import pickle
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from sklearn.utils import shuffle


class CIFAR100(VisionDataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 val=False, seed=1000, tr_ratio=0.9):
        super(CIFAR100, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.val = val

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.images = []
        self.labels = []
        self.coarse_labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.images.append(entry['data'])
                self.labels.extend(entry['fine_labels'])
                self.coarse_labels.extend(entry['coarse_labels'])

                self.coarse2fine = {}
                for cl, fl in zip(entry['coarse_labels'], entry['fine_labels']):
                    if cl in self.coarse2fine:
                        if fl not in self.coarse2fine[cl]:
                            self.coarse2fine[cl].append(fl)
                    else:
                        self.coarse2fine[cl] = [fl]

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        # split training data into training and validation sets
        if self.train is True:
            images, labels, coarse_labels = shuffle(self.images, self.labels, self.coarse_labels, random_state=seed)
            num_tr = int(len(images) * tr_ratio)

            if self.val is True:
                self.images = images[num_tr:]
                self.labels = labels[num_tr:]
                self.coarse_labels = coarse_labels[num_tr:]
            else:
                self.images = images[:num_tr]
                self.labels = labels[:num_tr]
                self.coarse_labels = coarse_labels[:num_tr]

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    @property
    def num_classes(self):
        return len(np.unique(self.labels))

    @property
    def num_coarse_classes(self):
        return len(np.unique(self.coarse_labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, coarse_target = self.images[index], self.labels[index], self.coarse_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, coarse_target, index

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100_META(CIFAR100):
    def __init__(self, data_root, n_test_runs, n_ways, n_shots, n_queries, n_aug_support_samples=1, meta_train=False, meta_val=False, transform_train=None, transform_test=None, fg=False, fix_seed=True):
        super(CIFAR100_META, self).__init__(root=data_root, train=meta_train, val=meta_val)
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
        if self.fix_seed is True:
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
            imgs = np.asarray(self.datadic[cls]).astype('uint8')

            support_xs_ids_sampled = np.random.choice(np.arange(imgs.shape[0]), self.n_shots, replace=False)
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

        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        support_xs = support_xs.reshape((-1, height, width, channel))  # (n_ways * n_shots, h, w, c)
        support_ys = support_ys.reshape((-1,))  # (n_ways * n_shots,)
        query_xs = query_xs.reshape((-1, height, width, channel))  # (n_ways * n_queries, h, w, c)
        query_ys = query_ys.reshape((-1,))  # (n_ways * n_queries,)

        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, [self.n_aug_support_samples, 1, 1, 1])  # (n_ways * n_shots * n_aug_support_samples, h, w, c)
            support_ys = np.tile(support_ys, self.n_aug_support_samples)  # (n_ways * n_shots * n_aug_support_samples,)

        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.transform_train(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.transform_test(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs
