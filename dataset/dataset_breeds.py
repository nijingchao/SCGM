import numpy as np
import torch
import pickle as pkl
import os
from torch.utils.data import Dataset
from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping
from sklearn.utils import shuffle


class BREEDSFactory:
    def __init__(self, info_dir, data_dir):
        self.info_dir = info_dir
        self.data_dir = data_dir

    def get_breeds(self, ds_name, partition, transforms=None, split=None):
        superclasses, subclass_split, label_map = self.get_classes(ds_name, split)
        partition = 'val' if partition == 'validation' else partition
        # print(f"==> Preparing dataset {ds_name}, mode: {mode}, partition: {partition}..")
        if split is not None:
            # split can be  'good','bad' or None. if not None, 'subclass_split' will have 2 items, for 'train' and 'test'. otherwise, just 1
            index = 0 if partition == 'train' else 1
            return self.create_dataset(partition, subclass_split[index], transforms)
        else:
            return self.create_dataset(partition, subclass_split[0], transforms)

    def create_dataset(self, partition, subclass_split, transforms):
        coarse_custom_label_mapping = get_label_mapping("custom_imagenet", subclass_split)
        fine_subclass_split = [[item] for sublist in subclass_split for item in sublist]
        fine_custom_label_mapping = get_label_mapping("custom_imagenet", fine_subclass_split)

        active_custom_label_mapping = fine_custom_label_mapping
        active_subclass_split = fine_subclass_split

        # if mode == 'coarse':
        #     active_custom_label_mapping = coarse_custom_label_mapping
        #     active_subclass_split = subclass_split
        # elif mode == 'fine':
        #     active_custom_label_mapping = fine_custom_label_mapping
        #     active_subclass_split = fine_subclass_split
        # else:
        #     raise NotImplementedError

        dataset = folder.ImageFolder(root=os.path.join(self.data_dir, partition), transform=transforms, label_mapping=active_custom_label_mapping)
        coarse2fine, coarse_labels = self.extract_c2f_from_dataset(dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition)
        setattr(dataset, 'num_classes', len(active_subclass_split))
        setattr(dataset, 'coarse2fine', coarse2fine)
        setattr(dataset, 'coarse_targets', coarse_labels)
        return dataset

    def extract_c2f_from_dataset(self, dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition):
        classes, original_classes_to_idx = dataset._find_classes(os.path.join(self.data_dir, partition))
        _, coarse_classes_to_idx = coarse_custom_label_mapping(classes, original_classes_to_idx)
        _, fine_classes_to_idx = fine_custom_label_mapping(classes, original_classes_to_idx)
        coarse2fine = {}
        for k, v in coarse_classes_to_idx.items():
            if v in coarse2fine:
                coarse2fine[v].append(fine_classes_to_idx[k])
            else:
                coarse2fine[v] = [fine_classes_to_idx[k]]

        # modification
        # ---
        fine2coarse = {}
        for k in coarse2fine:
            fine_labels_k = coarse2fine[k]
            for i in range(len(fine_labels_k)):
                assert fine_labels_k[i] not in fine2coarse
                fine2coarse[fine_labels_k[i]] = k

        fine_labels = dataset.targets
        coarse_labels = []
        for i in range(len(fine_labels)):
            coarse_labels.append(fine2coarse[fine_labels[i]])

        return coarse2fine, coarse_labels

    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError


class BREEDS(Dataset):
    def __init__(self, info_dir, data_dir, ds_name, partition, split, transform, train=True, seed=1000, tr_ratio=0.9):
        super(Dataset, self).__init__()
        breeds_factory = BREEDSFactory(info_dir, data_dir)
        self.dataset = breeds_factory.get_breeds(ds_name=ds_name,
                                                 partition=partition,
                                                 transforms=None,
                                                 split=split)

        self.transform = transform
        self.loader = self.dataset.loader

        images = [s[0] for s in self.dataset.samples]
        labels = self.dataset.targets
        coarse_labels = self.dataset.coarse_targets

        if partition == 'train':
            images, labels, coarse_labels = shuffle(images, labels, coarse_labels, random_state=seed)
            num_tr = int(len(images) * tr_ratio)

            if train is True:
                self.images = images[:num_tr]
                self.labels = labels[:num_tr]
                self.coarse_labels = coarse_labels[:num_tr]
            else:
                self.images = images[num_tr:]
                self.labels = labels[num_tr:]
                self.coarse_labels = coarse_labels[num_tr:]
        else:
            self.images = images
            self.labels = labels
            self.coarse_labels = coarse_labels

        # if hasattr(self.dataset, "samples"):
        #     self.images = [s[0] for s in self.dataset.samples]
        # elif hasattr(self.dataset, "images"):
        #     self.images = self.dataset.images
        #
        # if hasattr(self.dataset, "targets"):
        #     self.labels = self.dataset.targets
        # elif hasattr(self.dataset, "labels"):
        #     self.labels = self.dataset.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, coarse_target = self.images[index], self.labels[index], self.coarse_labels[index]

        if self.transform is not None:
            img = self.transform(self.loader(img))

        return img, target, coarse_target, index

    def __len__(self):
        return len(self.images)


class BREEDS_META(Dataset):
    def __init__(self, info_dir, data_dir, ds_name, split, transform_train, transform_test, n_test_runs, n_ways, n_shots, n_queries, n_aug_support_samples=1, fg=False, fix_seed=False):
        super(Dataset, self).__init__()
        breeds_factory = BREEDSFactory(info_dir, data_dir)
        self.dataset = breeds_factory.get_breeds(ds_name=ds_name,
                                                 partition='val',
                                                 transforms=None,
                                                 split=split)
        self.n_test_runs = n_test_runs
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.n_aug_support_samples = n_aug_support_samples
        self.fg = fg
        self.fix_seed = fix_seed

        self.transform_train = transform_train
        self.transform_test = transform_test
        self.loader = self.dataset.loader
        self.coarse2fine = self.dataset.coarse2fine
        self.coarse2fine_keys = list(self.coarse2fine.keys())

        self.images = [s[0] for s in self.dataset.samples]
        self.labels = self.dataset.targets
        self.coarse_labels = self.dataset.coarse_targets
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
            imgs = np.asarray(self.datadic[cls])

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

        support_xs = support_xs.reshape((-1, 1))  # (n_ways * n_shots, 1)
        support_ys = support_ys.reshape((-1,))  # (n_ways * n_shots,)
        query_xs = query_xs.reshape((-1, 1))  # (n_ways * n_shots, 1)
        query_ys = query_ys.reshape((-1,))  # (n_ways * n_queries,)

        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, [self.n_aug_support_samples, 1])  # (n_ways * n_shots * n_aug_support_samples, 1)
            support_ys = np.tile(support_ys, self.n_aug_support_samples)  # (n_ways * n_shots * n_aug_support_samples,)

        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.transform_train(self.loader(x.squeeze(0)[0])), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.transform_test(self.loader(x.squeeze(0)[0])), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


# class BREEDS_PKL(Dataset):
#     def __init__(self, ds_name, partition):
#         super(Dataset, self).__init__()
#         data_path = '/nfs/data/usr/jni/datasets/imagenet_ilsvrc/pklfile/' + ds_name + '_' + partition + '.pkl'
#         with open(data_path, 'rb') as pklfile:
#             dic = pkl.load(pklfile)
#
#         self.images = dic['images_set']
#         self.labels = dic['labels_set']
#         self.labels_coarse = dic['labels_coarse_set']
#
#     def __getitem__(self, index):
#         img, target, target_coarse = self.images[index], self.labels[index], self.labels_coarse[index]
#         return img, target, target_coarse, index
#
#     def __len__(self):
#         return len(self.images)
