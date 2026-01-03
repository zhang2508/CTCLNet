from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from typing import Callable, Tuple, List, Union, Optional
from collections import Counter
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import random
import rasterio
import torch.nn as nn
import os
from datetime import datetime
from sklearn.decomposition import PCA

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    explained_var = pca.explained_variance_ratio_
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    matrix_1 = np.round(cm/np.sum(cm,axis=1)[:, np.newaxis],4)
    per_acc = np.diag(cm)/np.sum(cm,axis=1)
    PA = np.diag(matrix_1)
    UA = np.diag(np.round(cm / np.sum(cm, axis=0)[:, np.newaxis], 4))

    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    OA = total_correct / total_samples

    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    AA = np.nanmean(per_class_accuracy)

    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total_samples ** 2)
    Kappa = (OA - pe) / (1 - pe) if (1 - pe) != 0 else 0

    return OA, AA, Kappa, per_acc, PA, UA, cm, matrix_1

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]), dtype=np.float32)
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt=None,is_shuffle=True, **hyperparams):
        super(HyperX, self).__init__()
        if gt is None:
            H,W,_ = data.shape
            mask = np.all(data == 0, axis=-1)
            gt = np.zeros((H, W, 1))
            gt[~mask] = 1

        data = padWithZeros(data,int((hyperparams['patch_size'] - 1) / 2))
        gt = padWithZeros(gt,int((hyperparams['patch_size'] - 1) / 2))
        gt = np.squeeze(gt)

        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        self.model_3DCNN = hyperparams['is_3D']
        supervision = hyperparams['supervision']
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x >= p and x < data.shape[0] - p and y >= p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        if is_shuffle:
            np.random.shuffle(self.indices)
        else:
            pass

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            # data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1 and self.model_3DCNN:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label -1

from sklearn import model_selection
import sklearn
def sample_gt(gt, train_size, mode='random',train_size_second=15,tsne_confid=False):
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    # if mode == 'random':
    if train_size<1:
        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=345)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices[0],train_indices[1]] = gt[train_indices[0],train_indices[1]]
        test_gt[test_indices[0],test_indices[1]] = gt[test_indices[0],test_indices[1]]

    # elif mode == 'fixed':
    elif train_size>1:
        print("Sampling {} with train size = {}".format(mode, train_size))
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            lon_ = len(X)
            if train_size<lon_:
                train_size_new = train_size
            else:
                train_size_new = lon_-1 if tsne_confid else train_size_second
            train, test = model_selection.train_test_split(X, train_size=train_size_new, random_state=345)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices[0],train_indices[1]] = gt[train_indices[0],train_indices[1]]
        test_gt[test_indices[0],test_indices[1]] = gt[test_indices[0],test_indices[1]]
        # train_gt[train_indices] = gt[train_indices]
        # test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def calculate_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MAPE': mape
    }

    return metrics

