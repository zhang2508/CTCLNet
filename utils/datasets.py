try:
    from .utils import HyperX, sample_gt
except:
    from utils import HyperX, sample_gt
from torch.utils.data import DataLoader
import numpy as np

def get_target_dataset(img, gt_cls, gt_regression_train, gt_regression_val, train_size_cls, patch=9):
    if len(gt_cls.shape) == 2:
        gt_cls = np.expand_dims(gt_cls, 2)
    if len(gt_regression_train.shape) == 2:
        gt_regression_train = np.expand_dims(gt_regression_train, 2)
    if len(gt_regression_val.shape) == 2:
        gt_regression_val = np.expand_dims(gt_regression_val, 2)

    hyperparams = {"patch_size": patch,
                   "ignored_labels": [0],
                   "flip_augmentation": False,
                   "radiation_augmentation": False,
                   "mixture_augmentation": False,
                   "center_pixel": True,
                   "supervision": "full",
                   "is_3D": False}

    train_gt, test_gt = sample_gt(gt_cls,train_size=train_size_cls)
    train_dataset = HyperX(img, train_gt, **hyperparams)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    test_dataset = HyperX(img, test_gt, **hyperparams)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
    del test_dataset, train_dataset, gt_cls

    train_dataset_regression = HyperX(img, gt_regression_train, **hyperparams)
    train_loader_regression = DataLoader(dataset=train_dataset_regression, batch_size=int(np.sum(gt_regression_train>0)), shuffle=False)

    test_dataset_regression = HyperX(img, gt_regression_val, **hyperparams)
    test_loader_regression = DataLoader(dataset=test_dataset_regression, batch_size=int(np.sum(gt_regression_val>0)), shuffle=False)
    del train_dataset_regression, test_dataset_regression, img

    return train_loader, test_loader, train_loader_regression, test_loader_regression







