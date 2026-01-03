import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time
import random
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from utils import datasets
from model import mymodels as models

import os
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=10000)
import itertools
from typing import Dict, Tuple
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def Machine_Learning(ml_name):
    if ml_name == "PLSR":
        ml_model = PLSRegression(n_components=2)
        param_grid = {
            'n_components': list(range(1, 20))  # 1 ~ 20
        }
    elif ml_name == "SVR":
        ml_model = SVR(kernel='linear', C=100)
        param_grid = {
            'kernel': ["rbf", "linear", "poly", "sigmoid"],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto'],
        }
    elif ml_name == "DT":
        ml_model = DecisionTreeRegressor(random_state=342)
        param_grid = {
            'max_depth': [None, 3, 5, 10, 20, 50],
        }
    elif ml_name == "RF":
        ml_model = RandomForestRegressor(n_estimators=100, random_state=342)
        param_grid = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 3, 5, 10, 20, 50],
        }
    elif ml_name == "XGBoost":
        ml_model = XGBRegressor(n_estimators=100, random_state=342)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
        }

    grid_search = GridSearchCV(
        estimator=ml_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        return_train_score=True
    )

    return grid_search

def model_sets():
    model = models.HybridMambaCNN(FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH)
    return model


def process_features_torch(
        all_features: torch.Tensor,
        all_pixel_labels: torch.Tensor,
        label_to_y_map: Dict[int, float],
        n_bins: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = all_features.device
    unique_labels = torch.unique(all_pixel_labels)

    y_values_for_samples = torch.tensor(
        [label_to_y_map[label.item()] for label in unique_labels],
        dtype=torch.float32, device=device
    )

    if y_values_for_samples.numel() > 0:
        y_np = y_values_for_samples.cpu().numpy()
        stratified_labels_for_samples_np = pd.cut(y_np, bins=n_bins, labels=False, duplicates='drop')
        stratified_labels_for_samples = torch.from_numpy(stratified_labels_for_samples_np).to(device)
    else:
        stratified_labels_for_samples = torch.tensor([], dtype=torch.long, device=device)

    aggregated_features_list = []
    for label_val in unique_labels:
        mask = (all_pixel_labels == label_val)
        mean_feature = torch.mean(all_features[mask], dim=0)
        aggregated_features_list.append(mean_feature)

    aggregated_features = torch.stack(aggregated_features_list, dim=0)
    aggregated_y = y_values_for_samples
    aggregated_labels = unique_labels
    stratified_labels = stratified_labels_for_samples
    return aggregated_features, aggregated_y, aggregated_labels, stratified_labels

EXCEL_FILE_PATH = r'./data/data.xlsx'
df = pd.read_excel(EXCEL_FILE_PATH, header=0, sheet_name="生物量")
original_labels_np = df['label'].values - 1
original_y_np = df['生物量'].values
label_to_y_map_dict = {int(label): y for label, y in zip(original_labels_np, original_y_np)}

def train():
    train_loader, test_loader, train_loader_regression, test_loader_regression = datasets.get_target_dataset(img, gt,
                                patch=PATCH,
                                gt_regression_train=gt_regression_train, gt_regression_val=gt_regression_val,
                                train_size_cls=TEST_LSAMPLE_NUM_PER_CLASS)

    feature_encoder = model
    classifier = nn.Linear(FEATURE_DIM, CLASS_NUM) # 分类
    regressor = nn.Linear(FEATURE_DIM, 5) # 回归

    feature_encoder.cuda()
    classifier.cuda()
    regressor.cuda()

    feature_encoder.train()
    classifier.train()
    regressor.train()

    optimizer = torch.optim.Adam(itertools.chain(feature_encoder.parameters(),
                                                 classifier.parameters(),
                                                 regressor.parameters()), lr=args.learning_rate)
    print("开始训练...")

    best_r2 = 0.0
    best_episdoe = 0
    total_loss_list_cls = []
    total_loss_list_regressor = []
    total_loss = []
    train_episode_cls = []
    train_episode_regressor = []
    train_acc_list = []
    R2 = []
    test_episode = []
    total_hit, total_num = 0.0, 0.0

    train_iter_cls = iter(train_loader)
    train_iter_regression = iter(train_loader_regression)
    for episode in range(EPISODE):
        try:
            train_data_cls, train_label_cls = next(train_iter_cls)
        except StopIteration:
            train_iter_cls = iter(train_loader)
            train_data_cls, train_label_cls = next(train_iter_cls)
        try:
            train_data_regression, train_label_regression = next(train_iter_regression)
        except StopIteration:
            train_iter_regression = iter(train_loader_regression)
            train_data_regression, train_label_regression = next(train_iter_regression)

        # ==============================================
        features = feature_encoder(train_data_cls.cuda())
        outputs = classifier(features)
        loss_cls = crossEntropy(outputs, train_label_cls.cuda().long())

        total_hit += torch.sum(torch.argmax(outputs, dim=1).cpu() == train_label_cls).item()
        total_num += train_data_cls.shape[0]

        total_loss_list_cls.append(loss_cls.item())
        train_episode_cls.append(episode+1)
        # ==============================================

        # ==============================================
        features = feature_encoder(train_data_regression.cuda())
        X_aggregated, Y_aggregated, Labels_aggregated,stratified_labels = process_features_torch(
            all_features=features,
            all_pixel_labels=train_label_regression.cuda(),
            label_to_y_map=label_to_y_map_dict,
            n_bins=5
        )

        outputs = regressor(X_aggregated)
        loss_regressor = crossEntropy(outputs, stratified_labels.cuda().long())

        total_loss_list_regressor.append(loss_regressor.item())
        train_episode_regressor.append(episode+1)
        # ==============================================

        loss = loss_cls*(1-args.Lambda) + loss_regressor*args.Lambda
        total_loss.append(loss.item())

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'分类 train episode {episode+1:>4d}: '
              f'loss_cls: {loss_cls.item():.4f}, '
              f'acc {(total_hit / total_num)*100:.2f} '
              f'loss_regressor: {loss_regressor.item():.4f}')
        train_acc_list.append(total_hit / total_num)

        if (episode + 1) % TEST_EPISODE == 0:
            feature_encoder.eval()

            train_data_regression, train_label_regression = next(iter(train_loader_regression))
            features = feature_encoder(Variable(train_data_regression).cuda())

            X_aggregated_train, Y_aggregated_train, Labels_aggregated_train,_ = process_features_torch(
                all_features=features,
                all_pixel_labels=train_label_regression.cuda(),
                label_to_y_map=label_to_y_map_dict,
            )

            max_value = X_aggregated_train.max()
            min_value = X_aggregated_train.min()
            train_features_norm = (X_aggregated_train - min_value) * 1.0 / (max_value - min_value)
            ml_model.fit(train_features_norm.cpu().detach().numpy(), Y_aggregated_train.cpu().detach().numpy())
            model_regression = ml_model.best_estimator_
            test_datas_regression, test_labels_regression = next(iter(test_loader_regression))
            features = feature_encoder(Variable(test_datas_regression).cuda())

            X_aggregated_test, Y_aggregated_test, Labels_aggregated_test,_ = process_features_torch(
                all_features=features,
                all_pixel_labels=test_labels_regression.cuda(),
                label_to_y_map=label_to_y_map_dict,
            )
            test_features = (X_aggregated - min_value) * 1.0 / (max_value - min_value)
            y_pred = model_regression.predict(test_features.cpu().detach().numpy())

            results = calculate_regression_metrics(Y_aggregated_test.cpu().detach().numpy(), y_pred)
            print(f"test episode: {episode + 1:>4d} MSE: {results['MSE']:.4f} RMSE: {results['RMSE']:.4f}"
                  f" R2: {results['R2']:.4f} MAE: {results['MAE']:.4f}")

            R2.append(results['R2'])
            test_episode.append(episode+1)

            # Training mode
            feature_encoder.train()
            if results['R2'] > best_r2:
                # save networks
                torch.save(feature_encoder.state_dict(), str("feature_encoder_" + ".pkl"))
                print("save networks for episode: ", episode + 1)

                best_r2 = results['R2']
            print(f'best episode: {best_episdoe + 1}, best r2={best_r2:.4f} \n\n')

if __name__ == '__main__':
    from utils.utils import same_seeds,calculate_regression_metrics
    from sklearn import preprocessing

    # -----------------------------------------参数定义------------------------------------
    parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
    parser.add_argument("-f", "--feature_dim", type=int, default=64)
    parser.add_argument("-d", "--tar_input_dim", type=int, default=480)
    parser.add_argument("-w", "--class_num", type=int, default=13)
    parser.add_argument("-e", "--episode", type=int, default=500)
    parser.add_argument("-t", "--test_episode", type=int, default=5)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-g", "--gpu", type=int, default=0)

    parser.add_argument("-p", "--patch", type=int, default=9)
    parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=1.0)
    parser.add_argument("-Lambda", "--Lambda", type=float, default=0.3)
    parser.add_argument("-ML", "--ML", type=str, default="RF")
    args = parser.parse_args()

    FEATURE_DIM = args.feature_dim
    TAR_INPUT_DIMENSION = args.tar_input_dim
    CLASS_NUM = args.class_num
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    PATCH = args.patch

    TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class
    same_seeds(0)

    from osgeo import gdal
    img = gdal.Open(r"data.dat").ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    h,w,c = img.shape
    mask = np.all(img == 0, axis=-1)
    img_ = preprocessing.scale(img[~mask])
    img[~mask] = img_
    del img_

    has_nan = np.isnan(img).any()
    print("数组中是否有缺失值:", has_nan)
    gt = gdal.Open(r"labelr7.dat").ReadAsArray()
    gt_regression_train = gdal.Open(r"train_labels.tif").ReadAsArray()
    gt_regression_val = gdal.Open(r"validation_labels.tif").ReadAsArray()

    crossEntropy = nn.CrossEntropyLoss().cuda()  # loss

    model = model_sets()
    ml_model = Machine_Learning(args.ML)
    train()

