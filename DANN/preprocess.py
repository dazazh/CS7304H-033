import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from utils.smooth import local_linear_regression_smooth
from sklearn.metrics import accuracy_score
import torch
from sklearn.decomposition import PCA
def preprocess(data, test_data):
    # ----------------- 原有特征预处理流程 -----------------
    # 1. 加载数据
    X_train = data.iloc[:, :-1].values  # 前512列为特征
    y = data.iloc[:, -1].values         # 最后一列为标签
    X_test = test_data.iloc[:, :].values

    # 2 数据预处理
    # 2.0 归一化
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    k = 100  # 邻居数
    D = 64 
    bandwidth = 0.5

    # 2.1 PCA
    pca = PCA(n_components=D)
    pca.fit(X_train)  # 仅用训练集数据进行PCA拟合
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # 2.2 平滑
    # X_train = local_linear_regression_smooth(X_train, X_train, bandwidth, k)
    # X_test = local_linear_regression_smooth(X_test, X_train, bandwidth, k)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

    # ----------------- 增加DANN模块 -----------------

    # 将数据转换为PyTorch张量
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    print("数据预处理完成")
    
    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t