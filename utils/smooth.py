import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import Parallel, delayed
import numpy as np

# ---------------------
train_data = pd.read_csv('Datasets/train.csv', header=None)
test_data = pd.read_csv('Datasets/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values  # 特征
y_train = train_data.iloc[:, -1].values   # 标签
X_test = test_data.values                 # 测试集特征

# 参数设置
k = 10  # 近邻数量，可根据数据量调整
bandwidth = 0.5  # 高斯核的带宽(根据特征缩放情况调整)

# 建立近邻搜索结构
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)

def local_linear_regression_smooth(X, X_ref, bandwidth, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_ref)
    distances, indices = nbrs.kneighbors(X)

    def process_point(i):
        x0 = X[i]
        neigh_ids = indices[i]
        neigh_points = X_ref[neigh_ids]  # k x D
        
        # 计算权重
        dists = distances[i]  # k维
        weights = np.exp(-0.5 * (dists**2) / (bandwidth**2))  # 高斯核权重
        W = np.diag(weights)  # k x k

        # 构建设计矩阵Z = [1, (x_n - x0)]
        diff = neigh_points - x0  # k x D
        ones = np.ones((k, 1))
        Z = np.hstack([ones, diff])  # k x (D+1)

        # 要求解 Beta: (Z^T W Z) Beta = Z^T W X_neigh
        ZTWZ = Z.T @ W @ Z  # (D+1) x (D+1)
        ZTWX = Z.T @ W @ neigh_points  # (D+1) x D

        Beta = np.linalg.pinv(ZTWZ) @ ZTWX  # (D+1) x D
        return Beta[0, :]  # 截距项

    # 并行化
    X_smooth = Parallel(n_jobs=-1)(delayed(process_point)(i) for i in range(X.shape[0]))
    return np.array(X_smooth)

if __name__ == '__main__':
    
    # 对训练集进行平滑
    X_train_smooth = local_linear_regression_smooth(X_train, X_train, bandwidth)

    # 对测试集进行平滑 (注意使用训练集构建的nbrs和同样的bandwidth)
    X_test_smooth = local_linear_regression_smooth(X_test, X_train, bandwidth)

    # 使用平滑后的特征训练分类器 (Logistic Regression)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train_smooth, y_train)

    # 在测试集评估
    y_pred = clf.predict(X_test_smooth)

# 会引起矩阵不可逆