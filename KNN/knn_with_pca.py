import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np

k = 5 
D = 50  # 设定降维后的维度

# 1. 加载数据
train_data = pd.read_csv('../Datasets/train.csv', header=None)
test_data = pd.read_csv('../Datasets/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values  # 特征
y_train = train_data.iloc[:, -1].values   # 标签
X_test = test_data.values                 # 测试集特征

# 2. 数据预处理
# 2.1 smooth
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)
distances_train, indices_train = nbrs.kneighbors(X_train)
X_train_smoothed = np.zeros_like(X_train)
for i in range(X_train.shape[0]):
    # indices_train[i][0] 是样本自身，可以选择包含或不包含自身
    neighbor_ids = indices_train[i][1:]  # 去掉自身，只平均邻居
    X_train_smoothed[i] = np.mean(X_train[neighbor_ids], axis=0)
distances_test, indices_test = nbrs.kneighbors(X_test)
X_test_smoothed = np.zeros_like(X_test)
for i in range(X_test.shape[0]):
    neighbor_ids = indices_test[i][1:]  # 去掉自身（其实测试样本本身不在训练集中，indices_test的第一个邻居是最接近的训练样本）
    X_test_smoothed[i] = np.mean(X_train[neighbor_ids], axis=0)

# 2.2 Normalization
mean = np.mean(X_train_smoothed, axis=0)
std = np.std(X_train_smoothed, axis=0) + 1e-8  # 防止除零
X_train_norm = (X_train_smoothed - mean) / std
X_test_norm = (X_test_smoothed - mean) / std

# 2.3 PCA降维
pca = PCA(n_components=D)
pca.fit(X_train_norm)  # 仅用训练集数据进行PCA拟合
X_train = pca.transform(X_train_norm)
X_test = pca.transform(X_test_norm)


# 3. 确定合理的 k 范围
n_samples = len(X_train)
k_range = range(1, int(n_samples**0.5) + 1, 2)  # 选择奇数 k，避免平局

# 4. 交叉验证选择最佳 k
best_k = 1
best_score = 0
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # 使用 CPU 并行加速
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # 5 折交叉验证
    if scores.mean() > best_score:
        best_k = k
        best_score = scores.mean()

print(f"最佳 k 值: {best_k}, 交叉验证准确率: {best_score:.4f}")

# 5. 用最佳 k 值训练模型并预测测试集
knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 6. 保存预测结果
output = pd.DataFrame({'Id': range(0, len(y_pred)), 'Label': y_pred})
output.to_csv('knn_predictions.csv', index=False)
print("预测结果已保存至 knn_predictions.csv")
