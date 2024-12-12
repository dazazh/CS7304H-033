import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
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
X_train_pca = pca.transform(X_train_norm)
X_test_pca = pca.transform(X_test_norm)




# 应用 PCA 降维，保留 95% 的方差
pca = PCA(n_components=0.95)  # 保留 95% 的累计方差

X_train = pca.fit_transform(X_train)
print(f"PCA 降维后特征维度: {X_train.shape[1]}")

# 2. 定义随机森林分类器和超参数范围
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [50, 100, 200],          # 森林中树的数量
    'max_depth': [None, 10, 20, 30],         # 每棵树的最大深度
    'min_samples_split': [2, 5, 10],         # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4],           # 叶子节点所需的最小样本数
    'max_features': ['sqrt', 'log2'],        # 每次分裂时的最大特征数量
}

# 3. 使用 GridSearchCV 和 K折交叉验证
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                                    # 使用 5 折交叉验证
    scoring='accuracy',                      # 评价指标：准确率
    # verbose=2,                               # 打印搜索过程
    n_jobs=-1                                # 并行加速
)

grid_search.fit(X_train, y_train)

# 4. 获取最佳超参数和交叉验证得分
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_rf = grid_search.best_estimator_

print("最佳超参数:", best_params)
print(f"交叉验证平均准确率: {best_score:.4f}")

X_test = pca.fit_transform(X_test)

# 5. 使用最佳模型对测试集进行预测
y_pred = best_rf.predict(X_test)

# 6. 保存预测结果，添加 Id 列
output = pd.DataFrame({'Id': range(0, len(y_pred)), 'Label': y_pred})
output.to_csv('res/rf_predictions.csv', index=False)
print("预测结果已保存至 kfold_rf_predictions.csv")
