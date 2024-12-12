import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 1. 加载数据
train_data = pd.read_csv('../Datasets/train.csv', header=None)
test_data = pd.read_csv('../Datasets/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values  # 特征
y_train = train_data.iloc[:, -1].values   # 标签
X_test = test_data.values                 # 测试集特征

# 2. 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 确定合理的 k 范围
n_samples = len(X_train)
k_range = range(5, int(n_samples**0.5) + 1, 2)  # 选择奇数 k，避免平局

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

# 6. 保存预测结果，添加 Id 列
output = pd.DataFrame({'Id': range(0, len(y_pred)), 'Label': y_pred})
output.to_csv('knn_predictions.csv', index=False)
print("预测结果已保存至 predictions.csv")
