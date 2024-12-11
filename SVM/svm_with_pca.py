import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载数据
data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/train.csv", header=None)
X = data.iloc[:, :-1].values  # 前512列为特征
y = data.iloc[:, -1].values   # 最后一列为标签

# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 应用 PCA 降维，保留 95% 的方差
pca = PCA(n_components=0.95)  # 保留 95% 的累计方差
X = pca.fit_transform(X)
print(f"PCA 降维后特征维度: {X.shape[1]}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型训练
# 定义SVM模型，使用RBF核
svm = SVC(random_state=42)

# 使用网格搜索调整超参数
# param_grid = {
#     'C': [0.1, 1, 10],
#     'gamma': [0.01, 0.1, 1]
# }
param_grid = {
    'C':[0.01, 0.001],
    'kernel':['linear','poly'], # Kernel type
    'degree':[2,3,4],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳参数和模型
best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")
best_model = grid_search.best_estimator_

# 4. 模型评估
# 在验证集上评估模型
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"验证集准确率: {val_accuracy}")
print("分类报告:\n", classification_report(y_val, y_val_pred))

# 5. 测试数据预测
# 加载测试数据
test_data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/test.csv", header=None)
X_test = test_data.iloc[:, :].values  # 假设测试数据没有标签
X_test = scaler.transform(X_test)  # 标准化
X_test = pca.transform(X_test)  # 应用与训练集相同的 PCA

# 生成预测结果
y_test_pred = best_model.predict(X_test)

# 将预测结果和行号（Id）组合成 DataFrame
result_df = pd.DataFrame({
    'Id': np.arange(0, len(y_test_pred)),  # Id 是测试集的行号
    'Label': y_test_pred  # 将预测结果转换为 numpy 数组
})

# 保存为 CSV 文件
result_df.to_csv('/data2/yuhao/class/CS7304H-033/SVM/svm_predictions.csv', index=False)

print("测试集预测结果已保存为 'svm_predictions.csv'")
