import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from utils.smooth import local_linear_regression_smooth



# 1. 加载数据
data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/train.csv", header=None)
test_data = pd.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/test.csv", header=None)
X_train = data.iloc[:, :-1].values  # 前512列为特征
y = data.iloc[:, -1].values   # 最后一列为标签
X_test = test_data.iloc[:, :].values  # 假设测试数据没有标签

# 2 数据预处理

# 2.0 归一化
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8  # 防止除零
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
# 2.1 添加高斯噪声
mean = 0.0
std = 0.01  # 根据需要调整标准差大小
k = 100 # sqrt(N)
D = 50 
bandwidth = 0.5

# 生成与X同维度的高斯噪声矩阵
# noise_train = np.random.normal(loc=mean, scale=std, size=X_train.shape)

# # 将噪声加到特征上
# X_train = X_train + noise_train

# 2.2 PCA降维
# pca = PCA(n_components=D)
# pca.fit(X_train)  # 仅用训练集数据进行PCA拟合
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)

# 2.3 smooth
X_train = local_linear_regression_smooth(X_train, X_train, bandwidth,k)
X_test = local_linear_regression_smooth(X_test, X_train, bandwidth,k)

# 3. 模型训练

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)
# 定义SVM模型
svm = SVC(random_state=42)

param_grid = {
    'C':[0.01, 0.001],
    'kernel':['linear','poly'], # Kernel type
    'degree':[2,3,4],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
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
