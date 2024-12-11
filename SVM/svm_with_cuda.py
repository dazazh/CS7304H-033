import cuml
import cudf
import pandas as pd
from cuml.svm import SVC as cumlSVC
from sklearn.model_selection import train_test_split
from cuml.preprocessing import StandardScaler as cumlScaler

# 1. 加载数据
# 使用 cuDF 来读取 CSV 数据文件
data = cudf.read_csv("/data2/yuhao/class/CS7304H-033/Datasets/train.csv", header=None)

# 2. 提取特征和标签
X = data.iloc[:10, :-1]  # 特征列（前512列）
y = data.iloc[:10, -1]   # 标签列（最后一列）

# 3. 数据标准化
# 使用 cuML 的 StandardScaler 进行标准化
scaler = cumlScaler()
X = scaler.fit_transform(X)

# 4. 切分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化 cuML SVM 模型
svm = cumlSVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# 6. 训练模型
svm.fit(X_train, y_train)

# 7. 使用模型进行预测
y_val_pred = svm.predict(X_val)

# 8. 评估准确率
accuracy = (y_val_pred == y_val).mean()
print(f"验证集准确率: {accuracy}")
