from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd

# 假设已经有数据特征和标签
train_data = pd.read_csv('../Datasets/train.csv', header=None)
test_data = pd.read_csv('../Datasets/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values  # 特征
y_train = train_data.iloc[:, -1].values   # 标签
X_test = test_data.values                 # 测试集特征

# 初始化Logistic回归模型
# C为正则化强度的倒数，C越小正则化越强，可根据需要调整
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

y_pred = model.predict(X_train)

# 计算准确率
acc = accuracy_score(y_train, y_pred)
print("测试集准确率:", acc)

# 获得概率输出
y_proba = model.predict_proba(X_train)  # 返回的是每一类的概率分布
# 计算log_loss
loss = log_loss(y_train, y_proba)
print("测试集log_loss:", loss)

# 在测试集上预测标签
y_pred = model.predict(X_test)

# 如需查看模型参数，可打印model.coef_和model.intercept_
print("模型权重:", model.coef_)
print("模型偏置:", model.intercept_)

output = pd.DataFrame({'Id': range(0, len(y_pred)), 'Label': y_pred})
output.to_csv('res/lr_predictions.csv', index=False)
print("预测结果已保存至 lr_predictions.csv")



