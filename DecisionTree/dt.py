import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
data = pd.read_csv("./Datasets/train.csv", header=None)
test_data = pd.read_csv("./Datasets/test.csv", header=None)

X_train = data.iloc[:, :-1].values  # 前512列为特征
y_train = data.iloc[:, -1].values   # 最后一列为标签
X_test = test_data.iloc[:, :].values  # 测试数据没有标签


# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)  # 标准化

# 决策树模型和参数网格
param_grid = {
    'criterion': ['gini', 'entropy'],  # 切分标准
    'max_depth': [10, 20, 30],  # 最大深度
    'min_samples_split': [2, 5, 10],  # 每次切分最小样本数
    'min_samples_leaf': [1, 2, 4],    # 叶子节点的最小样本数
}

# 使用GridSearchCV进行超参数搜索
clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 输出最优参数和交叉验证得分
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# 用最优模型在测试集上预测
best_clf = grid_search.best_estimator_
y_test_pred = best_clf.predict(X_test)

# 将预测结果和行号（Id）组合成 DataFrame
result_df = pd.DataFrame({
    'Id': np.arange(0, len(y_test_pred)),  # Id 是测试集的行号
    'Label': y_test_pred  # 将预测结果转换为 numpy 数组
})

# 保存为 CSV 文件
result_df.to_csv('./DecisionTree/res/dt_predictions.csv', index=False)