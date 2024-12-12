import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# 1. 加载数据
train_data = pd.read_csv('../Datasets/train.csv', header=None)
test_data = pd.read_csv('../Datasets/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values  # 特征
y_train = train_data.iloc[:, -1].values   # 标签
X_test = test_data.values                 # 测试集特征

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

# 5. 使用最佳模型对测试集进行预测
y_pred = best_rf.predict(X_test)

# 6. 保存预测结果，添加 Id 列
output = pd.DataFrame({'Id': range(0, len(y_pred)), 'Label': y_pred})
output.to_csv('res/rf_predictions.csv', index=False)
print("预测结果已保存至 kfold_rf_predictions.csv")
