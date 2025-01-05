# CS7304H-033
统计学习理论与方法

#### 文件结构
CS7304H-033
├── DANN
├── Datasets
├── DecisionTree
├── KNN
├── LogisticRegression
├── RandomForest
├── SVM
├── utils
├── __init__.py
├── requirements.txt
└── README.md


#### 运行环境：
python 3.8
`pip install -r requirements.txt`
经测试，SVM和DANN方法效果相对较好，如需测试可按照下面步骤，输出相应预测文件

#### 数据集：
运行前需按照下列文件夹结构导入训练/测试集
Datasets
├── test.csv
└── train.csv

#### 四种传统模型：
`cd CS7304H-033`
1. Logistic Regression
`python LogisticRegression/lr.py `

2. Support Vector Machine
`python SVM/SVM_with_pca.py`

3. K-Nearest Neighbors
`python KNN/knn_with_pca.py`

4. Decision Tree
时间开销较长
`python DecisionTree/dt.py`

5. Random Forest
`python RandomForest/rf_with_pca.py`

#### 深度学习方法（DANN）
可使用此方式训练，训练结束后会输出测试集的预测结果
`python DANN/train.py`

