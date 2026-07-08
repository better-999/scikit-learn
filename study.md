# scikit-learn 学习课程

## 课程概述
本课程将带你从零开始深入学习 scikit-learn 机器学习库，通过理论学习和实践练习，掌握机器学习的核心概念和实际应用。

## 学习目标
- 理解机器学习的基本概念和分类
- 掌握 scikit-learn 的核心 API 和使用方法
- 能够独立完成分类、回归、聚类等常见任务
- 理解模型评估和调优的方法
- 具备解决实际机器学习问题的能力

## 课程大纲

### 第一阶段：基础入门 (预计 2-3 天)

#### 第1课：机器学习基础概念
**学习目标**
- 理解什么是机器学习
- 了解机器学习的分类（监督学习、无监督学习、半监督学习、强化学习）
- 掌握基本术语：特征、标签、训练集、测试集、模型

**学习内容**
- 机器学习定义和应用场景
- 监督学习 vs 无监督学习
- 分类问题 vs 回归问题
- 过拟合与欠拟合

**实践任务**
- 阅读 todo.md 中的应用场景部分
- 思考 3 个可以用机器学习解决的实际问题

**检查点**
- 能解释监督学习和无监督学习的区别
- 能说出至少 3 个机器学习应用场景

---

#### 第2课：环境搭建与项目结构
**学习目标**
- 成功搭建 scikit-learn 开发环境
- 理解项目结构
- 学会使用 VSCode 调试

**学习内容**
- 按照 todo.md 中的步骤搭建环境
- 理解虚拟环境的作用
- 配置 VSCode Python 解释器
- 运行第一个示例

**实践任务**
- 完成环境搭建
- 运行 `examples/classification/plot_digits_classification.py`
- 尝试修改代码参数，观察结果变化

**检查点**
- 环境搭建成功，能正常运行示例代码
- 理解虚拟环境的作用

---

#### 第3课：数据加载与探索
**学习目标**
- 掌握 scikit-learn 数据集的使用
- 学会数据基本探索
- 理解数据预处理的重要性

**学习内容**
- 使用 sklearn.datasets 加载数据
- 数据集结构：features、target、DESCR
- 数据可视化基础
- 数据统计特征

**实践任务**
```python
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 转换为 DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 数据探索
print(df.head())
print(df.describe())
print(df.info())
```

**检查点**
- 能独立加载和探索数据集
- 理解特征和标签的概念

---

### 第二阶段：监督学习 - 分类 (预计 3-4 天)

#### 第4课：K近邻分类器
**学习目标**
- 理解 KNN 算法原理
- 掌握 KNN 的使用方法
- 理解 K 值选择的影响

**学习内容**
- KNN 算法原理
- 距离度量方法
- K 值的选择
- 优缺点分析

**源码位置**
- `sklearn/neighbors/_classification.py`

**实践任务**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测和评估
y_pred = knn.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred)}")

# 尝试不同的 K 值
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print(f"K={k}, 准确率: {score}")
```

**检查点**
- 理解 KNN 算法原理
- 能使用 KNN 完成分类任务
- 理解 K 值对结果的影响

---

#### 第5课：逻辑回归
**学习目标**
- 理解逻辑回归原理
- 掌握逻辑回归的使用
- 理解 Sigmoid 函数和概率输出

**学习内容**
- 逻辑回归原理
- Sigmoid 函数
- 损失函数
- 正则化（L1、L2）

**源码位置**
- `sklearn/linear_model/_logistic.py`

**实践任务**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# 预测和评估
y_pred = lr.predict(X_test)
print("分类报告:")
print(classification_report(y_test, y_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 查看概率
y_proba = lr.predict_proba(X_test)
print(f"前5个样本的概率预测:\n{y_proba[:5]}")
```

**检查点**
- 理解逻辑回归原理
- 能解释分类报告和混淆矩阵
- 理解概率输出的含义

---

#### 第6课：支持向量机 (SVM)
**学习目标**
- 理解 SVM 原理
- 掌握核函数的使用
- 理解超参数 C 和 gamma

**学习内容**
- SVM 基本原理
- 核函数（线性、多项式、RBF）
- 超参数调优
- 优缺点分析

**源码位置**
- `sklearn/svm/_classes.py`

**实践任务**
```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成数据
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 尝试不同的核函数
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    print(f"{kernel} 核函数准确率: {score}")

# 超参数调优
for C in [0.1, 1, 10]:
    for gamma in ['scale', 'auto']:
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        print(f"C={C}, gamma={gamma}, 准确率: {score}")
```

**检查点**
- 理解 SVM 原理和核函数作用
- 能进行基本的超参数调优
- 理解数据标准化的重要性

---

#### 第7课：决策树
**学习目标**
- 理解决策树原理
- 掌握决策树的使用和可视化
- 理解决策树的优缺点

**学习内容**
- 决策树原理
- 信息增益和基尼系数
- 剪枝策略
- 树的可视化

**源码位置**
- `sklearn/tree/_classes.py`
- `sklearn/tree/_export.py`

**实践任务**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 训练模型
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# 文本形式导出
print("决策树文本表示:")
print(export_text(dt, feature_names=iris.feature_names))

# 可视化
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.savefig('decision_tree.png')

# 特征重要性
print("\n特征重要性:")
for name, importance in zip(iris.feature_names, dt.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

**检查点**
- 理解决策树原理
- 能解释决策树的决策过程
- 理解特征重要性的含义

---

#### 第8课：随机森林
**学习目标**
- 理解集成学习概念
- 掌握随机森林的使用
- 理解 bagging 原理

**学习内容**
- 集成学习原理
- Bagging 和 Boosting
- 随机森林算法
- 特征重要性

**源码位置**
- `sklearn/ensemble/_forest.py`

**实践任务**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 评估
score = rf.score(X_test, y_test)
print(f"随机森林准确率: {score}")

# 特征重要性
print("\n前10个最重要的特征:")
indices = rf.feature_importances_.argsort()[::-1][:10]
for idx in indices:
    print(f"{data.feature_names[idx]}: {rf.feature_importances_[idx]:.4f}")

# 比较不同树的数量
for n_trees in [10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    print(f"树数量={n_trees}, 准确率={score}")
```

**检查点**
- 理解集成学习原理
- 理解随机森林相对于单棵决策树的优势
- 能解释特征重要性

---

### 第三阶段：监督学习 - 回归 (预计 2-3 天)

#### 第9课：线性回归
**学习目标**
- 理解线性回归原理
- 掌握线性回归的使用
- 理解回归评估指标

**学习内容**
- 线性回归原理
- 最小二乘法
- 回归评估指标（MSE、MAE、R²）
- 多项式回归

**源码位置**
- `sklearn/linear_model/_base.py`

**实践任务**
```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 生成回归数据
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)

# 评估
print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² 分数: {r2_score(y_test, y_pred):.4f}")
print(f"斜率: {lr.coef_[0]:.2f}")
print(f"截距: {lr.intercept_:.2f}")
```

**检查点**
- 理解线性回归原理
- 能解释回归评估指标
- 理解模型参数的含义

---

#### 第10课：岭回归和 Lasso 回归
**学习目标**
- 理解正则化概念
- 掌握岭回归和 Lasso 的使用
- 理解 L1 和 L2 正则化的区别

**学习内容**
- 过拟合问题
- L1 正则化（Lasso）
- L2 正则化（Ridge）
- Elastic Net

**源码位置**
- `sklearn/linear_model/_ridge.py`
- `sklearn/linear_model/_coordinate_descent.py`

**实践任务**
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成数据
X, y = make_regression(n_samples=1000, n_features=20, n_informative=5, noise=10, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 比较不同模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Ridge (alpha=10.0)': Ridge(alpha=10.0),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: R² = {score:.4f}")

# 查看 Lasso 的特征选择效果
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
print(f"\nLasso 选择的特征数量: {sum(lasso.coef_ != 0)}")
print(f"Lasso 系数: {lasso.coef_}")
```

**检查点**
- 理解正则化的作用
- 理解 L1 和 L2 正则化的区别
- 能根据需求选择合适的正则化方法

---

### 第四阶段：无监督学习 (预计 2-3 天)

#### 第11课：K均值聚类
**学习目标**
- 理解聚类概念
- 掌握 KMeans 算法
- 学会选择最佳的 K 值

**学习内容**
- 算法原理
- K 值选择方法（肘部法则、轮廓系数）
- 聚类评估
- 优缺点分析

**源码位置**
- `sklearn/cluster/_kmeans.py`

**实践任务**
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 生成聚类数据
X, y = make_blobs(n_samples=500, centers=4, random_state=42)

# 肘部法则选择 K 值
inertias = []
K_range = range(1, 11)
for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K 值')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.savefig('elbow_method.png')

# 轮廓系数
silhouette_scores = []
for K in range(2, 11):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={K}, 轮廓系数={score:.4f}")

# 使用最佳 K 值聚类
best_K = 4
kmeans = KMeans(n_clusters=best_K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print(f"\n聚类中心:\n{centers}")
```

**检查点**
- 理解 KMeans 算法原理
- 能使用肘部法则和轮廓系数选择 K 值
- 理解聚类评估方法

---

#### 第12课：层次聚类和 DBSCAN
**学习目标**
- 掌握层次聚类方法
- 理解 DBSCAN 密度聚类
- 比较不同聚类算法

**学习内容**
- 层次聚类原理
- DBSCAN 算法
- 密度聚类 vs 划分聚类
- 聚类算法选择

**源码位置**
- `sklearn/cluster/_agglomerative.py`
- `sklearn/cluster/_dbscan.py`

**实践任务**
```python
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons, make_blobs
import numpy as np

# 生成不同形状的数据
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_blobs, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 层次聚类
agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_moons)
print("层次聚类月牙数据:", np.unique(labels_agg, return_counts=True))

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_moons)
print("DBSCAN 聚类月牙数据:", np.unique(labels_dbscan, return_counts=True))

# KMeans 对比
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_moons)
print("KMeans 聚类月牙数据:", np.unique(labels_kmeans, return_counts=True))

# 对 blob 数据使用不同算法
agg_blobs = AgglomerativeClustering(n_clusters=3)
labels_agg_blobs = agg_blobs.fit_predict(X_blobs)
print("层次聚类 blob 数据:", np.unique(labels_agg_blobs, return_counts=True))

dbscan_blobs = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan_blobs = dbscan_blobs.fit_predict(X_blobs)
print("DBSCAN 聚类 blob 数据:", np.unique(labels_dbscan_blobs, return_counts=True))
```

**检查点**
- 理解层次聚类和 DBSCAN 的原理
- 能根据数据特点选择合适的聚类算法
- 理解不同聚类算法的适用场景

---

### 第五阶段：数据预处理与特征工程 (预计 2 天)

#### 第13课：数据预处理
**学习目标**
- 掌握数据标准化方法
- 理解缺失值处理
- 掌握类别编码方法

**学习内容**
- 标准化（StandardScaler）
- 最小-最大缩放（MinMaxScaler）
- 鲁棒缩放（RobustScaler）
- 标签编码（LabelEncoder）
- 独热编码（OneHotEncoder）

**源码位置**
- `sklearn/preprocessing/_data.py`
- `sklearn/preprocessing/_label.py`
- `sklearn/preprocessing/_encoders.py`

**实践任务**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# 创建示例数据
data = {
    'age': [25, 30, 35, 40, 45, np.nan, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 200000],
    'city': ['Beijing', 'Shanghai', 'Beijing', 'Guangzhou', 'Shanghai', 'Beijing', 'Guangzhou'],
    'purchased': [0, 1, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])

# 数值特征标准化
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# 类别编码
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# 独热编码
ohe = OneHotEncoder(sparse_output=False)
city_ohe = ohe.fit_transform(df[['city']])
print("独热编码结果:")
print(city_ohe)

print("\n预处理后的数据:")
print(df)
```

**检查点**
- 掌握各种数据预处理方法
- 理解不同预处理方法的适用场景
- 能独立完成数据预处理流程

---

#### 第14课：特征工程
**学习目标**
- 理解特征工程的重要性
- 掌握多项式特征生成
- 学习特征选择方法

**学习内容**
- 多项式特征
- 特征选择（SelectKBest、RFE）
- 特征重要性分析
- 特征变换

**源码位置**
- `sklearn/preprocessing/_polynomial.py`

**实践任务**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)

# 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"原始特征数: {X.shape[1]}")
print(f"多项式特征数: {X_poly.shape[1]}")

# 特征选择
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print(f"选择后的特征数: {X_selected.shape[1]}")

# 递归特征消除
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
print(f"RFE 选择后的特征数: {X_rfe.shape[1]}")
print(f"被选择的特征索引: {rfe.support_}")
```

**检查点**
- 理解特征工程的重要性
- 能使用多项式特征增强模型
- 能进行特征选择

---

### 第六阶段：模型评估与调优 (预计 2-3 天)

#### 第15课：交叉验证
**学习目标**
- 理解交叉验证原理
- 掌握交叉验证方法
- 理解交叉验证的作用

**学习内容**
- K折交叉验证
- 分层K折交叉验证
- 留一交叉验证
- 交叉验证评分

**源码位置**
- `sklearn/model_selection/_split.py`
- `sklearn/model_selection/_validation.py`

**实践任务**
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 创建模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 普通K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(rf, X, y, cv=kf)
print(f"K折交叉验证分数: {scores_kf}")
print(f"平均分数: {scores_kf.mean():.4f}")
print(f"标准差: {scores_kf.std():.4f}")

# 分层K折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(rf, X, y, cv=skf)
print(f"\n分层K折交叉验证分数: {scores_skf}")
print(f"平均分数: {scores_skf.mean():.4f}")
print(f"标准差: {scores_skf.std():.4f}")
```

**检查点**
- 理解交叉验证原理
- 能使用不同的交叉验证方法
- 理解交叉验证分数的含义

---

#### 第16课：超参数调优
**学习目标**
- 理解超参数概念
- 掌握网格搜索和随机搜索
- 学习学习曲线分析

**学习内容**
- 网格搜索（GridSearchCV）
- 随机搜索（RandomizedSearchCV）
- 学习曲线
- 验证曲线

**源码位置**
- `sklearn/model_selection/_search.py`
- `sklearn/model_selection/_validation.py`

**实践任务**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

print("网格搜索最佳参数:", grid_search.best_params_)
print("网格搜索最佳分数:", grid_search.best_score_)

# 随机搜索
param_dist = {
    'n_estimators': np.arange(50, 201, 10),
    'max_depth': [None] + list(np.arange(5, 31, 5)),
    'min_samples_split': np.arange(2, 11)
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, 
                                   n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X, y)

print("\n随机搜索最佳参数:", random_search.best_params_)
print("随机搜索最佳分数:", random_search.best_score_)

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

print(f"\n训练集平均分数: {train_mean}")
print(f"测试集平均分数: {test_mean}")
```

**检查点**
- 理解超参数调优的重要性
- 能使用网格搜索和随机搜索
- 能分析学习曲线

---

### 第七阶段：流水线与实战项目 (预计 3-4 天)

#### 第17课：Pipeline 流水线
**学习目标**
- 理解 Pipeline 的作用
- 掌握 Pipeline 的使用
- 学会构建完整的机器学习流程

**学习内容**
- Pipeline 原理
- Pipeline 的优势
- 自定义 Transformer
- Pipeline 与 GridSearchCV 结合

**源码位置**
- `sklearn/pipeline.py`

**实践任务**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 训练
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Pipeline 准确率: {score:.4f}")

# Pipeline 参数调优
param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

**检查点**
- 理解 Pipeline 的优势
- 能构建完整的机器学习流程
- 能对 Pipeline 进行参数调优

---

#### 第18课：综合实战项目
**学习目标**
- 应用所学知识完成完整项目
- 掌握端到端的机器学习流程
- 培养问题解决能力

**项目选择（任选其一）**

**项目1：泰坦尼克号生存预测**
- 数据加载和探索
- 特征工程（处理缺失值、类别编码）
- 模型选择和训练
- 模型评估和调优
- 结果分析和可视化

**项目2：房价预测**
- 数据加载和探索
- 特征工程
- 回归模型训练
- 模型评估
- 特征重要性分析

**项目3：客户细分**
- 数据加载和探索
- 数据预处理
- 聚类分析
- 聚类结果解释
- 可视化展示

**实践任务**
选择一个项目，按照以下步骤完成：
1. 数据加载和初步探索
2. 数据预处理和特征工程
3. 模型选择和训练
4. 模型评估和调优
5. 结果分析和可视化
6. 撰写项目报告

**检查点**
- 能独立完成完整的机器学习项目
- 能解释模型结果
- 能进行有效的特征工程

---

### 第八阶段：进阶主题 (预计 2-3 天)

#### 第19课：模型集成方法
**学习目标**
- 理解不同的集成策略
- 掌握 Voting、Bagging、Stacking
- 理解 Gradient Boosting

**学习内容**
- Voting Classifier（硬投票、软投票）
- Bagging Classifier
- Stacking Classifier
- Gradient Boosting
- XGBoost、LightGBM（可选）

**源码位置**
- `sklearn/ensemble/_voting.py`
- `sklearn/ensemble/_bagging.py`
- `sklearn/ensemble/_stacking.py`
- `sklearn/ensemble/_gb.py`

**实践任务**
```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基模型
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

# Voting Classifier
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)
print(f"Voting Classifier 准确率: {voting_clf.score(X_test, y_test):.4f}")

# Bagging Classifier
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging_clf.fit(X_train, y_train)
print(f"Bagging Classifier 准确率: {bagging_clf.score(X_test, y_test):.4f}")

# Stacking Classifier
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)
print(f"Stacking Classifier 准确率: {stacking_clf.score(X_test, y_test):.4f}")

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting 准确率: {gb_clf.score(X_test, y_test):.4f}")
```

**检查点**
- 理解不同集成方法的原理
- 能根据场景选择合适的集成方法
- 理解集成学习的优势

---

#### 第20课：降维与可视化
**学习目标**
- 理解降维的目的
- 掌握 PCA 和 t-SNE
- 学会高维数据可视化

**学习内容**
- PCA 原理
- 核 PCA
- t-SNE（需要安装额外的库）
- 降维效果评估
- 数据可视化

**源码位置**
- `sklearn/decomposition/_pca.py`
- `sklearn/decomposition/_kernel_pca.py`

**实践任务**
```python
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('PCA 降维')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')

# 核 PCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('核 PCA 降维')
plt.xlabel('第一核主成分')
plt.ylabel('第二核主成分')

plt.tight_layout()
plt.savefig('pca_visualization.png')

# 解释方差比例
pca_full = PCA()
pca_full.fit(X)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
print(f"前 10 个主成分解释的方差比例: {cumsum[:10]}")
print(f"保留 95% 方差需要的主成分数: {np.argmax(cumsum >= 0.95) + 1}")
```

**检查点**
- 理解降维的原理和作用
- 能使用 PCA 进行降维
- 能解释降维结果

---

## 学习建议

### 学习方法
1. **理论与实践结合**：每学完一个概念，立即动手实践
2. **循序渐进**：按照课程顺序学习，不要跳过基础
3. **多做练习**：除了课程中的实践任务，自己寻找额外的数据集练习
4. **记录笔记**：记录重要的概念、代码和遇到的问题
5. **阅读源码**：对于感兴趣的方法，阅读源码加深理解

### 学习资源
- 官方文档：https://scikit-learn.org/stable/user_guide.html
- API 参考：https://scikit-learn.org/stable/modules/classes.html
- 示例代码：https://scikit-learn.org/stable/auto_examples/
- 用户指南：https://scikit-learn.org/stable/user_guide.html

### 常见问题
1. **Q: 如何选择合适的算法？**
   A: 参考 todo.md 中的模型选择指南，从简单算法开始，逐步尝试复杂算法

2. **Q: 模型性能不好怎么办？**
   A: 检查数据质量、尝试特征工程、调整超参数、尝试不同的算法

3. **Q: 如何避免过拟合？**
   A: 使用交叉验证、增加数据量、正则化、简化模型

4. **Q: 数据预处理有哪些步骤？**
   A: 处理缺失值、特征缩放、类别编码、特征选择、特征工程

### 学习检查清单
完成课程后，你应该能够：
- [ ] 独立完成数据加载和探索
- [ ] 进行数据预处理和特征工程
- [ ] 选择合适的机器学习算法
- [ ] 训练和评估模型
- [ ] 进行超参数调优
- [ ] 使用 Pipeline 构建完整的机器学习流程
- [ ] 解释模型结果和特征重要性
- [ ] 完成一个端到端的机器学习项目

## 下一步学习建议
完成本课程后，你可以继续学习：
- 深度学习（PyTorch、TensorFlow）
- 自然语言处理（NLTK、spaCy、transformers）
- 计算机视觉（OpenCV、PIL）
- 时间序列分析
- 强化学习
- 大规模机器学习（Spark MLlib）

---

**祝你学习愉快！如有问题，随时向我提问。**
