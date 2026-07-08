# VSCode + Python 插件启动 scikit-learn 项目指南

## 前置要求
- Python >= 3.11
- VSCode
- Python 插件（Microsoft 官方）

## 启动步骤

### 1. 创建虚拟环境
在项目根目录打开终端，执行：
```bash
python -m venv venv
```

### 2. 激活虚拟环境
Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. 安装构建依赖
```bash
pip install --upgrade pip
pip install meson-python>=0.17.1 cython>=3.1.2 numpy>=2 scipy>=1.10.0
```

### 4. 安装项目（开发模式）
```bash
pip install -e . --no-build-isolation
```

### 5. 安装测试依赖（可选）
```bash
pip install pytest>=7.1.2 matplotlib>=3.6.1 pandas>=1.5.0
```

### 6. 验证安装
运行测试：
```bash
pytest sklearn
```

或运行示例代码：
```bash
cd examples
python classification/plot_digits_classification.py
```

## VSCode 配置建议

### 选择 Python 解释器
1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择虚拟环境中的 Python（`venv/Scripts/python.exe`）

### 创建 launch.json（用于调试）
在 `.vscode/launch.json` 中添加：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Python: pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["sklearn"],
            "console": "integratedTerminal"
        }
    ]
}
```

## 常见问题

### 构建失败
确保已安装 C 编译器：
- Windows: 安装 Visual STudio Build Tools 或 MinGW
- Linux: `sudo apt install build-essential`
- Mac: `xcode-select --install`

### 导入错误
确保在虚拟环境中，并且项目已正确安装。

## 运行示例
在 VSCode 中打开 `examples/` 目录下的任意 `.py` 文件，按 `F5` 或点击右上角的运行按钮即可执行。

## AI会话工具中的应用

### scikit-learn 在多模态AI会话工具中的作用

**重要说明**：scikit-learn 是传统机器学习库，主要用于经典机器学习任务。对于现代AI会话工具（如ChatGPT类应用），主要使用的是深度学习框架（PyTorch、TensorFlow）和大语言模型库（transformers、langchain）。但 scikit-learn 仍可以在以下方面发挥作用：

#### 文本处理相关
- **文本分类**：使用 LogisticRegression、SVC、NaiveBayes 对用户查询进行分类（如：技术问题、闲聊、投诉等）
- **情感分析**：分析用户输入的情感倾向（正面、负面、中性）
- **文本特征提取**：使用 TfidfVectorizer、CountVectorizer 提取文本特征
- **文档相似度**：计算文档之间的相似度，用于检索相关历史对话
- **主题建模**：使用 LatentDirichletAllocation 发现对话中的主题

#### 图像处理相关
- **图像分类**：使用传统机器学习方法对图像进行分类（需要先提取特征）
- **特征提取**：从图像中提取特征（需要配合其他特征提取工具）
- **图像聚类**：对相似图像进行分组

#### 数据预处理
- **文本标准化**：使用 StandardScaler、MinMaxScaler 对特征进行标准化
- **类别编码**：使用 LabelEncoder、OneHotEncoder 对类别变量编码
- **缺失值处理**：使用 SimpleImputer 处理数据中的缺失值
- **特征选择**：使用 SelectKBest、RFE 选择最重要的特征

#### 用户行为分析
- **用户分群**：使用 KMeans、DBSCAN 对用户进行聚类分群
- **用户意图识别**：分类用户的查询意图
- **推荐系统**：基于用户历史行为推荐相关内容
- **异常检测**：使用 IsolationForest、LocalOutlierFactor 检测异常查询或恶意行为

#### 对话管理
- **对话状态跟踪**：使用分类器跟踪对话状态
- **查询路由**：将用户查询路由到不同的处理模块
- **相似度计算**：使用 pairwise_distances 计算查询与知识库条目的相似度

#### 质量评估
- **回答质量评估**：使用回归模型评估回答质量
- **用户满意度预测**：预测用户对回答的满意度
- **A/B测试分析**：分析不同模型的表现

### scikit-learn 无法直接完成的任务
- **大语言模型推理**：需要使用 transformers、langchain 等库
- **语音识别**：需要使用 Whisper、SpeechRecognition 等库
- **语音合成**：需要使用 TTS（Text-to-Speech）库
- **视频理解**：需要使用 VideoMAE、Video transformers 等库
- **深度学习模型**：需要使用 PyTorch、TensorFlow 等框架
- **多模态融合**：需要使用专门的多模态模型（如 CLIP、GPT-4V）

### 推荐的技术栈组合
对于构建多模态AI会话工具，建议的技术栈：
- **大语言模型**：transformers、langchain、openai
- **深度学习框架**：PyTorch 或 TensorFlow
- **语音处理**：Whisper（识别）、pyttsx3/gTTS（合成）
- **图像处理**：PIL、opencv-python、CLIP
- **视频处理**：moviepy、opencv-python
- **文档处理**：PyPDF2、python-docx
- **传统机器学习**：scikit-learn（用于辅助任务）
- **向量数据库**：chromadb、faiss、pinecone
- **Web框架**：FastAPI、Flask、Streamlit

### scikit-learn 在项目中的具体应用示例
```python
# 示例1：用户查询分类
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 训练查询分类器
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()
# 将用户查询分类为：技术问题、闲聊、投诉等

# 示例2：用户分群
from sklearn.cluster import KMeans

# 对用户行为数据进行聚类，发现不同用户群体
kmeans = KMeans(n_clusters=5)
user_segments = kmeans.fit_predict(user_features)

# 示例3：文档相似度检索
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 计算查询与知识库文档的相似度
vectorizer = TfidfVectorizer()
query_vec = vectorizer.transform([query])
doc_vecs = vectorizer.transform(knowledge_base)
similarities = cosine_similarity(query_vec, doc_vecs)
```

## 应用场景

### 分类任务
- **图像识别**：识别手写数字、人脸识别、物体分类
- **文本分类**：垃圾邮件检测、情感分析、新闻分类
- **医疗诊断**：疾病预测、肿瘤良恶性判断
- **欺诈检测**：信用卡欺诈、网络入侵检测
- **客户流失预测**：预测客户是否会流失

### 回归任务
- **房价预测**：根据房屋特征预测房价
- **股票预测**：预测股票价格走势
- **销量预测**：预测产品销量
- **能源消耗预测**：预测电力、水资源消耗
- **风险评估**：信用评分、保险风险评估

### 聚类任务
- **客户分群**：根据消费行为将客户分组
- **文档聚类**：将相似文档归类
- **图像分割**：将图像分割成不同区域
- **异常检测**：发现数据中的异常点
- **推荐系统**：基于用户行为推荐商品

### 降维与特征提取
- **数据可视化**：将高维数据降维到2D/3D进行可视化
- **特征选择**：选择最重要的特征
- **数据压缩**：减少数据存储空间
- **噪声去除**：去除数据中的噪声
- **主题建模**：从文档集合中发现主题

### 模型选择与评估
- **交叉验证**：评估模型性能
- **超参数调优**：寻找最佳模型参数
- **模型比较**：比较不同算法的性能
- **学习曲线分析**：分析模型学习情况

### 数据预处理
- **数据标准化**：将数据缩放到统一范围
- **缺失值处理**：填充或删除缺失值
- **类别编码**：将类别变量转换为数值
- **特征缩放**：归一化、标准化等
- **数据清洗**：处理异常值和噪声

### 特征工程
- **多项式特征**：生成多项式特征
- **交互特征**：创建特征之间的交互
- **文本特征提取**：TF-IDF、词袋模型
- **图像特征提取**：HOG、SIFT等特征

## 核心函数和类

### 线性模型 (sklearn.linear_model)
- **LinearRegression** - 线性回归 `sklearn/linear_model/_base.py`
- **LogisticRegression** - 逻辑回归 `sklearn/linear_model/_logistic.py`
- **Ridge** - 岭回归 `sklearn/linear_model/_ridge.py`
- **Lasso** - Lasso回归 `sklearn/linear_model/_coordinate_descent.py`
- **ElasticNet** - 弹性网络 `sklearn/linear_model/_coordinate_descent.py`
- **SGDClassifier** - 随机梯度下降分类器 `sklearn/linear_model/_stochastic_gradient.py`
- **SGDRegressor** - 随机梯度下降回归器 `sklearn/linear_model/_stochastic_gradient.py`
- **Perceptron** - 感知机 `sklearn/linear_model/_perceptron.py`

### 集成方法 (sklearn.ensemble)
- **RandomForestClassifier** - 随机森林分类器 `sklearn/ensemble/_forest.py`
- **RandomForestRegressor** - 随机森林回归器 `sklearn/ensemble/_forest.py`
- **GradientBoostingClassifier** - 梯度提升分类器 `sklearn/ensemble/_gb.py`
- **GradientBoostingRegressor** - 梯度提升回归器 `sklearn/ensemble/_gb.py`
- **HistGradientBoostingClassifier** - 直方图梯度提升分类器 `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py`
- **AdaBoostClassifier** - AdaBoost分类器 `sklearn/ensemble/_weight_boosting.py`
- **VotingClassifier** - 投票分类器 `sklearn/ensemble/_voting.py`
- **BaggingClassifier** - Bagging分类器 `sklearn/ensemble/_bagging.py`
- **StackingClassifier** - 堆叠分类器 `sklearn/ensemble/_stacking.py`

### 支持向量机 (sklearn.svm)
- **SVC** - 支持向量分类器 `sklearn/svm/_classes.py`
- **SVR** - 支持向量回归器 `sklearn/svm/_classes.py`
- **LinearSVC** - 线性支持向量分类器 `sklearn/svm/_classes.py`
- **LinearSVR** - 线性支持向量回归器 `sklearn/svm/_classes.py`
- **NuSVC** - Nu支持向量分类器 `sklearn/svm/_classes.py`
- **OneClassSVM** - 单类支持向量机 `sklearn/svm/_classes.py`

### 决策树 (sklearn.tree)
- **DecisionTreeClassifier** - 决策树分类器 `sklearn/tree/_classes.py`
- **DecisionTreeRegressor** - 决策树回归器 `sklearn/tree/_classes.py`
- **ExtraTreeClassifier** - 极端随机树分类器 `sklearn/tree/_classes.py`
- **ExtraTreeRegressor** - 极端随机树回归器 `sklearn/tree/_classes.py`
- **export_text** - 导出决策树为文本 `sklearn/tree/_export.py`
- **plot_tree** - 绘制决策树 `sklearn/tree/_export.py`

### 聚类算法 (sklearn.cluster)
- **KMeans** - K均值聚类 `sklearn/cluster/_kmeans.py`
- **MiniBatchKMeans** - 小批量K均值 `sklearn/cluster/_kmeans.py`
- **DBSCAN** - 基于密度的聚类 `sklearn/cluster/_dbscan.py`
- **HDBSCAN** - 层次化基于密度的聚类 `sklearn/cluster/_hdbscan/hdbscan.py`
- **AgglomerativeClustering** - 层次聚类 `sklearn/cluster/_agglomerative.py`
- **SpectralClustering** - 谱聚类 `sklearn/cluster/_spectral.py`
- **Birch** - BIRCH聚类 `sklearn/cluster/_birch.py`
- **MeanShift** - 均值漂移 `sklearn/cluster/_mean_shift.py`

### 降维算法 (sklearn.decomposition)
- **PCA** - 主成分分析 `sklearn/decomposition/_pca.py`
- **IncrementalPCA** - 增量主成分分析 `sklearn/decomposition/_incremental_pca.py`
- **KernelPCA** - 核主成分分析 `sklearn/decomposition/_kernel_pca.py`
- **NMF** - 非负矩阵分解 `sklearn/decomposition/_nmf.py`
- **FastICA** - 快速独立成分分析 `sklearn/decomposition/_fastica.py`
- **TruncatedSVD** - 截断奇异值分解 `sklearn/decomposition/_truncated_svd.py`
- **LatentDirichletAllocation** - 潜在狄利克雷分配 `sklearn/decomposition/_lda.py`

### 最近邻 (sklearn.neighbors)
- **KNeighborsClassifier** - K近邻分类器 `sklearn/neighbors/_classification.py`
- **KNeighborsRegressor** - K近邻回归器 `sklearn/neighbors/_regression.py`
- **RadiusNeighborsClassifier** - 半径近邻分类器 `sklearn/neighbors/_classification.py`
- **NearestNeighbors** - 无监督最近邻 `sklearn/neighbors/_unsupervised.py`
- **LocalOutlierFactor** - 局部异常因子 `sklearn/neighbors/_lof.py`
- **KernelDensity** - 核密度估计 `sklearn/neighbors/_kde.py`

### 模型选择 (sklearn.model_selection)
- **train_test_split** - 训练测试集分割 `sklearn/model_selection/_split.py`
- **cross_val_score** - 交叉验证评分 `sklearn/model_selection/_validation.py`
- **cross_validate** - 交叉验证评估 `sklearn/model_selection/_validation.py`
- **GridSearchCV** - 网格搜索 `sklearn/model_selection/_search.py`
- **RandomizedSearchCV** - 随机搜索 `sklearn/model_selection/_search.py`
- **KFold** - K折交叉验证 `sklearn/model_selection/_split.py`
- **StratifiedKFold** - 分层K折交叉验证 `sklearn/model_selection/_split.py`
- **learning_curve** - 学习曲线 `sklearn/model_selection/_validation.py`

### 评估指标 (sklearn.metrics)
- **accuracy_score** - 准确率 `sklearn/metrics/_classification.py`
- **precision_score** - 精确率 `sklearn/metrics/_classification.py`
- **recall_score** - 召回率 `sklearn/metrics/_classification.py`
- **f1_score** - F1分数 `sklearn/metrics/_classification.py`
- **confusion_matrix** - 混淆矩阵 `sklearn/metrics/_classification.py`
- **classification_report** - 分类报告 `sklearn/metrics/_classification.py`
- **roc_auc_score** - ROC AUC分数 `sklearn/metrics/_ranking.py`
- **mean_squared_error** - 均方误差 `sklearn/metrics/_regression.py`
- **mean_absolute_error** - 平均绝对误差 `sklearn/metrics/_regression.py`
- **r2_score** - R²分数 `sklearn/metrics/_regression.py`
- **silhouette_score** - 轮廓系数 `sklearn/metrics/cluster/_unsupervised.py`

### 预处理 (sklearn.preprocessing)
- **StandardScaler** - 标准化 `sklearn/preprocessing/_data.py`
- **MinMaxScaler** - 最小-最大缩放 `sklearn/preprocessing/_data.py`
- **RobustScaler** - 鲁棒缩放 `sklearn/preprocessing/_data.py`
- **Normalizer** - 归一化 `sklearn/preprocessing/_data.py`
- **LabelEncoder** - 标签编码 `sklearn/preprocessing/_label.py`
- **OneHotEncoder** - 独热编码 `sklearn/preprocessing/_encoders.py`
- **PolynomialFeatures** - 多项式特征 `sklearn/preprocessing/_polynomial.py`

### 流水线 (sklearn.pipeline)
- **Pipeline** - 流水线 `sklearn/pipeline.py`
- **FeatureUnion** - 特征联合 `sklearn/pipeline.py`

### 数据集 (sklearn.datasets)
- **load_iris** - 鸢尾花数据集 `sklearn/datasets/_base.py`
- **load_digits** - 手写数字数据集 `sklearn/datasets/_base.py`
- **load_wine** - 葡萄酒数据集 `sklearn/datasets/_base.py`
- **load_breast_cancer** - 乳腺癌数据集 `sklearn/datasets/_base.py`
- **make_classification** - 生成分类数据集 `sklearn/datasets/_samples_generator.py`
- **make_regression** - 生成回归数据集 `sklearn/datasets/_samples_generator.py`

## 模型选择指南

### scikit-learn 不需要外部预训练模型
scikit-learn 是一个机器学习算法库，**不需要引用外部预训练模型**。它提供了各种算法实现，你只需要：
1. 准备自己的数据
2. 选择合适的算法
3. 在数据上训练模型
4. 使用训练好的模型进行预测

### 如何选择最合适的模型

#### 分类任务推荐模型
- **入门首选**：LogisticRegression（逻辑回归）- 简单快速，易于理解
- **通用性强**：RandomForestClassifier（随机森林）- 性能稳定，不需要太多调参
- **高精度需求**：GradientBoostingClassifier（梯度提升）- 通常能达到最佳性能
- **大数据集**：SGDClassifier（随机梯度下降）- 适合大规模数据
- **文本分类**：LinearSVC（线性支持向量机）- 在文本数据上表现优异
- **小数据集**：SVC（支持向量机）- 在小样本上表现好

#### 回归任务推荐模型
- **入门首选**：LinearRegression（线性回归）- 最基础的回归方法
- **通用性强**：RandomForestRegressor（随机森林回归）- 处理非线性关系
- **高精度需求**：GradientBoostingRegressor（梯度提升回归）- 性能最佳
- **大数据集**：SGDRegressor（随机梯度下降回归）- 适合大规模数据
- **特征选择**：Lasso（Lasso回归）- 自动进行特征选择
- **异常值多**：HuberRegressor（Huber回归）- 对异常值鲁棒

#### 聚类任务推荐模型
- **入门首选**：KMeans（K均值）- 最常用，简单高效
- **密度聚类**：DBSCAN - 能发现任意形状的簇，自动确定簇数量
- **层次聚类**：AgglomerativeClustering - 生成层次结构，适合小数据集
- **大数据集**：MiniBatchKMeans - K均值的加速版本
- **未知簇数**：HDBSCAN - 自动确定最佳簇数

#### 降维任务推荐模型
- **入门首选**：PCA（主成分分析）- 最常用的降维方法
- **非线性降维**：KernelPCA（核主成分分析）- 处理非线性关系
- **稀疏数据**：TruncatedSVD - 适合稀疏矩阵（如文本数据）
- **主题分析**：LatentDirichletAllocation - 文档主题建模
- **非负数据**：NMF（非负矩阵分解）- 适合非负数据

### 模型选择流程
1. **从简单模型开始**：先用 LogisticRegression、LinearRegression、KMeans 等简单模型建立基线
2. **评估性能**：使用交叉验证评估模型性能
3. **尝试集成方法**：如果性能不足，尝试 RandomForest、GradientBoosting 等集成方法
4. **超参数调优**：使用 GridSearchCV 或 RandomizedSearchCV 优化参数
5. **模型比较**：比较多个模型的性能，选择最优的

### 快速选择建议
- **不知道选什么**：先用 RandomForest（分类/回归）或 KMeans（聚类）
- **需要解释性**：用 LogisticRegression、LinearRegression、DecisionTree
- **追求最高精度**：用 GradientBoosting、HistGradientBoosting
- **数据量很大**：用 SGDClassifier、SGDRegressor、MiniBatchKMeans
- **数据量很小**：用 SVC、SVR、AgglomerativeClustering
