# Computer Vision Assignment Report

Title:A3 report

Student Name:莫丰源

Student ID:12311805



### 1. Experimental Design

本次实验基于 15-scene 数据集（每类最多 100 张训练 / 测试图像）。实现并评估三种组合：

- Tiny Image 特征 + 最近邻（1-NN）分类器。
- Bag-of-SIFT 特征 + 最近邻（χ² 距离）分类器。
- Bag-of-SIFT 特征 + 1-vs-all 线性 SVM 分类器。

关键实现与超参数（参见 `mycode/student_code_12311805.py` 与 `mycode/run_pipeline.py`）：

- Tiny image: 将灰度图缩放为 16x16, 零均值并 L2 归一化。
- Dense SIFT: 使用 cyvlfeat 的 `dsift`，步长 `step=15`（稀疏采样以加速），在构建词典时每图随机采样最多 100 个描述子。
- 视觉词典: kmeans 聚类，词典大小 `vocab_size=200`，已保存为 `mycode/vocab.pkl`。该文件是由 `mycode/run_pipeline.py` 在首次发现本地没有 `vocab.pkl` 时自动从训练集构建并写出的。
- Bag-of-SIFT 直方图: 对每图 SIFT 描述子量化到最近词，归一化直方图（L1 归一化）。
- 最近邻距离: Tiny 使用欧氏距离，Bag 使用 χ² 近似距离。
- SVM: 使用 `sklearn.svm.LinearSVC`，并在调参后选用 raw BoW 特征、`StandardScaler` 标准化、`C=1.0`、`loss='squared_hinge'` 的配置。

评估指标：对于每种组合，保存混淆矩阵并计算平均对角线（类平均准确率）。


### 2. Experimental Results Analysis

运行脚本：`mycode/run_pipeline.py`（会把输出保存到 `mycode/outputs/`）。主要结果如下：

![result](result.png)

- Tiny Image + NN: mean diagonal = 23.33%（符合预期 21% 以上）
- Bag-of-SIFT + NN: mean diagonal = 59.73%（符合预期 55% 以上）
- Bag-of-SIFT + SVM: mean diagonal = 70.07%（符合预期 65% 以上）

已保存文件（相对路径，位于仓库根目录下的 `Assignment3_code/mycode/outputs/`）：

- 混淆矩阵图像： `confusion_tiny_nn.png`, `confusion_bag_nn.png`, `confusion_bag_svm.png`
- 示例预测图像： `outputs/examples/example_*.jpg`

分析要点：
- Tiny Image 方法作为基线表现合理，证明流水线与数据加载正常。
- Bag-of-SIFT + NN 达到约 60% 的性能，说明词典与量化流程基本有效。
- Bag-of-SIFT + SVM 达到约 70%，说明最终的词典、特征表示与线性分类器组合已经达到作业预期。

本次有效调优策略：

- 对 Bag-of-SIFT 直方图做了参数搜索，并最终采用原始 BoW + 标准化 + `LinearSVC(C=1.0, loss='squared_hinge')` 的组合。
- 使用分层验证集与多组参数比较，选择最优 SVM 设置。
- 最终保留了 `vocab.pkl` 作为固定视觉词典，保证提交时结果可复现。


### 3. Bonus Report

当前已完成的加分项：

- 已完成：实现 Bag-of-SIFT 管线并缓存 `vocab.pkl`（节省重复计算时间）。
- 已完成：使用 notebook 记录并调优 `vocab_*.pkl`、Bag-NN 的 `k` 与 SVM 参数。

### 文件与文件夹说明

下列是仓库内重要文件/文件夹及其作用：

- **Assignment3_code/mycode/**：主要代码目录。
	- `student_code_12311805.py`：实现三种特征与分类器的核心函数（`get_tiny_images`, `build_vocabulary`, `get_bags_of_sifts`, `nearest_neighbor_classify`, `svm_classify`）。
	- `run_pipeline.py`：主运行脚本，按顺序生成特征、训练并评估 NN 与 SVM，会产生 `outputs/`。提交时用于复现实验结果。
	- `tuning_pipeline.ipynb`：参数搜索与词典变体探索的交互式 notebook（已用于寻找最优 `vocab` 与 SVM 配置）。
	- `optimize_svm.ipynb`：针对 SVM 的调参 notebook（历史实验记录）。
	- `vocab.pkl`：已构建并保存的视觉词典（kmeans 聚类中心），`run_pipeline.py` 在本地无该文件时会重建并写出；请随提交一并包含以保证结果可复现。
	- `utils.py`：数据加载、图像读写与辅助函数。
	- `outputs/`：运行时生成的结果文件夹，包含混淆矩阵图、示例预测图与 `opt_svm` 子文件夹（包含调参记录）。
- **Assignment3_code/data/**：数据集目录，包含 `train/` 与 `test/` 子文件夹（每类图像）。
- **crossvalidtion/**：额外的交叉验证笔记本与导出 HTML（调参分析的历史材料）。





