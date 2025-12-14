
# Iris 数据集分类与可视化实验

本项目基于经典的 Iris（鸢尾花）数据集，使用逻辑回归及其核近似模型，
在二维与三维特征空间中对分类结果进行可视化分析。
项目通过四个独立脚本（task1–task4），逐步从二维决策边界扩展到三维概率分布，
最终实现三特征三分类的 3D Probability Map 可视化。

---

## 一、项目结构

```text
.
├── task1.py        # 二维三分类：决策区域与概率热力图
├── task2.py        # 三特征二分类：3D 决策边界平面
├── task3.py        # 三特征二分类：条件切片概率曲面
├── task4.py        # 三特征三分类：3D Probability Map（核心）
├── figures/        # 保存生成的可视化结果图片（可选）
└── README.md       # 项目说明文件
```

---

## 二、实验环境

* 操作系统：Windows / Linux / macOS（均可）
* Python 版本：Python 3.8 及以上
* 运行方式：直接运行 `.py` 脚本

---

## 三、依赖库

请先确保已安装以下依赖：

```bash
pip install numpy pandas matplotlib scikit-learn
```

各依赖作用说明：

* `numpy` / `pandas`：数据处理
* `matplotlib`：二维与三维可视化绘图
* `scikit-learn`：

  * Iris 数据集
  * Logistic Regression 分类模型
  * `predict_proba` 概率预测接口
  * Nystroem（RBF）核近似（可选）

---

## 四、各任务说明

### task1：二维三分类决策边界与概率热力图

* 特征维度：2 个特征
* 分类类型：3 分类
* 可视化内容：

  * 最大类别区域（MaxClass 决策区域）
  * 各类别预测概率的二维热力图
* 绘图方式：

  * `meshgrid` 构建二维网格
  * `predict_proba` 预测概率
  * `contourf / imshow` 显示结果

运行方式：

```bash
python task1.py
```

---

### task2：三特征二分类 3D 决策边界平面

* 特征维度：3 个特征
* 分类类型：2 分类（合并其中一类）
* 可视化内容：

  * 三维空间中的决策边界平面
  * 两类样本的三维散点
* 理论基础：

  * 线性模型决策函数：
    $$ w^\top x + b = 0 $$

运行方式：

```bash
python task2.py
```

---

### task3：三特征二分类条件切片概率曲面

* 特征维度：3 个特征
* 分类类型：2 分类
* 核心思想：条件切片

  * 固定 1 个特征
  * 在另外 2 个特征构成的平面上绘制概率曲面
* 可视化内容：

  * 概率曲面
  * 底部或侧面的概率投影
* 扩展：

  * 轮换不同特征组合，分析各特征对分类的影响

运行方式：

```bash
python task3.py
```

---

### task4：三特征三分类 3D Probability Map（核心任务）

* 特征维度：3 个特征
* 分类类型：3 分类
* 核心思想：

  * 对三维特征空间进行均匀采样（点云或体素）
  * 对每个采样点预测三类概率
* 可视化编码方式：

  * 颜色：`argmax(proba)`（最大概率对应类别）
  * 透明度：`max(proba)`（模型预测置信度）
* 可选优化：

  * 设置置信度阈值，过滤低置信度点
  * 使用点云方式提高渲染效率

运行方式：

```bash
python task4.py
```

---

## 五、结果输出说明

* 脚本运行后将：

  * 直接弹出 Matplotlib 绘图窗口
  * 或将图像保存至 `figures/` 目录（取决于具体实现）
* 生成的图像可直接用于实验报告或展示

---

## 六、复现实验步骤（简要）

```bash
pip install numpy pandas matplotlib scikit-learn
python task1.py
python task2.py
python task3.py
python task4.py
```

按顺序运行即可完整复现实验结果。

---

## 七、说明

* 各 task 脚本相互独立，便于单独调试与展示
* 所有模型统一使用 `fit / predict / predict_proba` 接口，方便扩展到其他分类模型
* task4 为本项目的主要创新与扩展部分，重点展示三维空间中的多分类概率分布
