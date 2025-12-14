#coding=GBK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import to_rgba

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif']=['SimHei']#支持中文
plt.rcParams['axes.unicode_minus']=False#负号正常显示

from matplotlib.lines import Line2D

# FEATURE_X = 'petal length (cm)'
# FEATURE_Y = 'petal width (cm)'

FEATURE_X1='sepal width (cm)'
FEATURE_X2='petal length (cm)'
FEATURE_X3='petal width (cm)'
def load_iris_df():
  """加载鸢尾花数据"""
  iris=load_iris()
  X=iris.data
  y=iris.target
  feature_names=iris.feature_names
  target_names=iris.target_names
  df=pd.DataFrame(X,columns=feature_names)
  #加入数值标签
  df['label']=y
  #加入文字标签
  df['species']=df['label'].map(lambda i:target_names[i])
  # df['species']=target_names[0]
  return df

def get_3_features(df):
  """读取数据"""
  X = df[[FEATURE_X1, FEATURE_X2,FEATURE_X3]].values
  y = df['label'].values
  return X, y


# 体素渲染 弃用（太卡了）
def plot_3d_prob_voxels(clf, X, y, n=36, margin=0.5, alpha_min=0.05, conf_thresh=0.45):
    # 1) 网格
    clf.fit(X,y)
    mins = X.min(axis=0) - margin
    maxs = X.max(axis=0) + margin
    xs = np.linspace(mins[0], maxs[0], n)
    ys = np.linspace(mins[1], maxs[1], n)
    zs = np.linspace(mins[2], maxs[2], n)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]

    # 2) 概率
    proba = clf.predict_proba(grid)              # (N,3)
    cls = np.argmax(proba, axis=1)               # (N,)
    conf = np.max(proba, axis=1)                 # (N,)

    cls = cls.reshape((n, n, n))
    conf = conf.reshape((n, n, n))

    # 3) 过滤：低置信度不画
    filled = conf >= conf_thresh

    # 4) 颜色 + alpha
    base = np.zeros((n, n, n, 4), dtype=float)   # RGBA
    colors = np.array([
        to_rgba("tab:blue"),
        to_rgba("tab:orange"),
        to_rgba("tab:green")
    ])

    for c in [0, 1, 2]:
        mask = filled & (cls == c)
        base[mask, :3] = colors[c][:3]
        # alpha：把 conf 映射到 [alpha_min, 1]
        a = (conf[mask] - conf_thresh) / (1 - conf_thresh + 1e-9)
        base[mask, 3] = alpha_min + (1 - alpha_min) * a

    # 5) 画图
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
# 关键：用体素“边界坐标”（n+1），而不是中心点（n）
    x_edges = np.linspace(mins[0], maxs[0], n + 1)
    y_edges = np.linspace(mins[1], maxs[1], n + 1)
    z_edges = np.linspace(mins[2], maxs[2], n + 1)

    Xe, Ye, Ze = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    ax.voxels(Xe, Ye, Ze, filled, facecolors=base, edgecolor=None)

    # 散点用真实坐标，和 voxels 现在同一坐标系了
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, s=25, edgecolor='k')
    # 画散点（原数据）
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, s=25, edgecolor='k')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.tight_layout()
    plt.show()

def plot_3d_prob_scatter(clf, X, y, n=25, margin=0.5, conf_thresh=0.45):
    clf.fit(X,y)
    mins = X.min(axis=0) - margin
    maxs = X.max(axis=0) + margin
    xs = np.linspace(mins[0], maxs[0], n)
    ys = np.linspace(mins[1], maxs[1], n)
    zs = np.linspace(mins[2], maxs[2], n)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]

    proba = clf.predict_proba(grid)
    cls = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    keep = conf >= conf_thresh
    grid = grid[keep]
    cls = cls[keep]
    conf = conf[keep]

    # 颜色
    cmap =["Blues", "Oranges", "Greens"]
    # colors = cmap[cls]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for label in range(3):
      ax.scatter(grid[(cls==label),0], grid[(cls==label),1], grid[(cls==label),2], c=conf[(cls==label)]*2,cmap=cmap[label],alpha=conf[(cls==label)], s=8, linewidths=0)
      ax.scatter(X[(y==label),0], X[(y==label),1], X[(y==label),2],marker='*',cmap=cmap[label], s=50, edgecolor='k',label=f"Class {label}")
    
    legend_elements = [
        Line2D([0], [0], marker='*', color='w',
              label='Class 0',
              markerfacecolor='tab:blue', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='*', color='w',
              label='Class 1',
              markerfacecolor='tab:orange', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='*', color='w',
              label='Class 2',
              markerfacecolor='tab:green', markeredgecolor='k', markersize=8),
    ]

    ax.legend(handles=legend_elements, loc='upper left')
    
    # ax.scatter(grid[(cls==0),0],grid[(cls==0),1],grid[(cls==0),2])
    ax.set_xlabel("X:"+FEATURE_X1)
    ax.set_ylabel("Y:"+FEATURE_X2)
    ax.set_zlabel("Z:"+FEATURE_X3)
    plt.show()


if __name__=="__main__":
  df=load_iris_df()
  print(df.head(10))
  X,y=get_3_features(df)
  plot_3d_prob_scatter(
    make_pipeline(
      Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
      LogisticRegression(C=0.5),
    ),
    X,y
    )
  # 体素，可选
  # plot_3d_prob_voxels(
  #   make_pipeline(
  #     Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
  #     LogisticRegression(C=0.5),
  #   ),
  #   X,y
  #   )