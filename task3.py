#coding=GBK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem

from sklearn.pipeline import make_pipeline
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif']=['SimHei']#支持中文
plt.rcParams['axes.unicode_minus']=False#负号正常显示
FEATURE_X1='sepal width (cm)'
FEATURE_X2='petal length (cm)'
FEATURE_X3='petal width (cm)'
features_list=[FEATURE_X1,FEATURE_X2,FEATURE_X3]
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

def get_3_features2(df):
  """读取数据"""
  X = df[[FEATURE_X1, FEATURE_X2,FEATURE_X3]].values
  # 注意这里是按默认顺序放的
  y = df['label'].values
  for i,v in enumerate(y):
    if v==2:y[i]=0
  return X, y


def plot_3d_probability_map(clf, X, y, feat_idx=(0, 1, 2),n_grid=50, fixed_value='mean'):
    """
    clf      : 已经 fit 好的二分类模型（支持 predict_proba）
    X, y     : 数据与标签，X.shape = (n_samples, 3)
    feat_idx : 选哪三个特征的下标 (i, j, k)，其中 i,j 用来画平面，k 固定
    n_grid   : 网格分辨率
    fixed_value : 第3个特征的固定值方式，'mean' 或具体数值
    """
    clf.fit(X,y)
    X = np.asarray(X)
    y = np.asarray(y)

    i, j, k = feat_idx

    # 1. 取第3个特征的固定值
    if fixed_value == 'mean':
        xk0 = X[:, k].mean()
    else:
        xk0 = float(fixed_value)

    # 2. 在 (feat_i, feat_j) 上生成网格
    xi_min, xi_max = X[:, i].min() - 0.5, X[:, i].max() + 0.5
    xj_min, xj_max = X[:, j].min() - 0.5, X[:, j].max() + 0.5

    xi, xj = np.meshgrid(
        np.linspace(xi_min, xi_max, n_grid),
        np.linspace(xj_min, xj_max, n_grid)
    )

    # 3. 为网格拼出完整的 (x1,x2,x3)
    grid_full = np.empty((xi.size, X.shape[1]), dtype=float)
    grid_full[:, i] = xi.ravel()
    grid_full[:, j] = xj.ravel()
    grid_full[:, k] = xk0

    proba = clf.predict_proba(grid_full)[:, 1]
    Z = proba.reshape(xi.shape)

    Z = proba.reshape(xi.shape)

    # 5. 开始画图
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ----- (1) 概率曲面 -----
    surf = ax.plot_surface(
        xi, xj, Z,
        cmap=cm.Blues,  
        linewidth=0, antialiased=True, alpha=0.8
    )

    # ----- (2) 底面的等高线投影 -----
    z_min = Z.min()
    cset = ax.contourf(
        xi, xj, Z,
        zdir='z', offset=z_min,  # 投影到 z = z_min 的平面
        cmap='RdBu', alpha=0.5,levels=15
    )

    # 也可以加侧面的投影（类似你图里的那个）：
    ax.contourf(xi, xj, Z, zdir='x', offset=xi_min, cmap='RdBu', alpha=0.8,levels=15)
    ax.contourf(xi, xj, Z, zdir='y', offset=xj_max, cmap='RdBu', alpha=0.8,levels=15)
    # ----- (3) 把训练点也画上去（可选）-----
    # 用固定的 xk0，把 3D 点投到曲面上方/下方
    # mask0 = (y == 0)
    # mask1 = (y == 1)

    # ax.scatter(
    #     X[mask0, i], X[mask0, j],
    #     X[mask0, k], 
    #     c='tab:blue', edgecolor='k', s=30, label='Class 0'
    # )
    # ax.scatter(
    #     X[mask1, i], X[mask1, j],
    #     X[mask1, k],
    #     c='tab:orange', edgecolor='k', s=30, label='Class 1'
    # )

    # ----- (4) 轴、标题、颜色条 -----
    ax.set_xlabel("X:"+features_list[i])
    ax.set_ylabel("Y:"+features_list[j])
    ax.set_zlabel("Z:Probability")

    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(xj_min, xj_max)
    ax.set_zlim(z_min, Z.max()+0.1)

    # fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Probability")

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
  df=load_iris_df()
  print(df.head(10))

  X,y=get_3_features2(df)
  plot_3d_probability_map(make_pipeline(
      Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
      LogisticRegression(C=0.1),
    ),X,y,feat_idx=(0,1,2))
  plot_3d_probability_map(make_pipeline(
      Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
      LogisticRegression(C=0.1),
    ),X,y,feat_idx=(0,2,1))
  plot_3d_probability_map(make_pipeline(
      Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
      LogisticRegression(C=0.1),
    ),X,y,feat_idx=(1,2,0))