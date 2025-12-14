#coding=GBK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif']=['SimHei']#支持中文
plt.rcParams['axes.unicode_minus']=False#负号正常显示

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

def get_two_features(df):
  """读取数据"""
  X = df[[FEATURE_X2, FEATURE_X3]].values
  y = df['label'].values
  # for i in y:print(f"{i} ")
  return X, y

def get_classifiers():
  """存所有分类器内容"""
  classifiers = {
    "Logistic regression\n(C=0.01)": LogisticRegression(
      C=0.1
    ),
    "Logistic regression\n(C=1)": LogisticRegression(
      C=100
    ),
    "Gaussian Process": GaussianProcessClassifier(
      kernel=1.0 * RBF([1.0, 1.0])
    ),
    "Logistic regression\n(RBF features)": make_pipeline(
      Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
      LogisticRegression(C=10),
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(),
    "Logistic regression\n(binned features)": make_pipeline(
      KBinsDiscretizer(           # 连续特征离散化成 5 个区间
        n_bins=5,
        quantile_method="averaged_inverted_cdf",
      ),
      PolynomialFeatures(interaction_only=True),  # 加交叉项
      LogisticRegression(C=10),
    ),
    "Logistic regression\n(spline features)": make_pipeline(
      SplineTransformer(n_knots=5),
      PolynomialFeatures(interaction_only=True),
      LogisticRegression(C=10),
    ),
  }
  return classifiers

def make_meshgrid(X,h=0.01):
  """生成并返回数据图网格上的所有格点"""
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  # 这里xx,yy都是二维的列表，用来分别存上面的所有格点的x坐标和y坐标
  xx,yy=np.meshgrid(
    np.linspace(x_min,x_max,int((x_max-x_min)/h)),
    np.linspace(y_min,y_max,int((y_max-y_min)/h))
  )
  return xx,yy

def plot_all_classifiers_grid(df, classifiers):
  n_clf=len(classifiers)
  rows,cols=n_clf,4
  n_cbar=3
  
  X,y=get_two_features(df)
  xx,yy=make_meshgrid(X,h=0.02)
  grid=np.c_[xx.ravel(),yy.ravel()]

  fig=plt.figure(figsize=(13,10))
  gs=gridspec.GridSpec(rows,n_cbar+cols,width_ratios=[0.5]*n_cbar+[3]*cols,wspace=0.8)
  
  prob_cmap="Blues"
  color_prob_scatter="white"
  max_cmap = ListedColormap(["blue", "orange", "green"])
  
  last_prob_cs=None
  for row,(name,clf) in enumerate(classifiers.items()):
    clf.fit(X,y)
    proba=clf.predict_proba(grid)#(N,3)
    z0=proba[:,0].reshape(xx.shape)
    z1=proba[:,1].reshape(xx.shape)
    z2=proba[:,2].reshape(xx.shape)
    z_list=[z0,z1,z2]
    zmax=np.argmax(proba,axis=1).reshape(xx.shape)
    for col in range(3):
      cur_ax=fig.add_subplot(gs[row,n_cbar+col])
      cur_ax.set_xlabel(f"Class {col}")
      # cur_ax.axis('off')
      cur_ax.set_xticks([])
      cur_ax.set_yticks([])
      cur_cs=cur_ax.contourf(xx,yy,z_list[col],levels=100,cmap=prob_cmap)
      cur_ax.scatter(X[(y==col),0],X[(y==col),1],c=color_prob_scatter,edgecolors='k',s=15)
      last_prob_cs=cur_cs
      if(col==0):cur_ax.set_ylabel(name,fontsize=8)
    ax_max=fig.add_subplot(gs[row,n_cbar+3])
    print(n_cbar+3)
    ax_max.set_xticks([])
    ax_max.set_yticks([])
    mask0=(zmax==0)
    im0=ax_max.imshow(
      np.where(mask0,z0,np.nan),
      cmap="Blues",
      # alpha=0.8,
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      origin="lower",
      interpolation="bilinear"
    )
    mask1=(zmax==1)
    im1=ax_max.imshow(
      np.where(mask1,z1,np.nan),
      cmap="Oranges",
      # alpha=0.8,
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      origin="lower",
      interpolation="bilinear"
    )
    mask2=(zmax==2)
    im2=ax_max.imshow(
      np.where(mask2,z2,np.nan),
      cmap="Greens",
      # alpha=0.8,
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      origin="lower",
      interpolation="bilinear"
    )
    ax_max.scatter(X[:,0],X[:,1],c=y,cmap=max_cmap,edgecolors='k',s=15)
    ax_max.set_xlabel("Max class")
    
    # 统一的 x/y 范围（基于 xx, yy）
    x0, x1 = xx.min(), xx.max()
    y0, y1 = yy.min(), yy.max()

    # 对每个 cur_ax（前三列）设置相同的 limits 和 aspect
    cur_ax.set_xlim(x0, x1)
    cur_ax.set_ylim(y0, y1)
    cur_ax.set_aspect('auto')   # 或 'equal' 视你想要的横纵比例而定

    # 对 max 这一列也强制一致（关键）
    ax_max.set_xlim(x0, x1)
    ax_max.set_ylim(y0, y1)
    ax_max.set_aspect('auto')
    
  cax_prob=fig.add_subplot(gs[:-2,1])
  cb_prob=fig.colorbar(last_prob_cs,cax=cax_prob)
  cb_prob.ax.invert_yaxis()
  cb_prob.set_label("Probability",rotation=270,labelpad=15)
  cax_prob.yaxis.set_ticks_position('left')
  cax_prob.yaxis.set_label_position('right')
  
  cax_0=fig.add_subplot(gs[-2:,0])
  cb_0=fig.colorbar(im0,cax=cax_0)
  cb_0.ax.invert_yaxis()
  cb_0.set_label("Probability class0",rotation=270,labelpad=15)
  cax_0.yaxis.set_ticks_position('left')
  cax_0.yaxis.set_label_position('right')
  
  cax_1=fig.add_subplot(gs[-2:,1])
  cb_1=fig.colorbar(im1,cax=cax_1)
  cb_1.ax.invert_yaxis()
  cb_1.set_label("Probability class1",rotation=270,labelpad=15)
  cax_1.yaxis.set_ticks_position('left')
  cax_1.yaxis.set_label_position('right')
  cax_1.set_yticks([])
  cax_2=fig.add_subplot(gs[-2:,2])
  cb_2=fig.colorbar(im2,cax=cax_2)
  cb_2.ax.invert_yaxis()
  cb_2.set_label("Probability class2",rotation=270,labelpad=15)
  cax_2.yaxis.set_ticks_position('left')
  cax_2.yaxis.set_label_position('right')
  cax_2.set_yticks([])
  
  
  plt.show()

if __name__=="__main__":
  df=load_iris_df()
  print(df.head(10))
  classifiers=get_classifiers()
  X,y=get_two_features(df)
  plot_all_classifiers_grid(df,classifiers)
