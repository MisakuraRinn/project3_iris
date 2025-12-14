#coding=GBK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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

def get_3_features1(df):
  """读取数据"""
  X = df[[FEATURE_X1, FEATURE_X2,FEATURE_X3]].values
  y = df['label'].values
  for i,v in enumerate(y):
    if v==2:y[i]=1
  return X, y
def plot_3d_decision_boundary(clf,X,y,feature_names=None):
  clf.fit(X,y)
  X=np.asarray(X)
  y=np.asarray(y)
  w=clf.coef_[0]
  b=clf.intercept_[0]
  i,j,k=0,1,2
  xi_min,xi_max=X[:,i].min(),X[:,i].max()
  xj_min,xj_max=X[:,j].min(),X[:,j].max()
  xi,xj=np.meshgrid(np.linspace(xi_min,xi_max,30),np.linspace(xj_min,xj_max,30))
  xk=-(w[i]*xi+w[j]*xj+b)/w[k]
  fig=plt.figure(figsize=(6,5))
  ax=fig.add_subplot(111,projection='3d')
  ax.plot_surface(xi,xj,xk,alpha=0.3,color='C0')
  for cls,color,label in[(0,'tab:blue','Class 0'),(1,'tab:orange','Class 1')]:
    mask=(y==cls)
    ax.scatter(X[mask,i],X[mask,j],X[mask,k],c=color,edgecolor='k',s=50,label=label)
  zmin, zmax = X[:, 2].min(), X[:, 2].max()
  ax.set_zlim(zmin, zmax) 
  ax.set_xlim(xi_min,xi_max)
  ax.set_ylim(xj_min,xj_max)
  ax.set_xlabel(feature_names[i])
  ax.set_ylabel(feature_names[j])
  ax.set_zlabel(feature_names[k])
  ax.legend()
  plt.tight_layout()
  plt.show()
if __name__=="__main__":
  df=load_iris_df()
  print(df.head(10))
  X,y=get_3_features1(df)
  plot_3d_decision_boundary(
    LogisticRegression(C=0.01),
    X,y,
    feature_names=["x1:"+FEATURE_X1,"x2:"+FEATURE_X2,"x3:"+FEATURE_X3]
  )