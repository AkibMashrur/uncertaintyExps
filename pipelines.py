import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def normalize_1d(train,val,test):
  ss = StandardScaler()
  scales = ss.fit(train.reshape(-1,1))
  train = ss.transform(train.reshape(-1,1))
  val = ss.transform(val.reshape(-1,1))
  test = ss.transform(test.reshape(-1,1))
  return train.ravel(), val.ravel(), test.ravel(), scales.scale_

def normalize_2d(train,val,test):
  ss = StandardScaler()
  scales = ss.fit(train)
  train = ss.transform(train)
  val = ss.transform(val)
  test = ss.transform(test)
  return train, val, test

def from_git(**kwargs):
  dataset = kwargs['dataset']
  url = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/"+dataset+"/data/data.txt"
  data = np.loadtxt(url)
  device = kwargs['device']
  X = data[:,:kwargs['label_index']]
  y = data[:,kwargs['label_index']]
  N = len(X)
  all_indices = np.linspace(0,N-1,N,dtype=int)
  train_indices, test_indices = train_test_split(all_indices, random_state=kwargs['seed'], 
                                                test_size=1-kwargs['split'], shuffle=True)
  train_indices, val_indices = train_test_split(train_indices, random_state=kwargs['seed'], 
                                                test_size=1-kwargs['split'], shuffle=True)                                              
  X_train = X[train_indices]
  y_train = y[train_indices]
  X_val = X[val_indices]
  y_val = y[val_indices]
  X_test = X[test_indices]
  y_test = y[test_indices]
  X_train, X_val, X_test = normalize_2d(X_train, X_val, X_test)
  y_train, y_val, y_test, target_scale = normalize_1d(y_train, y_val, y_test)
  X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
  X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
  X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
  y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
  y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
  y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
  target_scale = torch.tensor(target_scale, dtype=torch.float32).to(device)
  train = (X_train, y_train)
  val = (X_val, y_val)
  test = (X_test, y_test)
  return train, val, test, target_scale






  
