import torch
import matplotlib.pyplot as plt
import numpy as np


def shuffle(tuples):
  permutation = torch.randperm(tuples[0].size()[0])
  return permutation


def load_batch(step, train_tuple, batch_size,permutation, shuffle_epoch=False):
  i = step
  if shuffle_epoch==True:
    indices = permutation[i:i+batch_size]
    x = train_tuple[0][indices]
    y = train_tuple[1][indices]

  elif shuffle_epoch==False:
    x = train_tuple[0][i:i+batch_size]
    y = train_tuple[1][i:i+batch_size]
  batch = (x,y)
  return batch

def evaluation(pred,true):
  
  # pe
  pred_mean = torch.mean(pred)
  true_mean = torch.mean(true)
  pe = torch.abs((pred_mean-true_mean)/true_mean)
  
  # rmse
  rmse = torch.mean((pred-true)**2.)**0.5

  return rmse, pe


def log_plots(model_name, dataset_name, train_logs, val_logs_1, val_logs_2, test_logs_1, test_logs_2, remove_n=5, log_transform=False, verbose=True):

  if log_transform == True:
    train_logs = np.log(train_logs)
    val_logs_1 = np.log(val_logs_1)
    val_logs_2 = np.log(val_logs_2)
    test_logs_1 = np.log(test_logs_1)
    test_logs_2 = np.log(test_logs_2)
  fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
  ax1.plot(train_logs[remove_n:])
  ax2.plot(val_logs_1[remove_n:], label='validation')
  ax3.plot(val_logs_2[remove_n:], label='validation')
  ax2.plot(test_logs_1[remove_n:], label='test')
  ax3.plot(test_logs_2[remove_n:], label='test')
  ax2.legend()
  ax3.legend()

  ax1.set_title(model_name+':'+dataset_name+'train loss')
  ax2.set_title(model_name+':'+dataset_name+'test rmse')
  ax3.set_title(model_name+':'+dataset_name+'test pe')

  if verbose==True:
    plt.show()

  elif verbose==False:
    pass