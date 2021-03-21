import numpy as np
import pandas as pd
from scipy import stats

def resultdf(model_list, experiment_logs):

  models = []
  for model in model_list:
    models.append(model.custom_name)

  d = {'Models':models}
  df = pd.DataFrame(d)
  resultarray = np.array(experiment_logs)

  resultarray_rmse = resultarray[:,:,0]
  resultarray_rmse = resultarray_rmse[~np.isnan(resultarray_rmse).any(axis=1)]
  avgs_rmse = np.mean(resultarray_rmse,axis=0)
  stdes_rmse = stats.sem(resultarray_rmse)
  df['test_Average_rmse'] = pd.Series(avgs_rmse)
  df['test_stde_rmse'] = pd.Series(stdes_rmse)

  resultarray_pe = resultarray[:,:,1]
  resultarray_pe = resultarray_pe[~np.isnan(resultarray_pe).any(axis=1)]
  avgs_pe = np.mean(resultarray_pe,axis=0)
  stdes_pe = stats.sem(resultarray_pe)
  df['test_average_pe'] = pd.Series(avgs_pe)
  df['test_stde_pe'] = pd.Series(stdes_pe)


  return df.T;