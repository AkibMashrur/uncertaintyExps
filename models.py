import torch
import torch.nn.functional as F
import math

def wls(pred,true,sigma):
  return torch.mean((pred-true)**2/sigma**2)

def nll_loss(y, mean, sigma):
    loss = torch.mean(torch.log(sigma**2)+(y-mean)**2/(sigma**2))
    return loss

def mse(pred,true):
  return torch.mean((pred-true)**2)

def pe(pred,true):
  pred_mean = torch.mean(pred)
  true_mean = torch.mean(true)
  return torch.abs((pred_mean-true_mean)/true_mean)

class MCDropout(torch.nn.Module):
    custom_name = "MCDropout"
    def __init__(self, **kwargs):
        super(MCDropout, self).__init__()   
        n_features = kwargs['n_features']
        n_hidden = kwargs['n_hidden']
        tau = kwargs['tau']
        n_train = kwargs['n_train']   
        dropout_rate = 0.1
        lengthscale = 1e-2    
        reg = lengthscale**2 * (1 - dropout_rate) / (2. * n_train * tau * 1.4)
        self.reg = reg
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.hidden = torch.nn.Linear(n_features, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, 1)   # output layer
        self.device = kwargs['device']

    def forward(self, x):
        x = self.dropout1(F.relu(self.hidden(x)))    # activation function for hidden layer
        x = self.dropout2(F.relu(self.hidden2(x)))
        x = self.predict(x)             # linear output
        return x

    def custom_compile(self):
      optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.reg)
      scheduler = None
      # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
      return optimizer, scheduler

    def train_step(self,**kwargs):
      self.train()
      preds = self(kwargs['batch'][0])
      optim_m = kwargs['optim']
      scheduler_m = kwargs['scheduler']      
      optim_m.zero_grad()
      loss = mse(preds, kwargs['batch'][1])
      loss.backward()
      optim_m.step()
      # scheduler_m.step()
      return loss.item();

    def evaluation_step(self,**kwargs):
      test_iters = kwargs['test_iters']
      target_scale = kwargs['target_scale']
      test_preds_sum = 0.
      for i in range(test_iters):
        self.train()
        test_preds_sum = test_preds_sum + (self(kwargs['test_tuple'][0])*target_scale)
      test_preds = test_preds_sum/test_iters
      test_score = kwargs['eval_func'](test_preds, kwargs['test_tuple'][1]*target_scale)
      scores = []
      for i in test_score:
        scores.append(i.detach().cpu().numpy())

      return scores

class Baseline(torch.nn.Module):
    custom_name = "Baseline"
    def __init__(self, **kwargs):
        super(Baseline, self).__init__()
        n_features = kwargs['n_features']
        n_hidden = kwargs['n_hidden']    
        self.device = kwargs['device']    
        self.hidden = torch.nn.Linear(n_features, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, 1)   # output laye

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x)) 
        x = self.predict(x)             # linear output
        return x

    def custom_compile(self):
      self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0)
      self.scheduler = None
      # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
      return self.optimizer, self.scheduler

    def train_step(self,**kwargs):
      preds = kwargs['model'](kwargs['batch'][0])
      optim_m = kwargs['optim']
      scheduler_m = kwargs['scheduler']
      
      kwargs['model'].train()
      optim_m.zero_grad()
      loss = mse(preds, kwargs['batch'][1])
      loss.backward()
      optim_m.step()
      # scheduler_m.step()
      return loss.item();

    def evaluation_step(self,**kwargs):
      target_scale = kwargs['target_scale']
      test_preds = kwargs['model'](kwargs['test_tuple'][0])*target_scale
      test_score = kwargs['eval_func'](test_preds, kwargs['test_tuple'][1]*target_scale)
      scores = []
      for i in test_score:
        scores.append(i.detach().cpu().numpy())

      return scores


### Classes for IVNet

class Mean_network(torch.nn.Module):
    def __init__(self, n_hidden):
        super(Mean_network, self).__init__()
        self.fc1 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 1)
        self.l_relu = torch.nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = self.l_relu(self.fc1(x))
        out = self.l_relu(self.fc2(out))
        return out 

class Variance_network(torch.nn.Module):
    def __init__(self,n_hidden):
        super(Variance_network, self).__init__()
        self.fc1 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 1)
        self.softplus = torch.nn.Softplus()
        self.l_relu = torch.nn.LeakyReLU(inplace=True)        
    def forward(self, x):
        out = self.l_relu(self.fc1(x))
        out = self.softplus(self.fc2(out))
        return out

class IVNet(torch.nn.Module):
    custom_name = "IVNet"
    def __init__(self, **kwargs):
        super(IVNet, self).__init__()
        self.shared_layer = torch.nn.Linear(kwargs['n_features'], kwargs['n_hidden'])
        self.mean = Mean_network(kwargs['n_hidden'])
        self.variance = Variance_network(kwargs['n_hidden'])
        self.device = kwargs['device']
        # self.device = device

    def forward(self, x, uniform_variance = False):
        if not uniform_variance:
          out = self.shared_layer(x)
          mean = self.mean(out)
          variance = self.variance(out)          
          return mean, variance;

        else:
          out = self.shared_layer(x)
          mean = self.mean(out)
          variance = torch.ones(mean.shape).to(self.device)
          return mean, variance;

    def custom_compile(self):
      # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0)
      self.optim_mean = torch.optim.SGD([ {'params': self.shared_layer.parameters()}, 
                                          {'params': self.mean.parameters()}], lr=0.01, momentum=0.9)
      self.optim_variance = torch.optim.SGD([{'params': self.variance.parameters()}], lr=1, momentum=0.9)
      # self.scheduler = None
      # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
      self.scheduler_mean = torch.optim.lr_scheduler.CyclicLR(self.optim_mean, base_lr=1e-4, max_lr=0.01, step_size_up=100)
      self.scheduler_variance = torch.optim.lr_scheduler.CyclicLR(self.optim_variance, base_lr=1e-3, max_lr=1, step_size_up=100)
      self.optimizer = [self.optim_mean, self.optim_variance]
      self.scheduler = [self.scheduler_mean, self.scheduler_variance]
      return self.optimizer, self.scheduler

    def train_step(self,**kwargs):
      # preds, sigmas = kwargs['model'](kwargs['batch'][0])
      optim_m = kwargs['optim'][0]
      optim_v = kwargs['optim'][1]
      scheduler_m = kwargs['scheduler'][0]
      scheduler_v = kwargs['scheduler'][1]
      
      self.train()
      optim_m.zero_grad()
      if len(kwargs['train_logs']) == 0:
        prediction, sigma = self.forward(kwargs['batch'][0], uniform_variance=True)
        loss = wls(prediction, kwargs['batch'][1], sigma)
        loss.backward()
        optim_m.step()
        scheduler_m.step()
        return loss.item()

      else:
        optim_v.zero_grad()
        prediction, sigma = self.forward(kwargs['batch'][0], uniform_variance=False)
        squared_residuals = (prediction-kwargs['batch'][1])**2
        squared_residuals = squared_residuals.clone().detach()
        variance_loss = mse(sigma, squared_residuals)
        variance_loss.backward()
        optim_v.step()
        scheduler_v.step()
        prediction, sigma = self.forward(kwargs['batch'][0], uniform_variance=False)
        loss = wls(prediction, kwargs['batch'][1], sigma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.)
        optim_m.step()
        scheduler_v.step() 
        return loss.item();

    def evaluation_step(self,**kwargs):
      target_scale = kwargs['target_scale']
      test_preds, test_sigmas = kwargs['model'](kwargs['test_tuple'][0])
      test_preds = test_preds*target_scale
      test_score = kwargs['eval_func'](test_preds, kwargs['test_tuple'][1]*target_scale)
      scores = []
      for i in test_score:
        scores.append(i.detach().cpu().numpy())

      return scores


### Classes for SDE net

class Drift(torch.nn.Module):
    def __init__(self, n_hidden):
        super(Drift, self).__init__()
        self.fc = torch.nn.Linear(n_hidden, n_hidden)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, t, x):
        out = self.relu(self.fc(x))
        return out    



class Diffusion(torch.nn.Module):
    def __init__(self, n_hidden):
        super(Diffusion, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 1)
    def forward(self, t, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    
class SDENet(torch.nn.Module):
    custom_name = "SDENet"
    def __init__(self, **kwargs):
        super(SDENet, self).__init__()
        self.layer_depth = kwargs['layer_depth']
        self.device = kwargs['device']
        n_hidden = kwargs['n_hidden']
        self.n_features = kwargs['n_features']
        self.downsampling_layers = torch.nn.Linear(kwargs['n_features'], n_hidden)
        self.drift = Drift(n_hidden)
        self.diffusion = Diffusion(n_hidden)
        self.fc_layers = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.Linear(n_hidden, 2))
        self.deltat = 4./self.layer_depth
        self.sigma = 0.5
    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma*self.diffusion(t, out)
            for i in range(self.layer_depth):
                t = 4*(float(i))/self.layer_depth
                out = out + self.drift(t, out)*self.deltat + diffusion_term*math.sqrt(self.deltat)*torch.randn_like(out).to(x)

            final_out = self.fc_layers(out) 
            mean = final_out[:,0]
            sigma = F.softplus(final_out[:,1])+1e-3
            return mean, sigma
            
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())  
            return final_out

    def custom_compile(self):
      # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0)
      self.optim_F = torch.optim.SGD([ {'params': self.downsampling_layers.parameters()}, {'params': self.drift.parameters()},
                                      {'params': self.fc_layers.parameters()}], lr=1e-5, momentum=0.9, weight_decay=5e-4)
      self.optim_G = torch.optim.SGD([ {'params': self.diffusion.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4)
      self.scheduler = None
      # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
      self.optimizer = [self.optim_F, self.optim_G]
      self.criterion = torch.nn.BCELoss()
      self.real_label = 0
      self.fake_label = 1
      return self.optimizer, self.scheduler


    def train_step(self,**kwargs):

      # preds, sigmas = kwargs['model'](kwargs['batch'][0])
      optim_F = kwargs['optim'][0]
      optim_G = kwargs['optim'][1]
      # scheduler_m = kwargs['scheduler'][0]
      # scheduler_v = kwargs['scheduler'][1]
      
      kwargs['model'].train()
      if len(kwargs['train_logs']) == 0:
        self.sigma = 0.1
      if len(kwargs['train_logs']) == 30:
        self.sigma = 0.5
      optim_F.zero_grad()
      prediction, sigma = self.forward(kwargs['batch'][0])
      loss = nll_loss(prediction, kwargs['batch'][1], sigma)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.parameters(), 100.)
      optim_F.step()
      label = torch.full((kwargs['batch_size'],1), self.real_label).to(torch.float32).to(self.device)
      optim_G.zero_grad()
      predict_in = self.forward(kwargs['batch'][0], training_diffusion=True).to(torch.float32)
      loss_in = self.criterion(predict_in, label)
      loss_in.backward()
      label.fill_(self.fake_label)

      inputs_out = 2*torch.randn(kwargs['batch_size'], self.n_features).to(self.device)+kwargs['batch'][0]
      predict_out = self.forward(inputs_out, training_diffusion=True)
      loss_out = self.criterion(predict_out, label)

      loss_out.backward()
      optim_G.step()
      return loss.item()

    def evaluation_step(self,**kwargs):
      test_iters = kwargs['test_iters']
      target_scale = kwargs['target_scale']
      test_preds_sum = 0.
      for i in range(test_iters):
        self.train()
        test_pred, _ = self.forward(kwargs['test_tuple'][0])
        test_preds_sum = test_preds_sum + test_pred
      test_preds = (test_preds_sum/test_iters)*target_scale
      test_score = kwargs['eval_func'](test_preds, kwargs['test_tuple'][1]*target_scale)
      scores = []
      for i in test_score:
        scores.append(i.detach().cpu().numpy())

      return scores
