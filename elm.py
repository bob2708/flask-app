from scipy.linalg import pinv2
import scipy.stats as stats
import numpy as np

class Extreme():
  
  def __init__(self, nodes=100):
    self.hidden_size = nodes
  
  def __str__(self):
    return 'ExtremeLearningMachine()'

  def fit(self, X_train, y_train):
    self.X = X_train
    self.y = y_train
    self.input_size = X_train.shape[1]
    self.input_weights, self.biases = self._init_input_weights(self.input_size, self.hidden_size)
    self.output_weights = np.dot(pinv2(self._hidden_nodes(X_train)), y_train)

  def _init_input_weights(self, input_size, hidden_size):
    mu, sigma = 0, 1
    w_lo = -1 
    w_hi = 1
    b_lo = -1 
    b_hi = 1

    #initialising input weights and biases randomly drawn from a truncated normal distribution
    input_weights = stats.truncnorm.rvs((w_lo - mu) / sigma, (w_hi - mu) / sigma, loc=mu, scale=sigma,size=[input_size,hidden_size])
    biases = stats.truncnorm.rvs((b_lo - mu) / sigma, (b_hi - mu) / sigma, loc=mu, scale=sigma,size=[hidden_size])
    return input_weights, biases
  
  def _relu(self, x):
    return np.maximum(x, 0, x)

  def _hidden_nodes(self, X):
    G = np.dot(X, self.input_weights)
    G = G + self.biases
    H = self._relu(G)
    return H

  def predict(self, X):
    out = self._hidden_nodes(X)
    out = np.dot(out, self.output_weights)
    return out