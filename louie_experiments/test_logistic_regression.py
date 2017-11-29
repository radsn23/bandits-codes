#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Logistic Regression 3-class Classifier
=========================================================

Show below is a logistic-regression classifiers decision boundaries on the
`iris <http://en.wikipedia.org/wiki/Iris_flower_data_set>`_ dataset. The
datapoints are colored according to their labels.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad
import matplotlib.pyplot as plt
import scipy.optimize as scopt
from scipy.special import expit
from sklearn import linear_model, datasets
from logistic_regression import *


add_bias = True

scikit = 0
autograd = 1
manual = 2
rlogreg = 3

opt_mode = scikit
opt_mode_str = ['scikit', 'autograd', 'manual', 'rlogreg']

def load_iris_data():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    h = .02  # step size in the mesh

    return X, Y, h


def load_random_data():

    num_examples = 1000
    threshold = 2

    if add_bias:
        X = np.random.randn(num_examples, 3)
        X[:, 2] = 1
    else:
        X = np.random.randn(num_examples, 2)

    Y = np.zeros(num_examples)
    Y[X[:, 0] >  threshold] = -1
    Y[X[:, 0] <= threshold] = 1

    h = .02

    return X, Y, h


def log_prob_and_gradient(w, sigma_squared, X, Y):
  
  T = -Y * np.dot(X, w)

  # compute the log-posterior of the data
  c1 = np.sum(np.dot(w, w)) / (2 * sigma_squared)
  c2 = np.sum(np.logaddexp(0, T))

  neg_log_posterior = c1 + c2

  # compute the gradient of the log posterior
  if opt_mode == manual:
    g1 = w / sigma_squared
    g2 = np.dot((expit(T) * Y).T, X)
    neg_gradient = g1 - g2
    return neg_log_posterior, neg_gradient

  #print 'log = {}, sum grad = {}'.format(neg_log_posterior, sum(abs(neg_gradient)))

  return neg_log_posterior


def compute_map(sigma_squared, X, Y):
  
  D = X.shape[1] # number of features
  num_iterations = 100
  w_0 = np.zeros(D)

  if opt_mode == scikit:
      # use scikit-learn logistic regression to optimize
      logreg = LogisticRegression(C = sigma_squared, solver = 'lbfgs', fit_intercept = True)
      logreg.fit(X, Y)
      w_map = logreg.coef_.ravel()

  elif opt_mode == autograd:
      # get a function which computes gradient of the log-posterior
      grad_logpost = value_and_grad(log_prob_and_gradient) #value_and_grad
      # SGD with mini-batch = 100 and using Adam
      #w_map = adam(grad_logpost, w_0, sigma_squared, X, Y, num_iters = num_iterations)

      # estimate minimum using l-bfgs-b method
      result = scopt.minimize(grad_logpost, w_0, args=(
        sigma_squared, X, Y), method='cg', jac=True, options={
          'maxiter': num_iterations, 'disp': False})
      w_map = result.x

  else:
      # estimate minimum using l-bfgs-b method
      result = scopt.minimize(log_prob_and_gradient, w_0, args=(
        sigma_squared, X, Y), method='cg', jac=True, options={
          'maxiter': num_iterations, 'disp': False})
      w_map = result.x

  return w_map


def predict(w, X):
    Y = expit(np.dot(X, w))
    Y[Y >= 0.5] =  1
    Y[Y <  0.5] = -1
    return Y


np.random.seed(0)

X, Y, h = load_random_data()

C = 1e5
sigma = C

## Plot the decision boundary. For that, we will assign a color to each
## point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

if add_bias:
    X_star = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]
else:
    X_star = np.c_[xx.ravel(), yy.ravel()]

if opt_mode == scikit:
    logreg = linear_model.LogisticRegression(C=1e5, fit_intercept = True)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)
    
    Z = logreg.predict(X_star)

elif opt_mode == rlogreg:
    lr = RLogReg(D = X.shape[1], Lambda = 1.0 / sigma)
    lr.set_data(X, Y)
    lr.compute_map()
    Z = lr.predict(X_star)
    
else:
    weights = compute_map(sigma, X, Y)
    Z = predict(weights, X_star)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
