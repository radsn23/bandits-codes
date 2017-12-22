import numpy as np 
import matplotlib.pyplot as plt 
import math
from scipy.optimize import minimize
np.random.seed(23)

class CB_TS_Logistic:
	def __init__(self, l, versions, X, y, d, w=None): 
		self.l = l
		self.versions = versions
		self.X = X
		self.y = y
		self.w = w
		self.d = d 
		self.m = np.zeros(d)
		self.q = np.ones(d)*l
	
	# Algorithm 3 code

 	def generate_weights(self):
 		self.w = np.random.normal(self.m, self.q**(-1), size = self.d)
 		return self.w

 	def generate_loss(self,X,y):
 		return 0.5 * (self.q * (self.w - self.m)).dot(self.w - self.m) + np.sum([np.log(1+np.exp(-1 * y[j] * (self.w).dot(X[j]))) for j in range(y.shape[0])])	

 	def param_update(self,X,y):
 		self.m = minimize(self.generate_loss,self.w,args=(X,y)).x
        P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1) #Laplace Approximation
        self.q = self.q + (P*(1-P)).dot(X ** 2)

class generate_data:
	def __init__(self,d,X_size,w=None):
		self.d = d
		self.X_size = X_size
		self.w  = w

	def sample_X(self):
		X = np.zeros(self.d)
		X[:self.X_size] = np.random.randn(size=self.X_size)
		return X

	def logprob(self):
		prob = 1 / (1 + np.exp(-1*(self.X).dot(self.w)))
		return np.array([1-prob,prob]).T

	def predict(self):
		prob=  self.logprob()
		p = np.zeros_like(prob)
		p[prob>=0.5] = 1
		return p

	def sample_y(self):
		return np.random.binomial(1,self.logprob()[1])

	def get_features(self,version):
		self.X[self.X_size+]	



T = 50
n = 10
versions = 3
l = 0.1
X_size = 3
X = np.random.rand(size=X_size)
d = X_size + versions + versions*X_size
version = np.random.randint(size=versions)

chosen_versions = np.zeros(T*n)
chosen_rewards = np.zeros(T*n)

# for t in range(T):









