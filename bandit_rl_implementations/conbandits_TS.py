import numpy as np 
import matplotlib.pyplot as plt 
import math
from scipy.optimize import minimize
import inspect
import seaborn as sns
np.random.seed(23)

"""
An implementation of An Empirical Evaluation of Thompson Sampling - O Chappelle and Lihong Li
NIPS'11 Proceedings of the 24th International Conference on Neural Information Processing Systems
(https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf)
"""

class CB_TS_Logistic:
	def __init__(self,l,versions,d,w=None):
		self.l=l
		self.d=d
		self.w=w
		self.m = np.zeros(d)
		self.q = np.ones(d)*l
		self.versions=versions

	def generate_weights(self):
		self.w = np.random.normal(self.m, self.q**(-1),size= d)
		return self.w

	def choose_version(self,x,w):
		scores = np.zeros(self.versions)
		for i in range((self.versions)):

			x[X_size + i] = 1
			scores[i] = generate_data.logprob(self,x,w)[1]
			x[X_size + i] = 0
		return scores.argmax()
	
	def get_batch(self,x,version):
		x[X_size + version] = 1
		for j in range(X_size):
			x[X_size + self.versions + j + X_size * version] = x[j]
		return x	

	
	def generate_loss(self,w,x,y):
		return 0.5*(self.q*(w-self.m)).dot(w-self.m)+np.sum([np.log(1+np.exp(-1*(y)[j]*(w).dot((x)[j]))) for j in range((y).shape[0])]) 

 	
	def param_update(self,X,y):	
		self.m = minimize(self.generate_loss,self.w,args=(X,y)).x
		P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1) #Laplace Approximation
		self.q = self.q + (P*(1-P)).dot(X ** 2)
					

class generate_data:
	def __init__(self,d,X_size,w):
		self.d = d
		self.X_size = X_size
		self.w=w

	def get_X(self):
		X = np.zeros(self.d)
		X[:self.X_size] = np.random.randn(self.X_size)
		self.X=X
		return X

	def logprob(self,x,w):
		prob = 1 / (1 + np.exp(-1*(x).dot(w)))
		return np.array([1-prob,prob]).T
	

	def get_y(self,x,w):
		self.y=y
		return np.random.binomial(1,self.logprob(x,w)[1])		


if __name__== "__main__":
	
	fig = plt.figure(figsize=(10, 6))
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2,1,2)
	T = 100 
	N = 10 #number of batches
	versions = 3 #or the number of arms of the bandit
	l=0.1 #lambda
	X_size = 3 

	# The set of past observations is made of triplets (x_i,a_i,r_i), so the dimension of the observation is-
	d = X_size + versions + versions*X_size 
	w= np.random.normal(d) #Initialising the weights
	chosen_versions = np.zeros(T*N)
	obtained_rewards = np.zeros(T*N)
	t_array=[]
	loss_array=[]
	expected_reward=[]
	data = generate_data(d,X_size,w)
	cbts = CB_TS_Logistic(l,versions,d,w)

	for t in range(T):

		X = np.zeros((N,d))
		y = np.zeros(N)

		for n in range(N):
			X[n] = data.get_X()
			weights = cbts.generate_weights() #Generate a prior on weights
			vers = cbts.choose_version(X[n],weights)	#From that distr, choose versions
			X[n] = cbts.get_batch(X[n],vers)		#Form a batch, with X, chosen arm, and rewards
			y[n] = data.get_y(X[n],weights)			# Get the rewards for each chosen arm using logprob.
			chosen_versions[t*N + n] = vers
		loss= cbts.generate_loss(weights,X,y)
		t_array.append(t)
		loss_array.append(loss)	
		cbts.param_update(X,y)						#Update q,m to update the weights dis
		obtained_rewards[t*N:(t+1)*N] = y 			#store all the rewards, they should get better (=1) over iterations
		er = np.sum(obtained_rewards)/t
		expected_reward.append(er)	
		
		#Plot of loss function
		plt.subplot(2,1,1)
		plt.plot(np.linspace(0,T,len(loss_array)),loss_array, label='Loss')
		plt.subplot(2,1,2)
		plt.plot(np.linspace(0,T,len(expected_reward)),expected_reward, color='m', label='Why does it dip in the start?')
		#ax.scatter(np.linspace(0,T,len(chosen_rewards)),chosen_rewards)
		ax1.set_ylabel('Loss')
		ax1.set_xlabel('Time')
		ax1.legend()
		ax2.set_ylabel('Expected Reward')
		ax2.legend()
		ax2.set_xlabel('Time')
		plt.show()
			






			

	




