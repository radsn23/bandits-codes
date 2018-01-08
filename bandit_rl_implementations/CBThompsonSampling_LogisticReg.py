import numpy as np 
import matplotlib.pyplot as plt 
import math
from scipy.optimize import minimize
import inspect
import seaborn as sns
import decimal
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
		

	def generate_weights(self,a):
		self.w = np.random.normal(self.m, a*self.q**(-1/2),size= d)
		return self.w

	def choose_version(self,x,w):
		scores = np.zeros(self.versions)
		for i in range((self.versions)):

			x[X_size + i] = 1
			scores[i] = generate_data.logprob(self,x,w)[1]
			x[X_size + i] = 0
		self.scores=scores	
		return scores.argmax()

	def get_regret(self,x,w,vers):
		#the best arm is 2?
		# if vers!=2:	
		# 	#global regret
		# 	regret+=1
		# return regret	
		mean_of_chosen_arm = self.m[X_size+vers]
		mean_of_best_arm = np.max(self.m[X_size+1:X_size+self.versions])
		return mean_of_best_arm - mean_of_chosen_arm



	
	def get_batch(self,x,version):
		x[X_size + version] = 1
		for j in range(X_size):
			x[X_size + self.versions + j + X_size * version] = x[j]
		return x	

	
	def generate_loss(self,w,x,y):
		return 0.5*(self.q*(w-self.m)).dot(w-self.m)+np.sum([np.log(1+np.exp(-1*(y)[j]*(w).dot((x)[j]))) for j in range((y).shape[0])]) 

 	
	def param_update(self,X,y):	
		self.m = minimize(self.generate_loss,self.w,args=(X,y),method='L-BFGS-B').x
		P = 1/(1 + np.exp(-1 * X.dot(self.m)))  #Laplace Approximation
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
	ax1 = fig.add_subplot(431)
	ax2 = fig.add_subplot(434)
	ax3 = fig.add_subplot(437)
	ax6 = fig.add_subplot(4,3,10)
	ax4 = fig.add_subplot(432)
	ax5 = fig.add_subplot(433)
	
	alphas = [1,2,0.5]
	for a in alphas:
		T = 500
		N = 10 #number of batches
		versions = 3 #or the number of arms of the bandit
		l=0.1 #lambda
		X_size = 3 
		#global regret
		regret=0
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
		regrets = []
		perc_regrets = []
		for t in range(T):

			X = np.zeros((N,d))
			y = np.zeros(N)
			
			for n in range(N):
				X[n] = data.get_X()
				weights = cbts.generate_weights(a) #Generate a prior on weights
				#print(data.logprob(X[n],weights))
				vers = cbts.choose_version(X[n],weights)	#From that distr, choose versions
				
				X[n] = cbts.get_batch(X[n],vers)		#Form a batch, with X, chosen arm, and rewards
				y[n] = data.get_y(X[n],weights)			# Get the rewards for each chosen arm using logprob.
			chosen_versions[t*N + n] = vers
			regret+= cbts.get_regret(X,weights,vers)
			regrets.append(regret)#/((t+1)*N)*100)
			perc_regrets.append(regret/((t+1)*N))
			loss= cbts.generate_loss(weights,X,y)
			#print(vers)
			
			t_array.append(t)
			loss_array.append(loss)	
			cbts.param_update(X,y)						#Update q,m to update the weights dis
			obtained_rewards[t*N:(t+1)*N] = y 			#store all the rewards, they should get better (=1) over iterations
			er = np.sum(obtained_rewards)/((t+1)*N)
			expected_reward.append(er)	
				
		#Plot of loss function
		k = alphas.index(a)+1
		plt.subplot(4,3,k)
		plt.plot(np.linspace(0,T,len(loss_array)),loss_array, label='Loss')
		plt.subplot(4,3,k+3)
		plt.plot(np.linspace(0,T,len(expected_reward)),expected_reward, color='m', label='Expected Reward')
		plt.subplot(4,3,k+6)
		plt.plot(np.linspace(0,T,len(regrets)),regrets, color='r',label='Cumulative Regret')
		plt.subplot(4,3,k+9)
		plt.plot(np.linspace(0,T,len(perc_regrets)),perc_regrets, color='orange',label='% Regret')
		#ax.scatter(np.linspace(0,T,len(chosen_rewards)),chosen_rewards)
	ax1.set_ylabel('Loss')
	ax1.set_xlabel('Time')
	ax1.legend()
	ax2.set_ylabel('Expected Reward')
	ax2.legend()
	ax2.set_xlabel('Time')
	ax3.set_ylabel('Cumulative Regret')
	ax3.legend()
	ax6.set_xlabel('Time')
	ax6.set_ylabel('% Regret')
	ax6.legend()
	ax3.set_xlabel('Time')
	ax1.set_title('alpha = 1')
	ax4.set_title('alpha = 2')
	ax5.set_title('alpha = 0.5')
	fig.tight_layout()
	plt.show()
			






			

	




