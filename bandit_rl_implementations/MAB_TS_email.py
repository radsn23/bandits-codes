import numpy as np 
import matplotlib.pyplot as plt 
import math
from scipy.stats import beta
"""
Multi Armed Bandit problem using Thompson Sampling for a simple Email MOOClet example
"""

class email_mooclet:
	def __init__(self,versions=None, prior = None):					#Taking in version set and prior as input
		self.trials = np.zeros((len(versions),), dtype = int)
		self.successes = np.zeros_like(self.trials)
		self.versions = versions
		if prior == None:
			self.prior = [(1.0,1.0) for i in range(len(versions))]

	def add_data(self, version_num, success):
		self.trials[version_num]+= 1
		if success:
			self.successes[version_num]+=1		

	def personalize(self):
		posterior_sample = np.zeros(len(self.versions))
		x = []
		params = []
		for i in range(len(self.versions)):			
			a = prior[i][0]+ self.successes[i] 						#alpha
			b = prior[i][1] + self.trials[i] - self.successes[i] 	#beta
			params.append([a,b])									#appending these parameters for plotting graphs
			x += [np.linspace(beta.ppf(0.01, a,b), beta.ppf(0.99, a,b), 100)]
			posterior_sample[i] = np.random.beta(a,b) 				#choosing a random sample from the beta distribution
			
		return np.argmax(posterior_sample),x,params 				#returning the maximum sample's version_id, alongwith a few plotting stuff

plt.figure(figsize=(20, 6))
plt.grid()
prior= [[4,33],[2,31],[4,50]]
versions = ['Survey','Brief','Acknowledgement']
trials_in = 1000
scores = [0,0,0]
tried_outputs = [0,0,0] 
params = []

# Plotting the initial prior graphs
for i in range(len(versions)):										
	plt.subplot(3,2,2*i+1)
	x = np.linspace(beta.ppf(0.01, prior[i][0],prior[i][1]), beta.ppf(0.99, prior[i][0],prior[i][1]), 100)
	plt.plot(x, beta.pdf(x, prior[i][0],prior[i][1]), label='beta pdf')
	plt.title(versions[i] + '- prior')
	plt.xlim([0,1])

# Now creating the posterior distribution as data is added. 
for trial in range(trials_in):
	e = email_mooclet(versions,prior)
	input_version = np.random.randint(len(versions))		#Choosing a version at random
	tried_outputs[input_version] +=1	
	#e.add_data(input_version,np.random.randint(2))		# Un-comment this to add successes randomly				
	#e.add_data(input_version,(np.random.choice(np.arange(len(versions)),p = [0.6,0.1,0.3])== input_version)) #Uncomment this to add successes based on a probability distr to simulate actual patterns	
	result,x,prior = e.personalize()
	scores[result]+=1	
	
		   
print(scores)													# Checking how many times each version won the sampling contest
print(tried_outputs)											# Checking how many times each version was chosen to add a success to
for i in range(len(versions)):									# As one particular version gets chosen more, it's probability of
  	plt.subplot(3,2,(i+1)*2)									# winning should increase. Here versions chosen at random
  	plt.plot(x[i], beta.pdf(x[i], prior[i][0],prior[i][1]), label='beta pdf')
  	plt.title(versions[i])	
  	plt.xlim([0,1])	
  	max_y = max(beta.pdf(x[i], prior[i][0],prior[i][1]))
  	max_x = x[i][max_y.argmax()]
  	plt.text(max_x, max_y, str((max_x)))								# Plotting final posterior graph
plt.tight_layout()

plt.show()


