# Creating an Email MOOClet using Thompson Sampling

This is just a basic python implementation of how an Email MOOClet works.The implementation uses Thompson Sampling on an MAB problem. It works by maintaining a prior for each arm, sampling values from them and then updating the prior based on the obtained reward. This procedure is then repeated for the next arm pull. Using a Beta distribution for setting priors for this proves to be intuitive and convenient, since for every success/failure, we only need to add 1 to a or ß in the Beta distribution.

				(B(a,ß) B(a+1,ß) for a success, or B(aß,)B(a,ß+1) for a failure) 

The code snippet below is a class for the Email MOOClet that allows you to add data to it and use it to update the Beta distribution parameters. (This is very similar to the code described in this [blog](https://www.chrisstucchio.com/blog/2013/bayesian_bandit.html)) 

For the email experiment, we have three versions of subject lines (Survey, Brief, Acknowledgement) that we send to users of a particular MOOC. We need to know which subject line generates the highest response rate from these users. 

The Email MOOClet takes as input the version set, as well as prior probabilities that we ‘think’ would be consistent with the results. This prior would then be updated as the data comes in. For each version, we update the number of trials and successes as the data is added in function add_data.

The function personalize is responsible for carrying out this update of the priors, as well as sampling randomly from all three versions and returning the version_id with the maximum value. From our initial distributions, any of the three samples could be maximum depending on our priors.
As data is added however, the probability distributions become sharper, and the samples are more indicative of the actual probabilities of responses.  



'''python

class email_mooclet:
	def __init__(self,versions=None, prior = None):	#Taking in version set and prior as input
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
				a = prior[i][0]+ self.successes[i] 			#alpha
				b = prior[i][1] + self.trials[i] - self.successes[i] 	#beta
				params.append([a,b])		#appending these parameters for plotting graphs
				x += [np.linspace(beta.ppf(0.01, a,b), beta.ppf(0.99, a,b), 100)]
				posterior_sample[i] = np.random.beta(a,b) 	#choosing a random sample from the beta distribution
			
			return np.argmax(posterior_sample),x,params 				#returning the maximum sample's version_id, alongwith a few plotting stuff 


'''

## Results

The prior probabilities are plotted below, along with the posteriors obtained after 1000 iterations. I took the priors [[4,33],[2,31],[4,50]], where for each version, we have [a, ß] corresponding to the number of successes and failures, respectively. Beta distributions with these priors are plotted on the left.

After adding 1000 data points, the resulting distributions are plotted on the right, where the number on the top shows the starting points. 

![result1](https://github.com/radsn23/bandits-codes/blob/master/bandit_rl_implementations/EmailMAB/MAB_TS_posteriors.png)

We can see that as the distributions become sharper, the likelihood of each random sample actually representing the actual probability distribution increases.

Since the data in this experiment is taken randomly( where I just select a version at random and add a success or failure to it), the posterior distributions might not actually show a clear winner. In actual simulations with real data, the algorithm will be able to learn which subject line works best.

Alternative to random input: While giving inputs, you can decide which versions you want to give more successes to (in order to simulate an actual distribution of user preferences to versions). In the graphs below, I set the probabilities of success of [‘Survey’, ‘Brief’, ‘Acknowledgement’] to [0.6,0.1,0.3]. The plots obtained were:



