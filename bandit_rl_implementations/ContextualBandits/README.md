# Contextual Bandits using Thompson Sampling

Referenced from [An Empirical Evaluation of Thompson Sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf) - Li et. al

## Basic Description
This implementation is of an Associative/Contextual Multi-Armed Bandit problem using Logistic Regression to model the rewards. The goal is to maximise the rewards obtained over time.

In this implementation, a Gaussian prior is set on the weights, and the mean and variance is updated each round based on the rewards observed. The rewards are modeled by the formula - 


![res1](https://github.com/radsn23/bandits-codes/blob/master/bandit_rl_implementations/ContextualBandits/Screen Shot 2018-01-19 at 2.21.57 PM.png)


The distribution is updated using the following algorithm -

![res2](https://github.com/radsn23/bandits-codes/blob/master/bandit_rl_implementations/ContextualBandits/Screen Shot 2018-01-19 at 1.45.58 PM.png)

*Steps-*
 
1. You generate data X, sample rewards (different ways of doing this)
2. Taking m=0, q=lambda, generate a weight distribution
3. For each round, minimize negative log likelihood, and find m,q that minimize loss
4. Substitute m,q into the weight distribution fuction, get rewards.

Changes to make: - Data generation methods and Linear Regression Trial

	1.Currently the Gaussian is serving as both the prior as well as the Thompson Sampling distribution. This is workaround artificial data collection.	
	Other methods of testing code-

 		1.Adding in importance to different contexts by changing the reward fn (like in MAB code)

		2.Trying out an actual dataset like calc_applied (lot of data_cleaning in that case)

	2.Make the code apply to linear regression, since all rewards are not binary

        3. Make code more modular to apply to all datasets. 
