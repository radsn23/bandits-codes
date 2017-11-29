'''
Created on Aug 22, 2016

@author: rafferty
'''
import numpy as np
import beta_bernoulli

# y1 ~ N(mu1, sigma1)
# y2 ~ N(mu2, sigma2)
# y = y1 + y2
# y1 might be relatability
# y2 might be actual impact on learning
# 
# We first are optimizing for reward y (might be collected as a rating)
# We then want to just collect based on later success
# - Could have later success just be y2, or could make it y4, where:
# y3 ~ N(mu3, sigma3) (prior ability)
# y4 ~ y3 + y2 (ability after explanation)
# 
# Probably want to turn all of these binary for the current model - get a real valued number and then sigmoid coin flip?
# - Additional wrinkle with second model is whether differences in prior ability could actually lead to lower final rewards even when measured on y4 when optimizing y4 vs y2; that would be interesting
# 
# Correlation factor that we could control might be size of y1 versus y2 mus (likely just set sigma/draw from some simple distribution)
# Ratio: mu1/mu2 : when high, y may be very distant from y4; when low, y and y4 should be similar (although also dependent on scale of mu3; worry about that later)

def generate_two_factor_rewards(num_actions, num_samples, lower_bound, upper_bound, variance_factor_2):
    '''
    Generates a reward file for two bandits. Idea is that first bandit has means based
    on two factors; later bandit only includes one factor.
    
    1. Select means for arms. [Sample uniformly on [lower_bound, upper_bound]].
    2. Select values for clarity/relatibility. [Set variance]
    3. Bandit 1 has clarity + mean as average, bandit 2 has mean as average.
    
    For now, just leave as two separate sets of probabilities, like MVN.
    
    Outputs a 2*num_actions columns x num_samples rows array where the first num_actions columns
    are the means for the first bandit and the second num_actions columns are
    the means for the second bandit.
    '''
    epsilon = .001 # Minimum value of a mean; maximum is 1-epsilon. I.e., arms can't be 100% on or off
    means_factor_1 = np.random.uniform(lower_bound, upper_bound, size=(num_samples, num_actions))
    clarity_factors = np.random.normal(scale = variance_factor_2, size=(num_samples, num_actions))

    means = np.append(means_factor_1 + clarity_factors, means_factor_1, axis = 1)
    means[means <= 0] = epsilon
    means[means >= 1] = 1 - epsilon
    return means

def generate_two_factor_rewards_with_prior_skill(num_actions, num_steps, lower_bound, 
                                                 upper_bound, variance_factor_2, prior_skill_dist,
                                                 scale_factor = .1):
    '''
    Generates a reward file for two bandits. Idea is that first bandit has means based
    on two factors; later bandit only includes one factor and prior skill.
    
    Note that this means that unlike MVN rewards, we can't just return a vector of
    arm means, since samples should be different for different users. Instead, we take
    in a parameter prior_skill_dist which allows us to sample a prior understanding level 
    for a user.
    
    
    Generation process is as follows:
    1. Select means for arms. [Sample uniformly on [lower_bound, upper_bound]].
    2. Select values for clarity/relatibility. [Set variance]
    3. Bandit 1 has clarity + mean as average, bandit 2 is a coin flip with weight
    prior_skill + mean*.1. .1 here is a scaling factor.
    
    num_steps corresponds to the number of steps to sample for this bandit - we only
    sample one bandit at a time.
    '''
    epsilon = .001 # Minimum value of a mean; maximum is 1-epsilon. I.e., arms can't be 100% on or off
    means_factor_1 = np.random.uniform(lower_bound, upper_bound, size=(1, num_actions))
    clarity_factors = np.random.normal(scale = variance_factor_2, size=(1, num_actions))

    means = np.append(means_factor_1 + clarity_factors, means_factor_1, axis = 1)
    means[means <= 0] = epsilon
    means[means >= 1] = 1 - epsilon
    
    # Now we go through each trial
    samples = np.zeros((num_steps, 2*num_actions))
    for i in range(num_steps):
        # Sample first arm rewards
        for action_index in range(num_actions):
            if np.random.rand() < means[action_index]:
                samples[i][action_index] = 1
        
        # Sample second arm rewards - requires sampling the skill level
        for action_index in range(num_actions):
            if np.random.rand() < means[action_index + num_actions]*scale_factor + prior_skill_dist.sample():
                samples[i][action_index + num_actions] = 1
                
                 
    return samples


def main():
    # Primarily for debugging
    means = generate_two_factor_rewards(3, 500, .45, .55, .1)
    print(means)
    
if __name__ == "__main__":
    main()
    