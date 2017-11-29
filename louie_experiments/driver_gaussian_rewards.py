from gaussian_reward import *
from generate_single_bandit import *
from thompson_policy import *
from random_policy import *
from ucb1 import *
from plot_graph import *

np.random.seed(0)

# input data files to create
gauss_input = 'gauss_single_bandit_input_{}.csv'

# create output covariance matrix
mvn_output = 'gauss_mvn_parameters.csv'

# result file for random policy
random_output_file = 'gauss_single_bandit_random_{}.csv'

# result file for thompson sampling
thompson_output_file = 'gauss_single_bandit_thompson_{}.csv'

# result file for UCB1
ucb1_output_file = 'gauss_single_bandit_ucb1_{}.csv'

num_rows = 4266
num_actions = 3
num_bandits = 3

""" 
Generate rewards for 2 bandits from a multivariate gaussian
See generate_gaussian_rewards method for more explanations
"""

# Covariance matrix
c = \
{
    # cov for arm 1 between bandit 1,2 = 1 and 2,3 = -1
    1: [ [1, 2, 1], [2, 3, -1] ],
    # cov for arm 2 between bandit 1,2 = -1 and 1,3 = 0.2
    2: [ [1, 2, -1], [1, 3, 2] ],
    # cov for arm 3 between bandit 1,2 = 0.5
    3: [ [1, 2, -5] ]
}

# Variance for each arm in each bandit
v = np.ones(num_bandits * num_actions) * 0.1

probs = generate_gaussian_rewards(num_bandits, num_actions, 0.5, c, v, num_rows, mvn_output)

# TODO: normalize probs properly
probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))

# generate data and write to file
plot_source = []
plot_source_thompson = []
plot_source_random = []
plot_source_ucb = []
for i in range(num_bandits):
    in_file = gauss_input.format(i + 1)
    random_out_file = random_output_file.format(i + 1)
    thompson_out_file = thompson_output_file.format(i + 1)
    ucb1_out_file = ucb1_output_file.format(i + 1)

    true_probs = probs[:, i * num_actions : (i + 1) * num_actions]
    generate_file(true_probs, num_rows, in_file)

    calculate_random_single_bandit(in_file, num_actions, random_out_file)
    calculate_thompson_single_bandit(in_file, num_actions, thompson_out_file)
    calculate_ucb1_single_bandit(in_file, num_actions, ucb1_out_file)

    plot_source_thompson.append((thompson_out_file, "thompson_{}".format(i + 1)))
    plot_source_random.append((random_out_file, "random_{}".format(i + 1)))
    plot_source_ucb.append((ucb1_out_file, "ucb1_{}".format(i + 1)))

    plot_source.append(plot_source_thompson[-1])
    plot_source.append(plot_source_random[-1])
    plot_source.append(plot_source_ucb[-1])

# generate graph
plot_graph(plot_source)
plot_graph(plot_source_thompson)
plot_graph(plot_source_random)
plot_graph(plot_source_ucb)
