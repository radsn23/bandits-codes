from generate_single_bandit import *
from epsilon_greedy import *
from thompson_policy import *
from ucb1 import *
from random_policy import *
from plot_graph import *

"""
Sample code for running non-contextual algorithms on
single-bandit data and plotting regret comparison.
"""

np.random.seed(0)

# input data file to create
data_file = 'simulated_single_bandit_input.csv'

# result file for random policy
random_output_file = 'simulated_single_bandit_random_r{}.csv'

# result file for epsilon greedy
epsilon_output_file = 'simulated_single_bandit_epsilon_r{}.csv'

# result file for thompson sampling
thompson_output_file = 'simulated_single_bandit_thompson_r{}.csv'

# result file for UCB1
ucb1_output_file = 'simulated_single_bandit_ucb1_r{}.csv'

# The ground truth probability of each arm generating a reward.
# To add more arm, simply add a new value to this array.
true_probs = np.array([0.2, 0.4, 0.5])

# number of rows to generate
num_rows = 4266

num_runs = 10

# generate data file
generate_file(true_probs, num_rows, data_file)

for r in range(num_runs):
    print("Run {}".format(r + 1))

    # run random policy on the data file
    calculate_random_single_bandit(data_file, num_actions=3, dest=random_output_file.format(r + 1))

    # run epsilon greedy on the data file
    eps = 0.2
    calculate_epsilon_single_bandit(data_file, num_actions=3, dest=epsilon_output_file.format(r + 1), epsilon = eps)

    # run thompson sampling on the data file
    calculate_thompson_single_bandit(data_file, num_actions=3, dest=thompson_output_file.format(r + 1))

    # run UCB1 on the data file
    calculate_ucb1_single_bandit(data_file, num_actions=3, dest=ucb1_output_file.format(r + 1))


output_randoms = []
output_epsilons = []
output_thompsons = []
output_ucb1s = []

for r in range(num_runs):
    output_randoms.append(random_output_file.format(r + 1))
    output_epsilons.append(epsilon_output_file.format(r + 1))
    output_thompsons.append(thompson_output_file.format(r + 1))
    output_ucb1s.append(ucb1_output_file.format(r + 1))

# generate graph
source = [(output_randoms, "random"),
          (output_epsilons, "epsilon = {}".format(eps)),
          (output_thompsons, "thompson"),
          (output_ucb1s, "ucb1")]
plot_graph_average(source, dest = "single-all.png", title = "Single Bandit Comparison averaged over {} runs".format(num_runs))