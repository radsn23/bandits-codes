from generate_single_bandit import *
from thompson_policy import *
from ucb1 import *
from random_policy import *
from epsilon_greedy import *
from LinUCB import *
from plot_graph import *

"""
Sample code for running contextual algorithms on
single-bandit data and plotting regret comparison.
"""

np.random.seed(0)

# input data file to create
data_file = 'contextual_single_bandit.csv'

num_actions = 3
num_rows = 4266

# result file for random policy
random_output_file = 'contextual_single_bandit_random.csv'

# result file for epsilon greedy
epsilon_output_file = 'contextual_single_bandit_epsilon_{}.csv'

# result file for thompson sampling
thompson_output_file = 'contextual_single_bandit_thompson_{}.csv'

# result file for LinUCB
linucb_output_file = 'contextual_single_bandit_linucb_{}.csv'

# The ground truth probability of each arm generating a reward.
# To add more arm, simply add a new value to this array.
true_probs = generate_probs(n1 = NUM_AGE_LEVEL, n2 = NUM_DAYS_LEVEL, num_actions = num_actions)

# generate data and write to file
generate_file(true_probs, num_rows, data_file)

# run random policy on the data file
calculate_random_single_bandit(data_file, num_actions, dest=random_output_file)

plot_source = [(random_output_file, "random")]
plot_thompson = []
plot_epsilon = []
plot_linucb = []

log_sigma_range = [1]
# run thompson sampling on the data file with different initial prior variance
for log_sigma in log_sigma_range:
    tof = thompson_output_file.format(log_sigma)
    lucbof = linucb_output_file.format(log_sigma)

    models_thompson = [RLogReg(D = NUM_FEATURES, Lambda = 1.0 / np.exp(log_sigma)) \
        for cond in range(num_actions)]
    calculate_thompson_single_bandit(data_file, num_actions, \
        dest=tof, models=models_thompson)

    models_linucb = [RLogReg(D = NUM_FEATURES, Lambda = 1.0 / np.exp(log_sigma)) \
        for cond in range(num_actions)]
    calculate_linucb_single_bandit(data_file, num_actions, \
        dest=lucbof, models=models_linucb)

    plot_source.append((tof, "thompson-logsigma={}".format(log_sigma)))
    plot_thompson.append((tof, "thompson-logsigma={}".format(log_sigma)))

    plot_source.append((lucbof, "linucb-logsigma={}".format(log_sigma)))
    plot_linucb.append((lucbof, "linucb-logsigma={}".format(log_sigma)))


for epsilon in [0.2]:
    for log_sigma in log_sigma_range:
        eof = epsilon_output_file.format('{}_eps_{}'.format(log_sigma, epsilon))
        models_epsilon = [RLogReg(D = NUM_FEATURES, Lambda = 1.0 / np.exp(log_sigma)) \
            for cond in range(num_actions)]
        calculate_epsilon_single_bandit(data_file, num_actions, \
            dest=eof, epsilon = epsilon, models=models_epsilon)
        plot_source.append((eof, "epsilon-logsigma={}-eps={}".format(log_sigma, epsilon)))
        plot_epsilon.append((eof, "epsilon-logsigma={}-eps={}".format(log_sigma, epsilon)))

# generate graph
plot_graph(plot_source, 'contextual-all.png')
plot_graph(plot_thompson, 'contextual-thompson.png')
plot_graph(plot_epsilon, 'contextual-epsilon.png')
plot_graph(plot_linucb, 'contextual-lincub.png')
