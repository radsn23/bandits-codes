from generate_single_bandit import *
from thompson_policy import *
from random_policy import *
from ucb1 import *
from plot_graph import *

np.random.seed(0)

# input data files to create
immediate_input = 'simulated_single_bandit_input.csv'
true_input = 'simulated_single_bandit_input_true.csv'

# The ground truth probability of each arm generating a reward.
# To add more arm, simply add a new value to this array.
true_probs = np.array([0.2, 0.9, 0.3])

# The immediate approximate probability of each arm generating a reward.
immediate_probs = np.array([0.2, 0.3, 0.9])

num_rows = 4266

# generate data and write to file
generate_file(immediate_probs, num_rows, immediate_input)
generate_file(true_probs, num_rows, true_input)

time_steps = np.arange(0, 201, 50) + 20

prob_flavor = '-'.join(str(p * 10) for p in true_probs)

for t in time_steps:
    # run random policy switching bandits between
    # immediate and true reward inputs
    imm_output_random = 'output_random_immediate_{}.csv'.format(t)
    true_output_random = 'output_random_true_{}.csv'.format(t)
    switch_bandit_random(immediate_input, true_input,
        imm_output_random, true_output_random,
        time_step = t, num_actions = len(true_probs))

    # run prob-best thompson sampling switching bandits
    # between immediate and true reward inputs
    imm_output_thompson = 'output_thompson_immediate_{}.csv'.format(t)
    true_output_thompson = 'output_thompson_true_{}.csv'.format(t)
    switch_bandit_thompson(immediate_input, true_input,
        imm_output_thompson, true_output_thompson,
        time_step = t, action_mode = ActionSelectionMode.prob_is_best,
        num_actions = len(true_probs))

    # run relative-reward thompson sampling switching
    # bandits between immediate and true reward inputs
    imm_output_rel_thompson = 'output_relative_thompson_immediate_{}.csv'.format(t)
    true_output_rel_thompson = 'output_relative_thompson_true_{}.csv'.format(t)
    switch_bandit_thompson(immediate_input, true_input,
        imm_output_rel_thompson, true_output_rel_thompson,
        time_step = t, action_mode = ActionSelectionMode.expected_value,
        num_actions = len(true_probs))

    # run UCB1 switching bandits between immediate and true reward inputs
    imm_output_ucb1 = 'output_ucb1_immediate_{}.csv'.format(t)
    true_output_ucb1 = 'output_ucb1_true_{}.csv'.format(t)
    switch_bandit_ucb1(immediate_input, true_input,
        imm_output_ucb1, true_output_ucb1,
        time_step = t,
        num_actions = len(true_probs))

    max_step_to_plot = t + 100

     # generate graph ONLY on immediate data 
    source = [(imm_output_random, "random-immediate-{}".format(t)),
              (imm_output_ucb1, "ucb1-immediate-{}".format(t)),
              (imm_output_thompson, "thompson-immediate-{}".format(t)),
              (imm_output_rel_thompson, "relative-thompson-immediate-{}".format(t))]
    plot_graph(source, dest = "immediate-{}.png".format(t), max_step = max_step_to_plot, title_suffix = prob_flavor)

    # generate graph ONLY on delayed data 
    source = [(true_output_random, "random-true-{}".format(t)),
              (true_output_ucb1, "ucb1-true-{}".format(t)),
              (true_output_thompson, "thompson-true-{}".format(t)),
              (true_output_rel_thompson, "relative-thompson-true-{}".format(t))]
    plot_graph(source, dest = "true-{}.png".format(t), max_step = max_step_to_plot, title_suffix = prob_flavor)

    # generate graph comparing thompson immediate vs delayed 
    source = [(imm_output_thompson, "thompson-immediate-{}".format(t)),
              (true_output_thompson, "thompson-true-{}".format(t))]
    plot_graph(source, dest = "thompson-{}.png".format(t), max_step = max_step_to_plot, title_suffix = prob_flavor)

    # generate graph comparing relative thompson immediate vs delayed 
    source = [(imm_output_rel_thompson, "relative-thompson-immediate-{}".format(t)),
              (true_output_rel_thompson, "relative-thompson-true-{}".format(t))]
    plot_graph(source, dest = "relative-thompson-{}.png".format(t), max_step = max_step_to_plot, title_suffix = prob_flavor)

    # generate graph comparing UCB1 immediate vs delayed 
    source = [(imm_output_ucb1, "ucb1-immediate-{}".format(t)),
              (true_output_ucb1, "ucb1-true-{}".format(t))]
    plot_graph(source, dest = "ucb1-{}.png".format(t), max_step = max_step_to_plot, title_suffix = prob_flavor)

