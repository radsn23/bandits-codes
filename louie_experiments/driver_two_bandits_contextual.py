from generate_single_bandit import *
from thompson_policy import *
from random_policy import *
from LinUCB import *
from epsilon_greedy import *
from plot_graph import *

np.random.seed(0)

num_actions = 3

# input data files to create
immediate_input = 'contextual_single_bandit.csv'
true_input = 'contextual_single_bandit_true.csv'

# output files for different algorithms
imm_output_format_random =   'contextual_switch_random_immediate{}_s{}.csv'
imm_output_format_thompson = 'contextual_switch_thompson_immediate_{}.csv'
imm_output_format_epsilon =  'contextual_switch_epsilon_immediate_{}.csv'
imm_output_format_linucb =   'contextual_switch_linucb_immediate_{}.csv'

true_output_format_random =   'contextual_switch_random_true_{}.csv'
true_output_format_thompson = 'contextual_switch_thompson_true_{}.csv'
true_output_format_epsilon =  'contextual_switch_epsilon_true_{}.csv'
true_output_format_linucb =   'contextual_switch_linucb_true_{}.csv'

# The ground truth probability of each arm generating a reward.
# To add more arm, simply add a new value to this array.
true_probs = generate_probs(n1 = 5, n2 = 8, num_actions = num_actions)

# The immediate approximate probability of each arm generating a reward.
immediate_probs = perturb_probs(true_probs)

num_rows = 1500

# generate data and write to file
generate_file(immediate_probs, num_rows, immediate_input)
generate_file(true_probs, num_rows, true_input)

time_steps = [500]
num_runs = 1

Lambda = 1 / np.exp(1)

for t in time_steps:
    for n in range(num_runs):
        # run random policy switching bandits between
        # immediate and true reward inputs
        imm_output_random = imm_output_format_random.format(t,n)
        true_output_random = true_output_format_random.format(t,n)
        switch_bandit_random(immediate_input, true_input,
            imm_output_random, true_output_random,
            time_step = t, num_actions = num_actions)

        # run contextual epsilon greedy switching bandits between
        # immediate and true reward inputs
        imm_output_epsilon = imm_output_format_epsilon.format(t,n)
        true_output_epsilon = true_output_format_epsilon.format(t,n)
        switch_bandit_epsilon(immediate_input, true_input,
            imm_output_epsilon, true_output_epsilon,
            time_step = t, use_regression = True, num_actions = num_actions, Lambda = Lambda)

        # run prob-best thompson sampling switching bandits
        # between immediate and true reward inputs
        imm_output_thompson = imm_output_format_thompson.format(t,n)
        true_output_thompson = true_output_format_thompson.format(t,n)
        switch_bandit_thompson(immediate_input, true_input,
            imm_output_thompson, true_output_thompson,
            time_step = t, action_mode = ActionSelectionMode.prob_is_best,
            use_regression = True, num_actions = num_actions, Lambda = Lambda)

        # run LinUCB switching bandits between immediate and true reward inputs
        imm_output_linucb = imm_output_format_linucb.format(t,n)
        true_output_linucb = true_output_format_linucb.format(t,n)
        switch_bandit_linucb(immediate_input, true_input,
            imm_output_linucb, true_output_linucb,
            time_step = t,
            num_actions = num_actions, Lambda = Lambda)


graph_title = 'Cumulative Regret from Single Sample as a function of action timestep'

for t in time_steps:
    max_step_to_plot = num_rows

    imm_random = []
    imm_thompson = []
    imm_epsilon = []
    imm_linucb = []
    true_random = []
    true_thompson = []
    true_epsilon = []
    true_linucb = []

    # aggregate all output files from all algorithms on immediate and delayed inputs
    # in order to compute and plot average cumulative regret
    for n in range(num_runs):
        imm_random.append(imm_output_format_random.format(t,n))
        imm_thompson.append(imm_output_format_thompson.format(t,n))
        imm_epsilon.append(imm_output_format_epsilon.format(t,n))
        imm_linucb.append(imm_output_format_linucb.format(t,n))

        true_random.append(true_output_format_random.format(t,n))
        true_thompson.append(true_output_format_thompson.format(t,n))
        true_epsilon.append(true_output_format_epsilon.format(t,n))
        true_linucb.append(true_output_format_linucb.format(t,n))

    # generate graph ONLY on immediate data 
    source = [(imm_random,   "random-immediate-{}".format(t)),
              (imm_epsilon,  "epsilon-immediate-{}".format(t)),
              (imm_linucb,   "linucb-immediate-{}".format(t)),
              (imm_thompson, "thompson-immediate-{}".format(t))]
    plot_graph_average(source, dest = "contextual-switch-immediate-{}.png".format(t),
                max_step = max_step_to_plot, title = graph_title, vertical_line = t)

    # generate graph ONLY on delayed data 
    source = [(true_random,   "random-true-{}".format(t)),
              (true_epsilon,  "epsilon-true-{}".format(t)),
              (true_linucb,   "linucb-true-{}".format(t)),
              (true_thompson, "thompson-true-{}".format(t))]
    plot_graph_average(source, dest = "contextual-switch-true-{}.png".format(t),
                max_step = max_step_to_plot, title = graph_title, vertical_line = t)

    # generate graph comparing epsilon greedy immediate vs delayed 
    source = [(imm_epsilon,  "epsilon-immediate-{}".format(t)),
              (true_epsilon, "epsilon-true-{}".format(t)),]
    plot_graph_average(source, dest = "contextual-switch-epsilon-{}.png".format(t),
                max_step = max_step_to_plot, title = graph_title, vertical_line = t)

    # generate graph comparing thompson immediate vs delayed 
    source = [(imm_thompson,  "thompson-immediate-{}".format(t)),
              (true_thompson, "thompson-true-{}".format(t))]
    plot_graph_average(source, dest = "contextual-switch-thompson-{}.png".format(t),
                max_step = max_step_to_plot, title = graph_title, vertical_line = t)

    # generate graph comparing LinUCB immediate vs delayed 
    source = [(imm_linucb,  "linucb-immediate-{}".format(t)),
              (true_linucb, "linucb-true-{}".format(t))]
    plot_graph_average(source, dest = "contextual-switch-linucb-{}.png".format(t),
                max_step = max_step_to_plot, title = graph_title, vertical_line = t)

