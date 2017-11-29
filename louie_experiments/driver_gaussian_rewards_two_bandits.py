from gaussian_reward import *
from generate_single_bandit import *
from thompson_policy import *
from epsilon_greedy import *
from random_policy import *
from data_reader import *
from ucb1 import *
from plot_graph import *
from cycler import cycler
import sys
import argparse
import queueBasedBandit
import two_factor_reward

np.random.seed(0)

"""
This class is the main driver file for running switch bandit simulations.

Usage:
python3 driver_gaussian_rewards_two_bandits.py [total time steps] [time steps for when to switch] [number of runs] [whether to use existing output files or resample]
[total time steps]: how many total time steps to run each bandit
[time steps for when to switch]: at what point to switch from reward function 1 to reward function 2
[number of runs]: how many different multivariate gaussian rewards to sample (default is running one simulation per sampled reward)
[whether to use existing output files or resample]: 0 if we should resample all the bandits, 1 if we just want to aggregate the output files
   (this is mainly useful to deal with wanting to re-create plots)
   
TODO:
- It would be really nice if it were more modular in terms of:
-- Running only a subset of the algorithms
-- Adding an additional algorithm. Right now, extensive additions are required in the second
   half of the file.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--totalSteps", type=int, default=240, help="total number of actions for the bandits to take")
parser.add_argument("-s", "--switch", type=int, default=60,
                    help="number of actions to take before switching to second bandit")
parser.add_argument("-n", "--numSwitchBandits", type=int, default=1, help="number of switch bandit problems")
parser.add_argument("-e", "--useExistingOutput", type=bool, default=False,
                    help="if true, use already generated input and action output files")
parser.add_argument("-w", "--writeAllData", type=bool, default=True,
                    help="if true, write all runs to the table file; otherwise, write only averages")
parser.add_argument("--twoFactor", type=bool, default=False,
                    help="if true, uses two-factor latent structure reward; otherwise, uses MVN")

args = parser.parse_args()

# input data files to create
immediate_input_format = 'imm/input/gauss_single_bandit_input_immediate_{}.csv'
true_input_format = 'true/input/gauss_single_bandit_input_true_{}.csv'

# create output covariance matrix
mvn_output = 'gauss_mvn_parameters.csv'

# output directory for graphs
graph_directory = "graphs/"

# output files for different algorithms
imm_output_format_random = 'imm/random/gauss_output_random_immediate_{}.csv'
imm_output_format_thompson = 'imm/thompson/gauss_output_thompson_immediate_{}.csv'
imm_output_format_rel_thompson = 'imm/rel_thompson/gauss_output_rel_thompson_immediate_{}.csv'
imm_output_format_rand_thompson = 'imm/rand_thompson/gauss_output_rand_thompson_immediate_{}.csv'
imm_output_format_obl_rand_thompson = 'imm/obl_rand_thompson/gauss_output_obl_rand_thompson_immediate_{}.csv'
imm_output_format_epsilon = 'imm/epsilon/gauss_output_epsilon_immediate_{}.csv'
imm_output_format_ucb1 = 'imm/ucb1/gauss_output_ucb1_immediate_{}.csv'
imm_output_format_ucb1_historical = 'imm/hist_ucb1/gauss_output_historical_ucb1_immediate_{}.csv'
imm_output_format_ucb1_var_historical = 'imm/var_hist_ucb1/gauss_output_var_historical_ucb1_immediate_{}.csv'
imm_output_format_rand_ucb1 = 'imm/rand_ucb1/gauss_output_rand_ucb1_immediate_{}.csv'
imm_output_format_obl_rand_ucb1 = 'imm/obl_rand_ucb1/gauss_output_obl_rand_ucb1_immediate_{}.csv'
imm_output_format_queuing_thompson = 'imm/queue_thompson/gauss_output_queuing_thompson_immediate_{}.csv'

true_output_format_random = 'true/random/gauss_output_random_true_{}.csv'
true_output_format_thompson = 'true/thompson/gauss_output_thompson_true_{}.csv'
true_output_format_rel_thompson = 'true/rel_thompson/gauss_output_rel_thompson_true_{}.csv'
true_output_format_rand_thompson = 'true/rand_thompson/gauss_output_rand_thompson_true_{}.csv'
true_output_format_obl_rand_thompson = 'true/obl_rand_thompson/gauss_output_obl_rand_thompson_true_{}.csv'
true_output_format_epsilon = 'true/epsilon/gauss_output_epsilon_true_{}.csv'
true_output_format_ucb1 = 'true/ucb1/gauss_output_ucb1_true_{}.csv'
true_output_format_ucb1_historical = 'true/hist_ucb1/gauss_output_historical_ucb1_true_{}.csv'
true_output_format_ucb1_var_historical = 'true/var_hist_ucb1/gauss_output_var_historical_ucb1_true_{}.csv'
true_output_format_rand_ucb1 = 'true/rand_ucb1/gauss_output_rand_ucb1_true_{}.csv'
true_output_format_obl_rand_ucb1 = 'true/obl_rand_ucb1/gauss_output_obl_rand_ucb1_true_{}.csv'
true_output_format_queuing_thompson = 'true/queue_thompson/gauss_output_queuing_thompson_true_{}.csv'

table_file = 'gauss_all_table_t{}.csv'

num_rows = args.totalSteps
time_steps = [args.switch]

num_actions = 3
num_bandits = 2
num_samples = args.numSwitchBandits  # number of samples from MVN, also number of switch-bandit problems
num_runs = 1  # number of runs over each switch-bandit problem


def calculate_bandits(probs_matrix, cc):
    for i in range(num_samples):

        file_flavor = "cor{}sample{}".format(cc, i + 1)

        probs = probs_matrix[i, :]

        # TODO: normalize probs properly
        probs[probs < 0] = 0
        probs[probs > 1] = 1
        # probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))

        immediate_probs = probs[:num_actions]
        true_probs = probs[num_actions:]

        immediate_input = immediate_input_format.format(file_flavor)
        true_input = true_input_format.format(file_flavor)

        # generate data and write to file
        generate_file(immediate_probs, num_rows, immediate_input)
        generate_file(true_probs, num_rows, true_input)

        for t in time_steps:
            for r in range(num_runs):
                print('cor = {} sample = {} switch = {} run = {}'.format(cc, i + 1, t, r + 1))

                t_flavor = "t{}{}run{}".format(t, file_flavor, r + 1)

                # run random policy switching bandits between
                # immediate and true reward inputs
                switch_bandit_random(immediate_input, true_input,
                                     imm_output_format_random.format(t_flavor),
                                     true_output_format_random.format(t_flavor),
                                     time_step=t, num_actions=len(true_probs))

                # run prob-best thompson sampling switching bandits
                # between immediate and true reward inputs
                switch_bandit_thompson(immediate_input, true_input,
                                       imm_output_format_thompson.format(t_flavor),
                                       true_output_format_thompson.format(t_flavor),
                                       time_step=t, action_mode=ActionSelectionMode.prob_is_best,
                                       num_actions=len(true_probs))

                # run relative thompson sampling
                switch_bandit_thompson(immediate_input, true_input,
                                       imm_output_format_rel_thompson.format(t_flavor),
                                       true_output_format_rel_thompson.format(t_flavor),
                                       time_step=t, action_mode=ActionSelectionMode.expected_value,
                                       relearn=True, num_actions=len(true_probs))

                # run random first then switch to regular thompson and relearns from old data
                switch_bandit_random_thompson(immediate_input, true_input,
                                              imm_output_format_rand_thompson.format(t_flavor),
                                              true_output_format_rand_thompson.format(t_flavor),
                                              time_step=t, action_mode=ActionSelectionMode.prob_is_best,
                                              relearn=True, num_actions=len(true_probs))

                # run random first then switch to regular thompson, oblivious to old data
                switch_bandit_thompson(immediate_input, true_input,
                                       imm_output_format_obl_rand_thompson.format(t_flavor),
                                       true_output_format_obl_rand_thompson.format(t_flavor),
                                       time_step=t, action_mode=ActionSelectionMode.prob_is_best,
                                       relearn=False, num_actions=len(true_probs))

                # run epsilon greedy
                switch_bandit_epsilon(immediate_input, true_input,
                                      imm_output_format_epsilon.format(t_flavor),
                                      true_output_format_epsilon.format(t_flavor),
                                      time_step=t, num_actions=len(true_probs), epsilon=0.2)

                # run UCB1 switching bandits between immediate and true reward inputs
                switch_bandit_ucb1(immediate_input, true_input,
                                   imm_output_format_ucb1.format(t_flavor),
                                   true_output_format_ucb1.format(t_flavor),
                                   time_step=t,
                                   num_actions=len(true_probs))

                # run UCB1 switching bandits between immediate and true reward inputs
                # treat historical arm pulls differently for the purposes of calculating UCB
                switch_bandit_ucb1(immediate_input, true_input,
                                   imm_output_format_ucb1_historical.format(t_flavor),
                                   true_output_format_ucb1_historical.format(t_flavor),
                                   time_step=t,
                                   num_actions=len(true_probs),
                                   treat_forced_as_historical=True)

                # run UCB1 switching bandits between immediate and true reward inputs
                # treat historical arm pulls differently for the purposes of calculating UCB and
                # use the approach that takes into account the variance of the arm
                switch_bandit_ucb1(immediate_input, true_input,
                                   imm_output_format_ucb1_var_historical.format(t_flavor),
                                   true_output_format_ucb1_var_historical.format(t_flavor),
                                   time_step=t,
                                   num_actions=len(true_probs),
                                   treat_forced_as_historical=True,
                                   use_sample_variance=True)

                # run random first then switch to ucb1 and relearns from old data
                switch_bandit_random_ucb1(immediate_input, true_input,
                                          imm_output_format_rand_ucb1.format(t_flavor),
                                          true_output_format_rand_ucb1.format(t_flavor),
                                          time_step=t,
                                          num_actions=len(true_probs),
                                          relearn=True)

                # run random first then switch to ucb1, oblivious to old data
                switch_bandit_ucb1(immediate_input, true_input,
                                   imm_output_format_obl_rand_ucb1.format(t_flavor),
                                   true_output_format_obl_rand_ucb1.format(t_flavor),
                                   time_step=t,
                                   num_actions=len(true_probs),
                                   relearn=False)

                # Run queuing Thompson
                queueBasedBandit.switch_bandit_queue(immediate_input, true_input,
                                                     imm_output_format_queuing_thompson.format(t_flavor),
                                                     true_output_format_queuing_thompson.format(t_flavor),
                                                     time_step_switch=t,
                                                     total_time_steps=num_rows,
                                                     num_actions=len(true_probs))


""" 
Generate rewards for 2 bandits from a multivariate gaussian
See generate_gaussian_rewards method for more explanations
"""

# Different correlation values
c_list = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

"""This can also generate the rewards for the latent factor structure - see 
command line arguments and two_factor_reward for details.
"""
# Different variance values for clarity
variance_list = [.05, .1, .15, .2]  # [.01, .05, .1]
lower_bound = .3
upper_bound = .7

if not args.useExistingOutput:
    if args.twoFactor:
        for variance in variance_list:
            probs_matrix = two_factor_reward.generate_two_factor_rewards(num_actions, num_samples, lower_bound,
                                                                         upper_bound, variance)
            calculate_bandits(probs_matrix, variance)
    else:
        for cc in c_list:
            # Covariance matrix
            c = \
                {
                    # cov for arm 1 between bandit 1,2
                    1: [[1, 2, cc]],
                    # cov for arm 2 between bandit 1,2
                    2: [[1, 2, cc]],
                    # cov for arm 3 between bandit 1,2
                    3: [[1, 2, cc]],
                }

            # Variance for each arm in each bandit
            v = np.ones(num_bandits * num_actions) * 1
            probs_matrix = generate_gaussian_rewards(num_bandits, num_actions, 0.5, c, v, num_samples, mvn_output)
            calculate_bandits(probs_matrix, cc)



# Generate average graph across all samples
# Average result across different samples within each time step and each correlation value
for t in time_steps:
    max_step_to_plot = num_rows

    csv_avg_data = []
    csv_all_data = []
    correlation_list = c_list
    if args.twoFactor:
        correlation_list = variance_list
    for cc in correlation_list:

        imm_random = []
        imm_thompson = []
        imm_rel_thompson = []
        imm_rand_thompson = []
        imm_obl_rand_thompson = []
        imm_epsilon = []
        imm_ucb1 = []
        imm_historical_ucb1 = []
        imm_var_historical_ucb1 = []
        imm_rand_ucb1 = []
        imm_obl_rand_ucb1 = []
        imm_queuing_thompson = []

        true_random = []
        true_thompson = []
        true_rel_thompson = []
        true_rand_thompson = []
        true_obl_rand_thompson = []
        true_epsilon = []
        true_ucb1 = []
        true_historical_ucb1 = []
        true_var_historical_ucb1 = []
        true_rand_ucb1 = []
        true_obl_rand_ucb1 = []
        true_queuing_thompson = []

        # aggregate all output files from all algorithms on immediate and delayed inputs
        # in order to compute and plot average cumulative regret
        for i in range(num_samples):
            for r in range(num_runs):
                file_flavor = "t{}cor{}sample{}run{}".format(t, cc, i + 1, r + 1)
                imm_random.append(imm_output_format_random.format(file_flavor))
                imm_thompson.append(imm_output_format_thompson.format(file_flavor))
                imm_rel_thompson.append(imm_output_format_rel_thompson.format(file_flavor))
                imm_rand_thompson.append(imm_output_format_rand_thompson.format(file_flavor))
                imm_obl_rand_thompson.append(imm_output_format_obl_rand_thompson.format(file_flavor))
                imm_epsilon.append(imm_output_format_epsilon.format(file_flavor))
                imm_ucb1.append(imm_output_format_ucb1.format(file_flavor))
                imm_historical_ucb1.append(imm_output_format_ucb1_historical.format(file_flavor))
                imm_var_historical_ucb1.append(imm_output_format_ucb1_var_historical.format(file_flavor))
                imm_rand_ucb1.append(imm_output_format_rand_ucb1.format(file_flavor))
                imm_obl_rand_ucb1.append(imm_output_format_obl_rand_ucb1.format(file_flavor))
                imm_queuing_thompson.append(imm_output_format_queuing_thompson.format(file_flavor))

                true_random.append(true_output_format_random.format(file_flavor))
                true_thompson.append(true_output_format_thompson.format(file_flavor))
                true_rel_thompson.append(true_output_format_rel_thompson.format(file_flavor))
                true_rand_thompson.append(true_output_format_rand_thompson.format(file_flavor))
                true_obl_rand_thompson.append(true_output_format_obl_rand_thompson.format(file_flavor))
                true_epsilon.append(true_output_format_epsilon.format(file_flavor))
                true_ucb1.append(true_output_format_ucb1.format(file_flavor))
                true_historical_ucb1.append(true_output_format_ucb1_historical.format(file_flavor))
                true_var_historical_ucb1.append(true_output_format_ucb1_var_historical.format(file_flavor))
                true_rand_ucb1.append(true_output_format_rand_ucb1.format(file_flavor))
                true_obl_rand_ucb1.append(true_output_format_obl_rand_ucb1.format(file_flavor))
                true_queuing_thompson.append(true_output_format_queuing_thompson.format(file_flavor))

        ''' 
        Create table result comparing average regret over all 
        trials for each algorithm and for each correlation value
        '''
        avg_reg_imm_ran, stderr_reg_imm_ran, all_reg_imm_ran = read_avg_regret_multiple(imm_random, t)
        avg_reg_imm_eps, stderr_reg_imm_eps, all_reg_imm_eps = read_avg_regret_multiple(imm_epsilon, t)
        avg_reg_imm_ucb, stderr_reg_imm_ucb, all_reg_imm_ucb = read_avg_regret_multiple(imm_ucb1, t)
        avg_reg_imm_hucb, stderr_reg_imm_hucb, all_reg_imm_hucb = read_avg_regret_multiple(imm_historical_ucb1, t)
        avg_reg_imm_vhucb, stderr_reg_imm_vhucb, all_reg_imm_vhucb = read_avg_regret_multiple(imm_var_historical_ucb1,
                                                                                              t)
        avg_reg_imm_ruc, stderr_reg_imm_ruc, all_reg_imm_ruc = read_avg_regret_multiple(imm_rand_ucb1, t)
        avg_reg_imm_oru, stderr_reg_imm_oru, all_reg_imm_oru = read_avg_regret_multiple(imm_obl_rand_ucb1, t)
        avg_reg_imm_tho, stderr_reg_imm_tho, all_reg_imm_tho = read_avg_regret_multiple(imm_thompson, t)
        avg_reg_imm_rth, stderr_reg_imm_rth, all_reg_imm_rth = read_avg_regret_multiple(imm_rel_thompson, t)
        avg_reg_imm_rat, stderr_reg_imm_rat, all_reg_imm_rat = read_avg_regret_multiple(imm_rand_thompson, t)
        avg_reg_imm_ort, stderr_reg_imm_ort, all_reg_imm_ort = read_avg_regret_multiple(imm_obl_rand_thompson, t)
        avg_reg_imm_qt, stderr_reg_imm_qt, all_reg_imm_qt = read_avg_regret_multiple(imm_queuing_thompson, t)

        avg_reg_true_ran, stderr_reg_true_ran, all_reg_true_ran = read_avg_regret_multiple(true_random, t)
        avg_reg_true_eps, stderr_reg_true_eps, all_reg_true_eps = read_avg_regret_multiple(true_epsilon, t)
        avg_reg_true_ucb, stderr_reg_true_ucb, all_reg_true_ucb = read_avg_regret_multiple(true_ucb1, t)
        avg_reg_true_hucb, stderr_reg_true_hucb, all_reg_true_hucb = read_avg_regret_multiple(true_historical_ucb1, t)
        avg_reg_true_vhucb, stderr_reg_true_vhucb, all_reg_true_vhucb = read_avg_regret_multiple(
            true_var_historical_ucb1, t)
        avg_reg_true_ruc, stderr_reg_true_ruc, all_reg_true_ruc = read_avg_regret_multiple(true_rand_ucb1, t)
        avg_reg_true_oru, stderr_reg_true_oru, all_reg_true_oru = read_avg_regret_multiple(true_obl_rand_ucb1, t)
        avg_reg_true_tho, stderr_reg_true_tho, all_reg_true_tho = read_avg_regret_multiple(true_thompson, t)
        avg_reg_true_rth, stderr_reg_true_rth, all_reg_true_rth = read_avg_regret_multiple(true_rel_thompson, t)
        avg_reg_true_rat, stderr_reg_true_rat, all_reg_true_rat = read_avg_regret_multiple(true_rand_thompson, t)
        avg_reg_true_ort, stderr_reg_true_ort, all_reg_true_ort = read_avg_regret_multiple(true_obl_rand_thompson, t)
        avg_reg_true_qt, stderr_reg_true_qt, all_reg_true_qt = read_avg_regret_multiple(true_queuing_thompson, t)

        # add average regret data into csv format
        csv_row = [cc] + \
                  avg_reg_imm_ran.tolist() + stderr_reg_imm_ran.tolist() + \
                  avg_reg_true_ran.tolist() + stderr_reg_true_ran.tolist() + \
                  avg_reg_imm_eps.tolist() + stderr_reg_imm_eps.tolist() + \
                  avg_reg_true_eps.tolist() + stderr_reg_true_eps.tolist() + \
                  avg_reg_imm_ucb.tolist() + stderr_reg_imm_ucb.tolist() + \
                  avg_reg_true_ucb.tolist() + stderr_reg_true_ucb.tolist() + \
                  avg_reg_imm_hucb.tolist() + stderr_reg_imm_hucb.tolist() + \
                  avg_reg_true_hucb.tolist() + stderr_reg_true_hucb.tolist() + \
                  avg_reg_imm_vhucb.tolist() + stderr_reg_imm_vhucb.tolist() + \
                  avg_reg_true_vhucb.tolist() + stderr_reg_true_vhucb.tolist() + \
                  avg_reg_imm_ruc.tolist() + stderr_reg_imm_ruc.tolist() + \
                  avg_reg_true_ruc.tolist() + stderr_reg_true_ruc.tolist() + \
                  avg_reg_imm_oru.tolist() + stderr_reg_imm_oru.tolist() + \
                  avg_reg_true_oru.tolist() + stderr_reg_true_oru.tolist() + \
                  avg_reg_imm_tho.tolist() + stderr_reg_imm_tho.tolist() + \
                  avg_reg_true_tho.tolist() + stderr_reg_true_tho.tolist() + \
                  avg_reg_imm_rth.tolist() + stderr_reg_imm_rth.tolist() + \
                  avg_reg_true_rth.tolist() + stderr_reg_true_rth.tolist() + \
                  avg_reg_imm_rat.tolist() + stderr_reg_imm_rat.tolist() + \
                  avg_reg_true_rat.tolist() + stderr_reg_true_rat.tolist() + \
                  avg_reg_imm_ort.tolist() + stderr_reg_imm_ort.tolist() + \
                  avg_reg_true_ort.tolist() + stderr_reg_true_ort.tolist() + \
                  avg_reg_imm_qt.tolist() + stderr_reg_imm_qt.tolist() + \
                  avg_reg_true_qt.tolist() + stderr_reg_true_qt.tolist()
        csv_avg_data.append(csv_row)

        # add all regret data into csv format

        # immediate data
        for imm_ran_tuple in all_reg_imm_ran:
            csv_all_data.append([imm_ran_tuple[0]] + imm_ran_tuple[1])
        for imm_eps_tuple in all_reg_imm_eps:
            csv_all_data.append([imm_eps_tuple[0]] + imm_eps_tuple[1])
        for imm_ucb_tuple in all_reg_imm_ucb:
            csv_all_data.append([imm_ucb_tuple[0]] + imm_ucb_tuple[1])
        for imm_hucb_tuple in all_reg_imm_hucb:
            csv_all_data.append([imm_hucb_tuple[0]] + imm_hucb_tuple[1])
        for imm_vhucb_tuple in all_reg_imm_vhucb:
            csv_all_data.append([imm_vhucb_tuple[0]] + imm_vhucb_tuple[1])
        for imm_ruc_tuple in all_reg_imm_ruc:
            csv_all_data.append([imm_ruc_tuple[0]] + imm_ruc_tuple[1])
            # ANR: 11 July 2016 - deleted below lines as exact duplicate of lines above
        #         for imm_ruc_tuple in all_reg_imm_ruc:
        #             csv_all_data.append([imm_ruc_tuple[0]] + imm_ruc_tuple[1])
        for imm_oru_tuple in all_reg_imm_oru:
            csv_all_data.append([imm_oru_tuple[0]] + imm_oru_tuple[1])
        for imm_tho_tuple in all_reg_imm_tho:
            csv_all_data.append([imm_tho_tuple[0]] + imm_tho_tuple[1])
        for imm_rth_tuple in all_reg_imm_rth:
            csv_all_data.append([imm_rth_tuple[0]] + imm_rth_tuple[1])
        for imm_rat_tuple in all_reg_imm_rat:
            csv_all_data.append([imm_rat_tuple[0]] + imm_rat_tuple[1])
        for imm_ort_tuple in all_reg_imm_ort:
            csv_all_data.append([imm_ort_tuple[0]] + imm_ort_tuple[1])
        for imm_qt_tuple in all_reg_imm_qt:
            csv_all_data.append([imm_qt_tuple[0]] + imm_qt_tuple[1])

        # delayed data
        for true_ran_tuple in all_reg_true_ran:
            csv_all_data.append([true_ran_tuple[0]] + true_ran_tuple[1])
        for true_eps_tuple in all_reg_true_eps:
            csv_all_data.append([true_eps_tuple[0]] + true_eps_tuple[1])
        for true_ucb_tuple in all_reg_true_ucb:
            csv_all_data.append([true_ucb_tuple[0]] + true_ucb_tuple[1])
        for true_hucb_tuple in all_reg_true_hucb:
            csv_all_data.append([true_hucb_tuple[0]] + true_hucb_tuple[1])
        for true_vhucb_tuple in all_reg_true_vhucb:
            csv_all_data.append([true_vhucb_tuple[0]] + true_vhucb_tuple[1])
        for true_ruc_tuple in all_reg_true_ruc:
            csv_all_data.append([true_ruc_tuple[0]] + true_ruc_tuple[1])
            # ANR: 11 July 2016 - deleted below lines as exact duplicate of lines above
        #         for true_ruc_tuple in all_reg_true_ruc:
        #             csv_all_data.append([true_ruc_tuple[0]] + true_ruc_tuple[1])
        for true_oru_tuple in all_reg_true_oru:
            csv_all_data.append([true_oru_tuple[0]] + true_oru_tuple[1])
        for true_tho_tuple in all_reg_true_tho:
            csv_all_data.append([true_tho_tuple[0]] + true_tho_tuple[1])
        for true_rth_tuple in all_reg_true_rth:
            csv_all_data.append([true_rth_tuple[0]] + true_rth_tuple[1])
        for true_rat_tuple in all_reg_true_rat:
            csv_all_data.append([true_rat_tuple[0]] + true_rat_tuple[1])
        for true_ort_tuple in all_reg_true_ort:
            csv_all_data.append([true_ort_tuple[0]] + true_ort_tuple[1])

        for true_qt_tuple in all_reg_true_qt:
            csv_all_data.append([true_qt_tuple[0]] + true_qt_tuple[1])

        ''' Plotting average result curve for different algorithms over all trials'''
        prob_flavor = "cor({})".format(cc)
        graph_title = 'Average Cumulative Regret from over MVN reward probability samples {}'.format(prob_flavor)
        # pyplot.rc('axes', prop_cycle=(cycler('color', ['red', 'green', 'blue', 'gray', 'teal', 'violet', 'black', 'orange', 'maroon' ])))


        t_flavor = "t{}cor{}".format(t, cc)
        # generate graph ONLY on immediate data
        source = [(imm_random, "random-immediate-{}".format(t)),
                  (imm_ucb1, "ucb1-immediate-{}".format(t)),
                  (imm_historical_ucb1, "hist-ucb1-immediate-{}".format(t)),
                  (imm_var_historical_ucb1, "var-hist-ucb1-immediate-{}".format(t)),
                  (imm_rand_ucb1, "rand-ucb1-immediate-{}".format(t)),
                  (imm_obl_rand_ucb1, "obl-rand-ucb1-immediate-{}".format(t)),
                  (imm_thompson, "thompson-immediate-{}".format(t)),
                  (imm_rel_thompson, "rel-thompson-immediate-{}".format(t)),
                  (imm_rand_thompson, "rand-thompson-immediate-{}".format(t)),
                  (imm_obl_rand_thompson, "obl-rand-thompson-immediate-{}".format(t)),
                  (imm_epsilon, "epsilon-immediate-{}".format(t)),
                  (imm_queuing_thompson, "queue-thompson-immediate-{}".format(t))]
        plot_graph_average(source, dest=graph_directory + "average-immediate-{}.png".format(t_flavor),
                           max_step=max_step_to_plot, title=graph_title,
                           vertical_line=t, raw_data_dest=graph_directory + "average-immediate-{}.csv".format(t_flavor))

        # generate graph ONLY on delayed data 
        source = [(true_random, "random-true-{}".format(t)),
                  (true_ucb1, "ucb1-true-{}".format(t)),
                  (true_historical_ucb1, "hist-ucb1-true-{}".format(t)),
                  (true_var_historical_ucb1, "var-hist-ucb1-true-{}".format(t)),
                  (true_rand_ucb1, "rand-ucb1-true-{}".format(t)),
                  (true_obl_rand_ucb1, "obl-rand-ucb1-true-{}".format(t)),
                  (true_thompson, "thompson-true-{}".format(t)),
                  (true_rel_thompson, "rel-thompson-true-{}".format(t)),
                  (true_rand_thompson, "rand-thompson-true-{}".format(t)),
                  (true_obl_rand_thompson, "obl-rand-thompson-true-{}".format(t)),
                  (true_epsilon, "epsilon-true-{}".format(t)),
                  (true_queuing_thompson, "queue-thompson-true-{}".format(t))]
        plot_graph_average(source, dest=graph_directory + "average-true-{}.png".format(t_flavor),
                           max_step=max_step_to_plot, title=graph_title,
                           vertical_line=t, raw_data_dest=graph_directory + "average-true-{}.csv".format(t_flavor))

        # generate graph comparing thompson immediate vs delayed 
        source = [(imm_thompson, "thompson-immediate-{}".format(t)),
                  (imm_rel_thompson, "rel-thompson-immediate-{}".format(t)),
                  (imm_rand_thompson, "rand-thompson-immediate-{}".format(t)),
                  (imm_obl_rand_thompson, "obl-rand-thompson-immediate-{}".format(t)),
                  (imm_queuing_thompson, "queue-thompson-immediate-{}".format(t)),
                  (true_thompson, "thompson-true-{}".format(t)),
                  (true_rel_thompson, "rel-thompson-true-{}".format(t)),
                  (true_rand_thompson, "rand-thompson-true-{}".format(t)),
                  (true_obl_rand_thompson, "obl-rand-thompson-true-{}".format(t)),
                  (true_queuing_thompson, "queue-thompson-true-{}".format(t))]
        plot_graph_average(source, dest=graph_directory + "average-thompson-{}.png".format(t_flavor),
                           max_step=max_step_to_plot, title=graph_title,
                           vertical_line=t, raw_data_dest=graph_directory + "average-thompson-{}.csv".format(t_flavor))

        # generate graph comparing UCB1 immediate vs delayed 
        source = [
            (imm_ucb1, "ucb1-immediate-{}".format(t)),
            (imm_historical_ucb1, "hist-ucb1-immediate-{}".format(t)),
            (imm_var_historical_ucb1, "var-hist-ucb1-immediate-{}".format(t)),
            (imm_rand_ucb1, "rand-ucb1-immediate-{}".format(t)),
            (imm_obl_rand_ucb1, "obl-rand-ucb1-immediate-{}".format(t)),
            (true_ucb1, "ucb1-true-{}".format(t)),
            (true_historical_ucb1, "hist-ucb1-true-{}".format(t)),
            (true_var_historical_ucb1, "var-hist-ucb1-true-{}".format(t)),
            (true_rand_ucb1, "rand-ucb1-true-{}".format(t)),
            (true_obl_rand_ucb1, "obl-rand-ucb1-true-{}".format(t))
        ]
        plot_graph_average(source, dest=graph_directory + "ucb1-{}.png".format(t_flavor),
                           max_step=max_step_to_plot, title=graph_title,
                           vertical_line=t, raw_data_dest=graph_directory + "ucb1-{}.csv".format(t_flavor))

        # generate graph comparing thompson and random immediate vs delayed 
        source = [(imm_random, "random-immediate-{}".format(t)),
                  (true_random, "random-true-{}".format(t)),
                  (imm_ucb1, "ucb1-immediate-{}".format(t)),
                  (imm_historical_ucb1, "hist-ucb1-immediate-{}".format(t)),
                  (imm_var_historical_ucb1, "var-hist-ucb1-immediate-{}".format(t)),
                  (imm_rand_ucb1, "rand-ucb1-immediate-{}".format(t)),
                  (imm_obl_rand_ucb1, "obl-rand-ucb1-immediate-{}".format(t)),
                  (true_historical_ucb1, "hist-ucb1-true-{}".format(t)),
                  (true_var_historical_ucb1, "var-hist-ucb1-true-{}".format(t)),
                  (true_ucb1, "ucb1-true-{}".format(t)),
                  (true_rand_ucb1, "rand-ucb1-true-{}".format(t)),
                  (true_obl_rand_ucb1, "obl-rand-ucb1-true-{}".format(t)),
                  (imm_thompson, "thompson-immediate-{}".format(t)),
                  (imm_rel_thompson, "rel-thompson-immediate-{}".format(t)),
                  (imm_rand_thompson, "rand-thompson-immediate-{}".format(t)),
                  (imm_obl_rand_thompson, "obl-rand-thompson-immediate-{}".format(t)),
                  (true_thompson, "thompson-true-{}".format(t)),
                  (true_rel_thompson, "rel-thompson-true-{}".format(t)),
                  (true_rand_thompson, "rand-thompson-true-{}".format(t)),
                  (true_obl_rand_thompson, "obl-rand-thompson-true-{}".format(t)),
                  (imm_epsilon, "epsilon-immediate-{}".format(t)),
                  (true_epsilon, "epsilon-true-{}".format(t))]
        plot_graph_average(source, dest=graph_directory + "average-all-{}.png".format(t_flavor),
                           max_step=max_step_to_plot, title=graph_title,
                           vertical_line=t)

    """Writing out average data to file"""
    num_algorithms = 12  # UPDATE THIS IF NEW ALGORITHMS ARE ADDED OR EXISTING ONES REMOVED !!!
    # write table data
    with open(table_file.format(t), 'w', newline='') as tfp:
        tfcsv = csv.writer(tfp, delimiter=',')
        commas = ['' for _ in range(11)]
        header1 = [''] + \
                  ['Random'] + commas + \
                  ['Epsilon Greedy'] + commas + \
                  ['UCB1'] + commas + \
                  ['Historical UCB1'] + commas + \
                  ['Variance Historical UCB1'] + commas + \
                  ['Random UCB1'] + commas + \
                  ['Oblivious Random UCB1'] + commas + \
                  ['Thompson'] + commas + \
                  ['Relative Thompson'] + commas + \
                  ['Random Thompson'] + commas + \
                  ['Oblivious Random Thompson'] + commas + \
                  ['Queuing Thompson']
        header2 = [[
            'Immediate-All', 'Immediate-Before', 'Immediate-After', \
            'Stderr-Immediate-All', 'Stderr-Immediate-Before', 'Stderr-Immediate-After', \
            'Delayed-All', 'Delayed-Before', 'Delayed-After', \
            'Stderr-Delayed-All', 'Stderr-Delayed-Before', 'Stderr-Delayed-After'] \
            for _ in range(num_algorithms)]
        header = []
        header.append(header1)
        header.append(['Correlation'] + [item for sublist in header2 for item in sublist])
        tfcsv.writerows(header)
        tfcsv.writerows(csv_avg_data)
        if args.writeAllData:
            tfcsv.writerows(csv_all_data)
