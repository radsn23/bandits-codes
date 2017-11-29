import sys
import csv
import random
import math
import numpy as np
from forced_actions import forced_actions
from bandit_data_format import *
from output_format import *
from logistic_regression import *


def create_headers(field_names, num_actions, outf):
    # Construct output column header names
    field_names_out = field_names[:]
    field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                            H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                            H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])
        
    # print group-level headers for readability
    group_header = ['' for i in range(len(field_names_out))]
    group_header[0] = "Input Data"
    group_header[len(field_names)] = "Algorithm's Performance"

    print(','.join(group_header), file=outf)

    return field_names_out


def compute_linucb_bound(model, x, alpha):
    m, q_inv = model.get_mean_variance()
    return np.dot(m, x) + alpha * np.sqrt(np.dot(q_inv, x ** 2))


def calculate_linucb_single_bandit(source, num_actions, dest, models = None, forced = forced_actions()):
    '''
    Calculates LinUCB.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    '''
    # number of trials used to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    if models == None:
        models = [RLogReg(D = NUM_FEATURES, Lambda = 1) for _ in range(num_actions)]

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names_out = create_headers(reader.fieldnames, num_actions, outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []
        
        alpha = 2

        for row in reader:
            sample_number += 1

            # get context features
            context = get_context(row)

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                # take action which maximizes the LinUCB bound based on current
                # model parameters (i.e. mean and variance of weight values)
                action = np.argmax([compute_linucb_bound(models[a], context, alpha) \
                    for a in range(num_actions)])
            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

            
            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            # update posterior distribution with observed reward
            # converted to range {-1,1}
            models[action].update_posterior(context, 2 * reward - 1)

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action = int(row[HEADER_OPTIMALACTION]) - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            out_row[H_ALGO_ACTION] = action + 1
            out_row[H_ALGO_OBSERVED_REWARD] = reward
            out_row[H_ALGO_MATCH_OPTIMAL] = 1 if optimal_action == action else 0
            out_row[H_ALGO_SAMPLE_REGRET] = sample_regret
            out_row[H_ALGO_SAMPLE_REGRET_CUMULATIVE] = cumulative_sample_regret

            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]

            # The oracle always chooses the best arm, thus expected reward
            # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            
            # TODO: compute expected regret for LinUCB
            expected_regret = 0
            cumulative_expected_regret += expected_regret

            out_row[H_ALGO_REGRET_EXPECTED] = expected_regret
            out_row[H_ALGO_REGRET_EXPECTED_CUMULATIVE] = cumulative_expected_regret

            writer.writerow(out_row)
        
        return chosen_actions, models


def switch_bandit_linucb(immediate_input, true_input, immediate_output,
                         true_output, time_step,
                         num_actions = 3, Lambda = 1):
    '''
    Run the algorithm on immediate-reward input up to specified time step then switch to the true-reward input and
    recompute policy by keeping the previously taken actions and matching with true rewards instead.
    :param immediate_input: The immediate-reward input file.
    :param true_input: The true-reward input file.
    :param immediate_output: The result output file from applying the algorithm to the immediate input.
    :param true_output: The result output file from applying the algorithm to the true input.
    :param time_step: The time step to switch bandit.
    :param num_actions: The number of actions in this bandit.
    :param Lambda: The prior inverse variance of the regression weights if regression is used.
    '''

    models = [RLogReg(D = NUM_FEATURES, Lambda = Lambda) for _ in range(num_actions)]

    # Run for 20 time steps on the immediate reward input
    chosen_actions, models = calculate_linucb_single_bandit(
        immediate_input, num_actions,
        immediate_output, models, forced_actions(time_step))

    # reset model state so that the algorithm forgets what happens
    for a in range(num_actions):
        models[a].reset_state()

    # Switch to true reward input, forcing actions taken previously
    calculate_linucb_single_bandit(
        true_input, num_actions, true_output,
        models, forced_actions(actions = chosen_actions))


def main():
    calculate_linucb_single_bandit('contextual_single_bandit.csv', 3, 'contextual_single_bandit_linucb.csv')

if __name__ == "__main__":
    main()
