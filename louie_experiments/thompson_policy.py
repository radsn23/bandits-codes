import sys
import csv
import random
import math
import numpy as np
from forced_actions import forced_actions
from enum import Enum
from bandit_data_format import *
from logistic_regression import *
from beta_bernoulli import *
from nig_normal import *
from output_format import *
from random_policy import *
from generate_single_bandit import *


class ActionSelectionMode(Enum):
    # Select action by probability it is best
    prob_is_best = 0

    # Select action in proportion to expected rewards
    expected_value = 1


def create_headers(field_names, num_actions):
    # Construct output column header names
    field_names_out = field_names[:]
    field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                            H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                            H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])

    # not important, store the position to write high level header to output file
    group_header_parameters_index = len(field_names_out)

    for a in range(num_actions):
        # field_names_out.append(H_ALGO_ACTION_SUCCESS.format(a + 1))
        # field_names_out.append(H_ALGO_ACTION_FAILURE.format(a + 1))
        # field_names_out.append(H_ALGO_ESTIMATED_PROB.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_MU.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_V.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_ALPHA.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_BETA.format(a + 1))
        field_names_out.append(H_ALGO_ESTIMATED_ARM_VARIANCE.format(a + 1))

    field_names_out.extend([H_ALGO_PROB_BEST_ACTION.format(a + 1) for a in range(num_actions)])
    field_names_out.append(H_ALGO_NUM_TRIALS)
    field_names_out.extend([H_ALGO_ACTION_SAMPLE.format(a + 1) for a in range(num_actions)])
    field_names_out.append(H_ALGO_CHOSEN_ACTION)

    # print group-level headers for readability
    group_header = ['' for i in range(len(field_names_out))]
    group_header[0] = "Input Data"
    group_header[len(field_names)] = "Algorithm's Performance"
    group_header[group_header_parameters_index] = "Model Parameters"

    return field_names_out, group_header


def write_performance(out_row, action, optimal_action, reward, sample_regret, cumulative_sample_regret, expected_regret,
                      cumulative_expected_regret):
    ''' write performance data (e.g. regret) '''
    out_row[H_ALGO_ACTION] = action + 1
    out_row[H_ALGO_OBSERVED_REWARD] = reward
    out_row[H_ALGO_MATCH_OPTIMAL] = 1 if optimal_action == action else 0
    out_row[H_ALGO_SAMPLE_REGRET] = sample_regret
    out_row[H_ALGO_SAMPLE_REGRET_CUMULATIVE] = cumulative_sample_regret
    out_row[H_ALGO_REGRET_EXPECTED] = expected_regret
    out_row[H_ALGO_REGRET_EXPECTED_CUMULATIVE] = cumulative_expected_regret
    pass


def write_parameters(out_row, action, samples, models,
                     chosen_action_counts, num_actions,
                     num_trials_prob_best_action):
    ''' write parameters data (e.g. beta parameters)'''
    for a in range(num_actions):
        models[a].write_parameters(out_row, a)

    # probability that each action is the best action
    # TODO: call a function to compute this value
    # for a in range(num_actions):
    #     out_row[H_ALGO_PROB_BEST_ACTION.format(a + 1)] = \
    #         float(chosen_action_counts[a]) / np.sum(chosen_action_counts)

    # number of repeated trials of Thompson Sampling to determine the
    # probability that each action is the best action
    out_row[H_ALGO_NUM_TRIALS] = num_trials_prob_best_action

    # samples for each action
    for a in range(num_actions):
        out_row[H_ALGO_ACTION_SAMPLE.format(a + 1)] = samples[a]

    # chosen action at this time step
    out_row[H_ALGO_CHOSEN_ACTION] = action + 1


def run_thompson_trial(context, num_samples, num_actions, models):
    '''
    Run Thompson Sampling many times using the specified Beta parameters.
    This is useful to compute several values in expectation, e.g. probability
    that each action is the best action, or the expected reward.
    :param context: Context features.
    :param num_samples: Number of times to run Thompson Sampling for.
    :param num_actions: Number of actions.
    :param models: The current model states for each action.
    '''
    ######################################################################################
    # NOTE: Uncomment these to print out model parameters for Thompson Sampling
    # NOTE: However, this will make performance quite a bit slower
    ######################################################################################
    ## tile the success and failure counts to generate beta samples
    ## efficiently using vectorized operations
    # samples = [models[a].draw_expected_value(context, num_samples) for a in range(num_actions)]

    ## generate a matrix of size (num_actions x num_trials) containing sampled expected values
    # samples = np.array(samples)

    ## take argmax of each row to get the chosen action
    # chosen_actions = np.argmax(samples, 0)

    # chosen_action_counts = np.zeros(num_actions)

    ## count how many times each action was chosen
    ## result is an array of size [num_actions] where value at
    ## i-th index is the number of times action index i was chosen.
    # bin_counts = np.bincount(chosen_actions)

    # chosen_action_counts[:len(bin_counts)] = bin_counts

    # return chosen_action_counts
    ######################################################################################
    return np.zeros(num_actions)


def calculate_thompson_single_bandit(source, num_actions, dest, models=None,
                                     action_mode=ActionSelectionMode.prob_is_best, forced=forced_actions(),
                                     relearn=True):
    '''
    Calculates non-contextual thompson sampling actions and weights.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param models: models for each action's probability distribution.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    if models == None:
        models = [BetaBern(success=1, failure=1) for cond in range(num_actions)]

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, num_actions)

        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []

        for row in reader:
            sample_number += 1

            # get context features
            context = get_context(row)

            should_update_posterior = True

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                # first decide which arm we'd pull using Thompson
                # (do the random sampling, the max is the one we'd choose)
                samples = [models[a].draw_expected_value(context) for a in range(num_actions)]

                if action_mode == ActionSelectionMode.prob_is_best:
                    # find the max of samples[i] etc and choose an arm
                    action = np.argmax(samples)
                else:
                    # take action in proportion to expected rewards
                    # draw samples and normalize to use as a discrete distribution
                    # action is taken by sampling from this discrete distribution
                    probs = samples / np.sum(samples)
                    rand = np.random.rand()
                    for a in range(num_actions):
                        if rand <= probs[a]:
                            action = a
                            break
                        rand -= probs[a]

            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

                if relearn == False:
                    should_update_posterior = False

            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            if should_update_posterior:
                # update posterior distribution with observed reward
                # converted to range {-1,1}
                models[action].update_posterior(context, 2 * reward - 1)

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)
                # save the model state in order so we can restore it
                # after switching to the true reward data.
                if sample_number == forced.time_step:
                    for a in range(num_actions):
                        models[a].save_state()

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action = int(row[HEADER_OPTIMALACTION]) - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            true_probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]

            # # The oracle always chooses the best arm, thus expected reward
            # # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action
            #
            # # Run thompson sampling many times and calculate how much reward it would
            # # have gotten based on the chosen actions.
            chosen_action_counts = run_thompson_trial(context, num_trials_prob_best_action, num_actions, models)
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(num_actions))

            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret

            write_performance(out_row, action, optimal_action, reward,
                              sample_regret, cumulative_sample_regret,
                              expected_regret, cumulative_expected_regret)

            write_parameters(out_row, action, samples, models,
                             chosen_action_counts, num_actions, num_trials_prob_best_action)

            writer.writerow(out_row)

        return chosen_actions, models


def switch_bandit_thompson(immediate_input, true_input, immediate_output,
                           true_output, time_step, action_mode, relearn=True,
                           use_regression=False, num_actions=3, Lambda=1):
    '''
    Run the algorithm on immediate-reward input up to specified time step then switch to the true-reward input and
    recompute policy by keeping the previously taken actions and matching with true rewards instead.
    :param immediate_input: The immediate-reward input file.
    :param true_input: The true-reward input file.
    :param immediate_output: The result output file from applying the algorithm to the immediate input.
    :param true_output: The result output file from applying the algorithm to the true input.
    :param time_step: The time step to switch bandit.
    :param action_mode: Indicates how to select actions, see ActionSelectionMode.
    :param relearn: At switch time, whether the algorithm will relearn from beginning.
    :param use_regression: Optional, indicate whether to use logistic regression to model reward distribution.
    :param num_actions: The number of actions in this bandit.
    :param Lambda: The prior inverse variance of the regression weights if regression is used.
    '''

    if use_regression:
        models = [RLogReg(D=NUM_FEATURES, Lambda=Lambda) for _ in range(num_actions)]
    else:
        models = [BetaBern(success=1, failure=1) for _ in range(num_actions)]

    # Run for 20 time steps on the immediate reward input
    chosen_actions, models = calculate_thompson_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        models,
        action_mode=action_mode,
        forced=forced_actions(time_step))

    # reset model state so that the algorithm forgets what happens
    for a in range(num_actions):
        models[a].reset_state()

    # Switch to true reward input, forcing actions taken previously
    calculate_thompson_single_bandit(
        true_input,
        num_actions,
        true_output,
        models,
        action_mode,
        forced_actions(actions=chosen_actions),
        relearn=relearn)


def switch_bandit_random_thompson(immediate_input, true_input, immediate_output,
                                  true_output, time_step, action_mode,
                                  relearn=True, use_regression=False,
                                  num_actions=3, Lambda=1):
    '''
    Similar to switch_bandit_thompson except that Random policy is run on the immediate data
    instead and thompson takes over once the switch happens.
    :param relearn: At switch time, whether the algorithm will relearn from beginning.
    '''

    if use_regression:
        models = [RLogReg(D=NUM_FEATURES, Lambda=Lambda) for _ in range(num_actions)]
    else:
        models = [BetaBern(success=1, failure=1) for _ in range(num_actions)]

    chosen_actions = calculate_random_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        forced=forced_actions(time_step))

    # Switch to true reward input, forcing actions taken previously
    calculate_thompson_single_bandit(
        true_input,
        num_actions,
        true_output,
        models,
        action_mode,
        forced_actions(actions=chosen_actions),
        relearn=relearn)


def main():
    num_actions = 3

    # init with inverse variance
    models = [RLogReg(D = NUM_FEATURES, Lambda = 1) for cond in range(num_actions)]

    #calculate_thompson_single_bandit('simulated_single_bandit_input.csv', 3, 'simulated_single_bandit_thompson.csv')
    calculate_thompson_single_bandit('contextual_single_bandit.csv', 3, 'contextual_single_bandit_thompson.csv', models)

if __name__ == "__main__":
    main()