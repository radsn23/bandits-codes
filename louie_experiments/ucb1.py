import sys
import csv
import random
import math
import copy
import numpy as np
from forced_actions import forced_actions
from bandit_data_format import *
from random_policy import *


theta = 1.2
# treat_forced_as_historical: Whether to treat switch actions differently as in Shivaswamy, P. K., \& Joachims, T. (2012, April). Multi-armed Bandit Problems with History.
def calculate_ucb1_single_bandit(source, num_actions, dest, forced = forced_actions(), seed_rewards = None, 
                                 relearn = True, treat_forced_as_historical=False, use_sample_variance=False):
    '''
    Calculates non-contextual UCB1.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    :param seed_rewards: Optional, the initialized state of the model to start with (i.e. rewards received for each action).
    :param relearn: Optional, at switch time, whether algorithm relearns on previous time steps using actions taken previously.
    '''
    # number of trials used to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    # constant header names for easy indexing

    # algorithm performance
    H_ALGO_ACTION = "AlgorithmAction"
    H_ALGO_OBSERVED_REWARD = "ObservedRewardofAction"
    H_ALGO_MATCH_OPTIMAL = "MatchesOptimalExpectedAction"
    H_ALGO_SAMPLE_REGRET = "SampleRegret"
    H_ALGO_SAMPLE_REGRET_CUMULATIVE = "CumulativeSampleRegret"
    H_ALGO_REGRET_EXPECTED = "ExpectedRegret"
    H_ALGO_REGRET_EXPECTED_CUMULATIVE = "CumulativeExpectedRegret"
    
    # if we're treating the past actions (from forced) as historical, then
    # need to record how many forced actions there were of each type
    if treat_forced_as_historical:
        arm_counts_from_history = [0]*num_actions

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out = field_names[:]
        field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                                H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                                H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])

        # print group-level headers for readability
        group_header = ['' for i in range(len(field_names_out))]
        group_header[0] = "Input Data"
        group_header[len(field_names)] = "Algorithm's Performance"
        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()

        sample_number = 0
        cumulative_sample_regret = 0
        cumulative_expected_regret = 0

        chosen_actions = []
        
        # list of rewards gotten for each action
        if seed_rewards != None:
            rewards = seed_rewards
        else:
            rewards = [[] for _ in range(num_actions)]
        rewards_at_switch = []

        num_ucb_pulls = 0

        for row in reader:
            sample_number += 1

            should_update_rewards = True

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                num_ucb_pulls += 1
                if len(forced.actions) == 0 and num_ucb_pulls <= num_actions:
                    # initially play every action once
                    action = num_ucb_pulls - 1
                else:
                    action = -1
                    # This forces playing very action once; seems like above isn't necessary and may cause problems
                    for a in range(len(rewards)):
                        if len(rewards[a]) == 0:
                            action = a
                            break
                    if action == -1:
                        if treat_forced_as_historical:
                            # take action with max (avg reward + sqrt(2*log(# non historical arm choices + # of historical pulls of this arm) / (# times chosen + # historical pulls of this arm))
                            # note  that the number of times chosen plus the number of historical times chosen is exactly the total number of rewards recorded
                            #(that is, the only change with the historical version is to change the numerator)
#                             print("Historical: " + str([
#                                 np.mean(rewards_a) + \
#                                 np.sqrt(2.0 * np.log(num_ucb_pulls + historical_count) / len(rewards_a))
#                                 for rewards_a, historical_count in zip(rewards,arm_counts_from_history)]))
#                             print("Non-Historical: " +str([
#                                 np.mean(rewards_a) + \
#                                 np.sqrt(2.0 * np.log(sample_number) / len(rewards_a))
#                                 for rewards_a in rewards]))
#                             print("Variance: " + str([
#                                 np.mean(rewards_a) + \
#                                 np.sqrt(2.0 * theta * np.var(rewards_a) * np.log(num_ucb_pulls + historical_count) / len(rewards_a)) +\
#                                 3 * theta * np.log(num_ucb_pulls + historical_count) / len(rewards_a)
#                                 for rewards_a, historical_count in zip(rewards,arm_counts_from_history)]))
                            if use_sample_variance:
                                conf_bounds = [np.mean(rewards_a) + \
                                np.sqrt(2.0 * theta * np.var(rewards_a) * np.log(num_ucb_pulls + historical_count) / len(rewards_a)) +\
                                3 * theta * np.log(num_ucb_pulls + historical_count) / len(rewards_a)
                                for rewards_a, historical_count in zip(rewards,arm_counts_from_history)]
                            
                            else:
                                conf_bounds = [np.mean(rewards_a) + \
                                               np.sqrt(2.0 * np.log(num_ucb_pulls + historical_count) / len(rewards_a))
                                               for rewards_a, historical_count in zip(rewards,arm_counts_from_history)]                 
                        else:
                            if use_sample_variance:
                                conf_bounds = [np.mean(rewards_a) + \
                                               np.sqrt(2.0 * theta * np.var(rewards_a) * np.log(sample_number) / len(rewards_a)) + \
                                               3 * theta * np.log(sample_number) / len(rewards_a)
                                               for rewards_a in rewards]
                            else:
                                # take action with max (avg reward + sqrt(2*log(t) / # times chosen))
                                conf_bounds = [np.mean(rewards_a) + \
                                               np.sqrt(2.0 * np.log(sample_number) / len(rewards_a))
                                               for rewards_a in rewards]
                        action = np.argmax(conf_bounds)
            else:
                samples = [0 for a in range(num_actions)]
                # take forced action if requested
                action = forced.actions[sample_number - 1]

                if relearn == False:
                    should_update_rewards = False


            # get reward signals
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            if should_update_rewards:
                rewards[action].append(reward)

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)

                if sample_number == forced.time_step:
                    rewards_at_switch = copy.deepcopy(rewards)

            # update history counts if necessary
            if treat_forced_as_historical and sample_number <= len(forced.actions):
                arm_counts_from_history[action] += 1
            
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
            
            # TODO: compute expected regret for UCB1
            expected_regret = 0
            cumulative_expected_regret += expected_regret

            out_row[H_ALGO_REGRET_EXPECTED] = expected_regret
            out_row[H_ALGO_REGRET_EXPECTED_CUMULATIVE] = cumulative_expected_regret

            writer.writerow(out_row)
        
        return chosen_actions, rewards_at_switch


def switch_bandit_ucb1(immediate_input, true_input, immediate_output, true_output, 
    time_step, num_actions = 3, relearn=True, treat_forced_as_historical=False, use_sample_variance = False):
    '''
    Run the algorithm on immediate-reward input up to specified time step then switch to the true-reward input and
    recompute policy by keeping the previously taken actions and matching with true rewards instead.
    :param immediate_input: The immediate-reward input file.
    :param true_input: The true-reward input file.
    :param immediate_output: The result output file from applying the algorithm to the immediate input.
    :param true_output: The result output file from applying the algorithm to the true input.
    :param time_step: The time step to switch bandit.
    :param num_actions: The number of actions in this bandit.
    :param relearn: At switch time, whether the algorithm will relearn from past data.

    '''

    # Run for 20 time steps on the immediate reward input
    chosen_actions, rewards_at_switch = calculate_ucb1_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        forced_actions(time_step), use_sample_variance = use_sample_variance)

    # Switch to true reward input, forcing actions taken previously
    # Also initializes model with that of before switching time.
    calculate_ucb1_single_bandit(
        true_input,
        num_actions,
        true_output,
        forced_actions(actions = chosen_actions),
        seed_rewards = None,
        relearn = relearn,
        treat_forced_as_historical = treat_forced_as_historical, use_sample_variance = use_sample_variance)


def switch_bandit_random_ucb1(immediate_input, true_input, immediate_output, true_output, 
    time_step, num_actions = 3, relearn = True, treat_forced_as_historical=False):
    '''
    Similar to switch_bandit_ucb1 except that Random policy is run on the immediate data
    instead and UCB1 takes over once the switch happens.
    :param relearn: At switch time, whether the algorithm will relearn from past data.
    '''

    chosen_actions = calculate_random_single_bandit(
        immediate_input,
        num_actions,
        immediate_output,
        forced = forced_actions(time_step))

    # Switch to true reward input, forcing actions taken previously
    calculate_ucb1_single_bandit(
        true_input,
        num_actions,
        true_output,
        forced_actions(actions = chosen_actions),
        seed_rewards = None,
        relearn = relearn,
        treat_forced_as_historical = treat_forced_as_historical)


def main():
    calculate_ucb1_single_bandit('simulated_single_bandit_input.csv', 3, 'simulated_single_bandit_ucb1.csv')

if __name__ == "__main__":
    main()
