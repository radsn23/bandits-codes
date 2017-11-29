import sys
import csv
import random
import math
import numpy as np
from forced_actions import forced_actions


def calculate_random_single_bandit(source, num_actions, dest, forced=forced_actions()):
    '''
    Calculates non-contextual random policy.
    :param source: simulated single-bandit data file with default rewards for each action and true probs.
    :param num_actions: number of actions for this bandit
    :param dest: outfile for printing the chosen actions and received rewards.
    :param forced: Optional, indicates to process only up to a certain time step or force take specified actions.
    '''
    # number of trials used to run Thompson Sampling to compute expectation stats
    # set to small value when debugging for faster speed
    num_trials_prob_best_action = int(1e4)

    # constant header names for easy indexing

    # data group from input file
    H_DATA_SAMPLE_NUMBER = "SampleNumber"
    H_DATA_AGE_GROUP = "agequartilesUSER"
    H_DATA_DAYS_ACTIVE = "ndaysactUSER"
    H_DATA_ACTUAL_REWARD = "Action{}OracleActualReward"
    H_DATA_TRUE_PROB = "Action{}OracleProbReward"
    H_DATA_OPTIMAL_ACTION = "ExpectedOptimalAction"

    # algorithm performance
    H_ALGO_ACTION = "AlgorithmAction"
    H_ALGO_OBSERVED_REWARD = "ObservedRewardofAction"
    H_ALGO_MATCH_OPTIMAL = "MatchesOptimalExpectedAction"
    H_ALGO_SAMPLE_REGRET = "SampleRegret"
    H_ALGO_SAMPLE_REGRET_CUMULATIVE = "CumulativeSampleRegret"
    H_ALGO_REGRET_EXPECTED = "ExpectedRegret"
    H_ALGO_REGRET_EXPECTED_CUMULATIVE = "CumulativeExpectedRegret"

    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)

        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out = field_names[:]
        field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                                H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                                H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])

        # not important, store the position to write high level header to output file
        group_header_parameters_index = len(field_names_out)

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

        for row in reader:
            sample_number += 1

            if len(forced.actions) == 0 or sample_number > len(forced.actions):
                # take a random action
                action = np.random.randint(0, num_actions)
            else:
                # take forced action if requested
                action = forced.actions[sample_number - 1]

            # only return action chosen up to specified time step
            if forced.time_step > 0 and sample_number <= forced.time_step:
                chosen_actions.append(action)

            # get reward signals
            observed_rewards = [int(row[H_DATA_ACTUAL_REWARD.format(a + 1)]) for a in range(num_actions)]
            reward = observed_rewards[action]

            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]

            ''' write performance data (e.g. regret) '''
            optimal_action = int(row[H_DATA_OPTIMAL_ACTION]) - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            out_row[H_ALGO_ACTION] = action + 1
            out_row[H_ALGO_OBSERVED_REWARD] = reward
            out_row[H_ALGO_MATCH_OPTIMAL] = 1 if optimal_action == action else 0
            out_row[H_ALGO_SAMPLE_REGRET] = sample_regret
            out_row[H_ALGO_SAMPLE_REGRET_CUMULATIVE] = cumulative_sample_regret

            true_probs = [float(row[H_DATA_TRUE_PROB.format(a + 1)]) for a in range(num_actions)]

            # The oracle always chooses the best arm, thus expected reward
            # is simply the probability of that arm getting a reward.
            optimal_expected_reward = true_probs[optimal_action] * num_trials_prob_best_action

            # Run random sampling many times and calculate how much reward it would
            # have gotten based on the chosen actions.
            chosen_action_counts = np.bincount(np.random.randint(0, num_actions, num_trials_prob_best_action))
            expected_reward = np.sum(chosen_action_counts[a] * true_probs[a] for a in range(num_actions))

            expected_regret = optimal_expected_reward - expected_reward
            cumulative_expected_regret += expected_regret

            out_row[H_ALGO_REGRET_EXPECTED] = expected_regret
            out_row[H_ALGO_REGRET_EXPECTED_CUMULATIVE] = cumulative_expected_regret

            writer.writerow(out_row)

        return chosen_actions


def switch_bandit_random(immediate_input, true_input, immediate_output, true_output, time_step, num_actions=3):
    '''
    Run the algorithm on immediate-reward input up to specified time step then switch to the true-reward input and
    recompute policy by keeping the previously taken actions and matching with true rewards instead.
    :param immediate_input: The immediate-reward input file.
    :param true_input: The true-reward input file.
    :param immediate_output: The result output file from applying the algorithm to the immediate input.
    :param true_output: The result output file from applying the algorithm to the true input.
    :param time_step: The time step to switch bandit.
    :param num_actions: The number of actions in this bandit.
    '''

    # Run for 20 time steps on the immediate reward input
    chosen_actions = calculate_random_single_bandit(immediate_input, num_actions, immediate_output,
                                                    forced_actions(time_step))

    # Switch to true reward input, forcing actions taken previously
    chosen_actions_2 = calculate_random_single_bandit(true_input, num_actions, true_output,
                                                      forced_actions(actions=chosen_actions))

    return chosen_actions + chosen_actions_2


def main():
    # calculate_random_single_bandit('../../data/sim_random.csv', 3, '../../data/sim_random.csv')
    print(switch_bandit_random('../../data/test_random/sim_random.csv',
                               '../../data/test_random/sim_random.csv',
                               '../../data/test_random/sim_random_imm.csv',
                               '../../data/test_random/sim_random_true.csv',
                               time_step=60, num_actions=3))


if __name__ == "__main__":
    main()
