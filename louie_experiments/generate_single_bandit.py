from __future__ import print_function
import sys
import random
import csv
import numpy as np
from bandit_data_format import *


def combine_file(src_immediate, src_true, dest):
    '''
    Combine two single-bandit files into one file to simulate the effect
    of one bandit with immediate approximate rewards and the other with
    delayed true rewards.
    :param src_immediate: The source file containing immediate approx reward data.
    :param src_true: The source file containing delayed true reward data.
    :param dest: The destination output file that is combined from the above two.
    '''

    out_group_header = ''
    out_header = ''

    with open(src_immediate, newline='') as imm_f:
        reader = csv.DictReader(imm_f)

        # Set high level group headers in output file to Immediate Data
        imm_names = reader.fieldnames
        high_level_header = ['' for _ in imm_names]
        high_level_header[0] = 'Immediate Data'
        out_group_header += ','.join(high_level_header)
        out_header += ','.join(imm_names)

        # read all lines
        imm_lines = imm_f.readlines()

    out_group_header += ','  # separate groups of data
    out_header += ','

    with open(src_true, newline='') as true_f:
        reader = csv.DictReader(true_f)

        # Set high level group headers in output file to True Data
        true_names = reader.fieldnames
        high_level_header = ['' for _ in true_names]
        high_level_header[0] = 'True Data'
        out_group_header += ','.join(high_level_header)
        out_header += ','.join(true_names)

        # read all lines
        true_lines = true_f.readlines()

    with open(dest, 'w', newline='') as out_f:
        print(out_group_header, file=out_f)
        print(out_header, file=out_f)

        for r in range(len(imm_lines)):
            print('{},{}'.format(imm_lines[r].strip(), true_lines[r].strip()), file=out_f)


def generate_file(true_probs, num_rows, out_file):
    '''
    Generate a single-bandit data file according to the specified true probabilities
    of each arm generating a reward with specified number of rows.
    :param true_probs: True reward probabilities for each arm. This can be specified
                       in 3 ways:
                       1. If it is a 1d array then each value is interpreted as true
                          reward probability for an arm.
                       2. If it is a 2d array then each row corresponds to reward
                          probabilities for each time step.
                       3. If 3d array, then the first 2 dimensions specify the context
                          features: age group and # days active.
    :param num_rows: Number of rows to generate.
    :param out_file: Output file to generate.
    '''
    with open(out_file, 'w', newline='') as f:

        prob_mode_single = 0
        prob_mode_multiple = 1
        prob_mode_contextual = 2

        if true_probs.ndim == 1:
            prob_mode = prob_mode_single
            num_actions = len(true_probs)
        elif true_probs.ndim == 2:
            prob_mode = prob_mode_multiple
            num_actions = true_probs.shape[1]
        elif true_probs.ndim == 3:
            prob_mode = prob_mode_contextual
            num_age, num_days, num_actions = true_probs.shape

        # construct header column names
        field_names_out = [HEADER_SAMPLENUMBER, HEADER_AGEGROUP, HEADER_DAYSACTIVE]

        # the reward realization of actions based on true probability at each time step
        for a in range(num_actions):
            field_names_out.append(HEADER_ACTUALREWARD.format(a + 1))

        # the true probability of generating a reward for each action
        # this is constant across time steps
        for a in range(num_actions):
            field_names_out.append(HEADER_TRUEPROB.format(a + 1))

        # the optimal action in expectation based on the true probabilities of reward
        field_names_out.append(HEADER_OPTIMALACTION)

        # write header column names
        writer = csv.DictWriter(f, fieldnames=field_names_out)
        writer.writeheader()

        # generate random context group
        if prob_mode == prob_mode_contextual:
            ages = np.random.randint(0, num_age, num_rows)
            days = np.random.randint(0, num_days, num_rows)

        # generate random data according to true probabilities
        sample_number = 0
        for r in range(num_rows):
            sample_number += 1

            current_row = {}
            current_row[HEADER_SAMPLENUMBER] = sample_number

            if prob_mode == prob_mode_contextual:
                # get context group generated randomly in previous step
                age = ages[sample_number - 1]
                day = days[sample_number - 1]
                probs = true_probs[age, day, :]

                # write out context features
                current_row[HEADER_AGEGROUP] = age
                current_row[HEADER_DAYSACTIVE] = day
            elif prob_mode == prob_mode_multiple:
                probs = true_probs[sample_number - 1, :]
            elif prob_mode == prob_mode_single:
                probs = true_probs

            for a in range(num_actions):
                # flip a coin and set actual reward based on true probability of getting one
                prob_reward = random.random()
                reward = 0
                if prob_reward < probs[a]:
                    reward = 1
                current_row[HEADER_ACTUALREWARD.format(a + 1)] = reward

                # set true probability
                current_row[HEADER_TRUEPROB.format(a + 1)] = probs[a]

            # set expected optimal action, which is just the action with highest probability
            # of generating a reward
            current_row[HEADER_OPTIMALACTION] = np.argmax(probs) + 1

            writer.writerow(current_row)


def generate_probs(n1, n2, num_actions):
    """
    Generate true probabilities for each context group.
    :param n1: Number of unique values for feature 1 of the context.
    :param n2: Number of unique values for feature 2 of the context.
    :param num_actions: Number of actions.
    """

    # generate random [n1 x n2 x num_actions] nd array
    probs = np.random.rand(n1, n2, num_actions)

    # normalize values in 3rd dimension so that they
    # sum to 1 (i.e. probability of each action)
    probs = probs / np.reshape(np.tile(np.sum(
        probs, 2).ravel(), [num_actions, 1]).T, [
                                   n1, n2, num_actions])

    return probs


def perturb_probs(probs):
    """
    Perturb probability values by shifting horizontally by 1.
    """
    num_actions = probs.shape[2]
    new_probs = probs.copy()
    for k in range(num_actions - 1):
        new_probs[:, :, k] = probs[:, :, k + 1]
    new_probs[:, :, num_actions - 1] = probs[:, :, 0]

    return new_probs


def create_sample_noncontext():
    # The ground truth probability of each arm generating a reward.
    # To add more arm, simply add a new value to this array.
    true_probs = np.array([0.2, 0.5, 0.4])

    # The immediate approximate probability of each arm generating a reward.
    immediate_probs = np.array([0.2, 0.4, 0.5])

    # output file path to write to
    file_immediate = 'simulated_single_bandit_input_immediate.csv'
    file_true = 'simulated_single_bandit_input_true.csv'
    combined_file = 'simulated_delayed_rewards_bandit_input.csv'

    num_rows = 4266

    # generate data and write to file
    generate_file(immediate_probs, num_rows, file_immediate)
    generate_file(true_probs, num_rows, file_true)
    combine_file(file_immediate, file_true, combined_file)


def create_sample_contextual():
    true_probs = generate_probs(n1=NUM_AGE_LEVEL, n2=NUM_DAYS_LEVEL, num_actions=3)
    immediate_probs = perturb_probs(true_probs)

    # output file path to write to
    file_immediate = 'contextual_single_bandit_immediate.csv'
    file_true = 'contextual_single_bandit_true.csv'
    combined_file = 'contextual_delayed_rewards_bandit.csv'

    num_rows = 4266

    # generate data and write to file
    generate_file(immediate_probs, num_rows, file_immediate)
    generate_file(true_probs, num_rows, file_true)
    combine_file(file_immediate, file_true, combined_file)


def generate_normal_distribution_file(true_means, true_stds, num_rows, out_file):
    '''
    Generate a single-bandit data file according to the specified normal distribution
    of each arm generating a reward with specified number of rows.
    :param true_means: True reward mean of the normal distribution
                        for each arm.
    :param true_stds: True reward standard deviation of the normal distribution
                        for each arm.
    :param num_rows: Number of rows to generate.
    :param out_file: Output file to generate.
    '''
    with open(out_file, 'w', newline='') as f:
        num_actions = len(true_means)

        # construct header column names
        field_names_out = [HEADER_SAMPLENUMBER, HEADER_AGEGROUP, HEADER_DAYSACTIVE]

        # the reward realization of actions based on true normal distribution at each time step
        for a in range(num_actions):
            field_names_out.append(HEADER_ACTUALREWARD.format(a + 1))

        # the true means and std of generating a reward for the normal distribution of each action
        # this is constant across time steps
        for a in range(num_actions):
            field_names_out.append(HEADER_TRUEMEAN.format(a + 1))
            field_names_out.append(HEADER_TRUESTD.format(a + 1))

        # the optimal action in expectation based on the true normal distribution of reward
        field_names_out.append(HEADER_OPTIMALACTION)

        # write header column names
        writer = csv.DictWriter(f, fieldnames=field_names_out)
        writer.writeheader()

        # generate random data according to true normal distributions
        sample_number = 0
        for r in range(num_rows):
            sample_number += 1

            current_row = {}
            current_row[HEADER_SAMPLENUMBER] = sample_number

            means = true_means
            stds = true_stds

            for a in range(num_actions):
                reward = np.random.normal(means[a], stds[a])
                current_row[HEADER_ACTUALREWARD.format(a + 1)] = reward

                # set true means and stds
                current_row[HEADER_TRUEMEAN.format(a + 1)] = means[a]
                current_row[HEADER_TRUESTD.format(a + 1)] = stds[a]

            # set expected optimal action, which is just the action with highest mean
            # of generating a reward
            current_row[HEADER_OPTIMALACTION] = np.argmax(means) + 1

            writer.writerow(current_row)


def main():
    # create_sample_noncontext()
    create_sample_contextual()


if __name__ == "__main__":
    main()
