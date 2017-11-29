import numpy as np
import csv
import glob
import os
import re
import pickle
from bandit_data_format import *
from output_format import *


def generate_gaussian_rewards(num_bandits, num_actions, m, c, v, n, out_file = ""):
    """
    Generate rewards that are correlated across bandits
    from a Multivariate Gaussian. For example, the reward of
    arm 1 in bandit 2 may be correlated with the reward of arm 1
    in bandit 1. This is modelled as a MVN distribution with the
    following covariance matrix (for two 2-arm bandits):
    ------------------------------------------------------------
          A1_B1 A2_B1 A1_B2 A2_B2
    A1_B1   v11   0     c1    0
    A2_B1   0     v21   0     c2
    A1_B2   c1    0     v12   0
    A2_B2   0     c2    0     v22
    ------------------------------------------------------------
    :param num_bandits: Number of bandits to generate.
    :param num_actions: Number of arms.
    :param m: A constant specifying the mean of the reward
              distribution for every arm in every bandit.
    :param c: A dictionary containing arrays specifying covariance
              values between the rewards for each arm across bandits.
              For example, this is { 1: [ [1,2,c1] ], 2: [ [1,2,c2] ] }
              for the 2-arm setup above. So the first entry above
              contains covariance values for arm 1 between bandit 1 and
              bandit 2. The covariance matrix is populated symmetrically.
    :param v: An array containing variance values for each arm in
              each bandit. For the example above, this is:
              [v11, v21, v12, v22].
    :param n: Number of reward samples to generate.
    :param out_file: Optional, output file to write MVN parameters to.
    """

    # array of mean values for each arm
    mean = np.ones(num_bandits * num_actions) * m

    # initialize covariance matrix
    cov = np.eye(num_bandits * num_actions)

    # set variance of each arm in each bandit
    for i in range(num_bandits * num_actions):
        cov[i, i] = v[i]

    # set covariance for each arm across bandits
    for a in range(num_actions):
        for cov_a in c.get(a + 1, []):
            assert len(cov_a) == 3, 'Covariance specifier must indicate ' + \
                'only the two bandits and the covariance value'

            bandit_1, bandit_2, cv = cov_a

            # adjust to 0-based
            bandit_1 -= 1
            bandit_2 -= 1

            assert bandit_1 < num_bandits and bandit_1 >= 0, 'invalid bandit index'
            assert bandit_2 < num_bandits and bandit_2 >= 0, 'invalid bandit index'

            # set values symmetrically
            cov[bandit_1 * num_actions + a, bandit_2 * num_actions + a] = cv
            cov[bandit_2 * num_actions + a, bandit_1 * num_actions + a] = cv

    # TODO: generalize to different values of mean and variance
    # dividing by 36 to make majority of samples within
    # [0,1]. This assumes mean = 0.5 and variance = 1, so that
    # std = 1/6, thus mean +- 3std is 0 and 1.
    cov = cov / 36

    if out_file != "":
        with open(out_file, 'w', newline='') as fp:
            fcsv = csv.writer(fp, delimiter=',')
            data = []

            # add header
            header = ['']
            for b in range(num_bandits):
                for a in range(num_actions):
                    header.append('Arm{}Bandit{}'.format(a + 1, b + 1))
            header.append('Mean')
            data.append(header)

            # add each data row
            for b in range(num_bandits):
                for a in range(num_actions):
                    row = ['Arm{}Bandit{}'.format(a + 1, b + 1)]
                    for i in range(num_bandits * num_actions):
                        row.append(cov[b * num_actions + a, i])
                    row.append(mean[b * num_actions + a])
                    data.append(row)

            fcsv.writerows(data)

    return np.random.multivariate_normal(mean, cov, n)


def load_matrix_from_table(table_file, num_algos = 9, num_correlations = 9):
    '''
    Load the MVN regret table result into matrix form.
    :param num_algos: Number of algorithms listed in the table file.
    :param num_correlations: Number of unique correlation values listed in the table file. 
                             Set to 9 by default because len([-1:1:0.25]) = 9
    '''
    reader = csv.reader(open(table_file, "r"), delimiter = ',')
    x = []
    for i in range(2 + num_correlations):
        x.append(next(reader))
    table = np.reshape(np.array([np.array(m[1:]) for m in x[2:]]).T, [num_algos,6,num_correlations]).astype(float).swapaxes(1,2)
    return table


def load_cumulative_regret_matrix(path, out_pickle_file):
    '''
    Load the MVN cumulative regret result into a dictionary/matrix.
    :param path: Path to folder containing the output results.
    :param out_pickle_file: Output path to save the pickled content.
    '''
    map = {}
    num_samples = 0
    file_meta_list = []

    files_list = glob.glob(path + os.sep + "gauss_output*.csv")
    for file_path in files_list:
        file_name = os.path.basename(file_path)
        match = re.match(r'gauss_output_(.*)_(.*)_t(.*)cor(.*)sample(.*).csv', file_name, re.M|re.I)
        algorithm = match.group(1)
        bandit_type = match.group(2)
        t_switch = int(match.group(3))
        correlation = float(match.group(4))
        sample = int(match.group(5))
        num_samples = max(sample, num_samples)
        file_meta_list.append((file_path, algorithm, t_switch, correlation, bandit_type, sample))

    for file_meta in file_meta_list:
        with open(file_meta[0], newline='') as inf:
            reader = csv.DictReader(inf)
            line_no = 0
            if HEADER_SAMPLENUMBER not in reader.fieldnames:
                # some output files have two-level headers
                # so need to read the second line.
                reader = csv.DictReader(inf)
                line_no = 1
            for row in reader:
                line_no += 1
                cum_regret = int(row[H_ALGO_SAMPLE_REGRET_CUMULATIVE])
                if int(row[HEADER_SAMPLENUMBER]) == t_switch:
                    cum_regret_switch = cum_regret
                cum_regret_final = cum_regret
                pass
            pass
        algorithm = file_meta[1]
        t_switch = file_meta[2]
        correlation = file_meta[3]
        bandit_type = file_meta[4]
        sample = file_meta[5]

        if algorithm not in map:
            map[algorithm] = {}
        if t_switch not in map[algorithm]:
            map[algorithm][t_switch] = {}
        if correlation not in map[algorithm][t_switch]:
            map[algorithm][t_switch][correlation] = np.zeros((num_samples, 4))

        if bandit_type == "immediate":
            map[algorithm][t_switch][correlation][sample - 1, 0] = cum_regret_switch
            map[algorithm][t_switch][correlation][sample - 1, 1] = cum_regret_final
        else:
            map[algorithm][t_switch][correlation][sample - 1, 2] = cum_regret_switch
            map[algorithm][t_switch][correlation][sample - 1, 3] = cum_regret_final

    with open(out_pickle_file, 'wb') as handle:
        pickle.dump(map, handle)
    return map


def split_table_file(table_file):
    '''
    Split gauss_all_table_t60.csv file into two files, one containing
    just the average regret table, the other containining all observed regrets
    over all trials.
    '''
    with open(table_file) as tf:
        with open('average_regret_table.csv', 'w') as avgf:
            with open('all_regret_table.csv', 'w') as allf:
                index = 0
                allf.write('algorithm,bandit type,t,correlation,sample,run,{}\n'.format(','.join([str(i + 1) for i in range(240)])))
                for line in tf:
                    index += 1
                    if index <= 11:
                        avgf.write(line)
                    else:
                        match = re.match(r'gauss_output_(.*)_(.*)_t(.*)cor(.*)sample(.*)run(.*).csv(.*)', line, re.M|re.I)
                        algorithm = match.group(1)
                        bandit_type = match.group(2)
                        t_switch = match.group(3)
                        correlation = match.group(4)
                        sample = match.group(5)
                        run = match.group(6)
                        remaining = match.group(7)
                        line = ','.join([algorithm, bandit_type, t_switch, correlation, sample, run]) + remaining + '\n'
                        allf.write(line)


def create_6D_MVN_file(path, output_path):
    '''
    Create a csv file containing all 6-d MVN samples generated in the input path.
    '''
    num_actions = 3
    files_list = glob.glob(path + os.sep + "gauss_single_bandit_input*.csv")
    
    name_to_prob_dict = {}
    for file_path in files_list:
        file_name = os.path.basename(file_path)
        match = re.match(r'gauss_single_bandit_input_(.*)_cor(.*)sample(.*).csv', file_name, re.M|re.I)
        bandit_type = match.group(1)
        correlation = match.group(2)
        sample = int(match.group(3))

        with open(file_path, newline='') as inf:
            reader = csv.DictReader(inf)
            line_no = 0
            if HEADER_SAMPLENUMBER not in reader.fieldnames:
                # some output files have two-level headers
                # so need to read the second line.
                reader = csv.DictReader(inf)
                line_no = 1
            for row in reader:
                probs = [float(row[HEADER_TRUEPROB.format(a + 1)]) for a in range(num_actions)]
                break # just need to read the first row since reward probs are constant over t
            
            name = (correlation, sample)

            if name not in name_to_prob_dict:
                name_to_prob_dict[name] = np.zeros(2 * num_actions)

            if bandit_type == 'true':
                name_to_prob_dict[name][num_actions:] = probs
            else:
                name_to_prob_dict[name][:num_actions] = probs

    with open(output_path, 'w', newline='') as outf:
        writer = csv.writer(outf, delimiter=',')
        data = []
        header = ['Correlation', 'Sample']
        header.extend(['Immediate-Prob-Arm-{}'.format(a + 1) for a in range(num_actions)])
        header.extend(['Delayed-Prob-Arm-{}'.format(a + 1) for a in range(num_actions)])
        data.append(header)
        for k,v in name_to_prob_dict.items():
            row = [k[0], k[1]] + list(v.astype(str))
            data.append(row)
        writer.writerows(data)

def main():
    #table = load_matrix_from_table('C:\\Users\\lhoang\\Google Drive\\Bandit Project Resources with Luong\\results\\non-contextual\\switch-bandit-MVN\\50MVN100Runs\\gauss_all_table_t60.csv')
    split_table_file('C:\\Users\\lhoang\\Google Drive\\Bandit Project Resources with Luong\\results\\non-contextual\\switch-bandit-MVN\\50MVN100Runs\\gauss_all_table_t60.csv')
    #create_6D_MVN_file('D:\\Git\\banditalgorithms\\data\\gauss-fixed', 'mvn_samples.csv')
    #generate_gaussian_rewards(3, [])
    #load_cumulative_regret_matrix(
    #    "C:\\Users\\lhoang\\Google Drive\\Bandit Project Resources with Luong\\results\\non-contextual\\switch-bandit-MVN\\include0.5",
    #    "C:\\Users\\lhoang\\Google Drive\\Bandit Project Resources with Luong\\results\\non-contextual\\switch-bandit-MVN\\include0.5\\cumulative_regret.pickle")


if __name__ == "__main__":
  main()