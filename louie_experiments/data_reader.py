import numpy as np
import scipy.stats
import sys
import csv
from bandit_data_format import *
from output_format import *


'''
Miscellaneous code for reading input and output data files.
'''

def read_avg_regret(source, switch_time_step):
    '''
    Read average sampled regret from source file and returns 4 values:
    the first is a list of all regret values incurred at each time step
    the next 3 values contain:
    one is average over all samples, second is average over samples
    up to specified timestep, third is average over samples from specified
    timestep until end of file.
    :param source: The source file to read from.
    :param switch_time_step: The time step of the switch.
    '''
    with open(source, newline='') as inf:
        reader = csv.DictReader(inf)
        if HEADER_SAMPLENUMBER not in reader.fieldnames:
            # some output files have two-level headers
            # so need to read the second line.
            reader = csv.DictReader(inf)
        pre_regret = []
        all_regret = []
        pos_regret = []
        for row in reader:
            regret = int(row[H_ALGO_SAMPLE_REGRET])
            if int(row[HEADER_SAMPLENUMBER]) < switch_time_step:
                pre_regret.append(regret)
            else:
                pos_regret.append(regret)
            all_regret.append(regret)

    if len(pre_regret) == 0:
        # some algorithms are oblivious to immediate data
        # so does not have regrets prior to switch
        pre_regret = [sys.maxsize]
    return all_regret, np.mean(all_regret), np.mean(pre_regret), np.mean(pos_regret)


def read_avg_regret_multiple(source_list, switch_time_step):
    '''
    Read average value of average sampled regret from all source files.
    Returns two values:
    1. The first is an array of 3 values for average regret:
       before switch, after switch, and both.
    2. The second is a list of tuples, each tuple contains the file
       name and a list of all regrets incurred at each time step.
    :param source: The list of source files to read from.
    :param switch_time_step: The time step of the switch.
    '''
    avg_regrets = []
    all_regrets = []
    for source in source_list:
        all, r_all, r_pre, r_pos = read_avg_regret(source, switch_time_step)
        avg_regrets.append([r_all, r_pre, r_pos])
        all_regrets.append((source, all))
    
    return np.mean(avg_regrets, 0), scipy.stats.sem(avg_regrets, axis=0, ddof=0), all_regrets
