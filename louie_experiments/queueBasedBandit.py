'''
Created on Jul 26, 2016

@author: arafferty
'''
import beta_bernoulli
import numpy as np
import numpy.random as nprand
import csv
from output_format import *
from bandit_data_format import *

SAMPLE_TIME = 0
ARRIVAL_TIME = 1
ARM_INDEX = 2
IMMEDIATE_REWARD = 3
FINAL_REWARD = 4

def get_thompson_arm_choice(models):
    # Pull each arm, ignoring context for now
    samples = [models[a].draw_expected_value(0) for a in range(len(models))]
    #find the max of samples[i] etc and choose an arm
    action = np.argmax(samples)
    return action

def get_thompson_sample_distribution(models, num_samples = 100, arm_choice=None):
    """
    Assumes models is a list containing one model per action such 
    that each model contains our representation of that arm. We assume
    the model contains a method draw_expected_value(context, num_samples) that
    allows us to draw the expected value of the arm. context of 0 means ignore
    context.
    
    num_samples is how many samples from the arms to use to make our empirical distribution.
    
    If arm_choice is None, it's ignored. Otherwise, we ensure close to 
    epsilon = 1/num_samples probability is placed on on the model with index equal
     to arm_choice . Crucially, we want arm_choice to not have 0 probability.
    Typically done because arm_choice is the arm that the base_distribution has selected, 
    meaning it's the one that doesn't have a maximum probability placed on it and 
    it will have at least some probability placed on it if probabilities are 
    reallocated in the sampling distribution.
    """
    samples = [model.draw_expected_value(0, num_samples) for model in models]        
    frequencies = np.argmax(samples,0)
    values, counts = np.unique(frequencies, return_counts = True)
    probs = []
    for i in range(len(models)):
        index, = np.where(values == i) # unpack the returned tuple so the array is stored inindex, datatype is discarded
        if len(index) != 0:
            try:
                counts[index[0]]
            except:
                print("issue")
            probs.append(counts[index[0]])
        else:
            probs.append(0)
                
    if arm_choice != None and probs[arm_choice] == 0:
        # Reallocate some probability
        probs[arm_choice] = 1
    return probs / sum(probs)
        
        
        


def get_sampling_dist(heuristic_dist, base_dist, num_actions, target_index, queue_sizes, max_queue_size, mixing_weight):
    """
    Implementation of algorithm 2 from paper
    heuristic_dist - distribution over arms for the heuristic
    base_dist - distribution over arms for the base algorithm
    num_actions - total number of arms
    target_index - arm index that we never modify - this is the one that was sampled from the base distribution 
    queue_sizes - list of the number of items in each arm queue (prevents heuristic from filling anything too full; S in paper)
    max_queue_size - maximum number of items in a single queue (B in paper)
    mixing_weight - how much of base_dist to include initially with heuristic_dist (alpha in paper)
    """
    arm_set = [a for a in range(num_actions) if a != target_index]
    sampling_dist = heuristic_dist # q in paper, this is the distribution that is close to but not exactly heuristic dist
    sampling_dist = (1-mixing_weight)*sampling_dist + mixing_weight*base_dist
    
    max_allowed_prob = np.zeros(len(sampling_dist)) # u in paper
    for a in range(len(sampling_dist)):
        max_allowed_prob[a] = max((max_queue_size - queue_sizes[a]) / max_queue_size, 0)
    
    cur_index = get_first_exceeded_index(sampling_dist[arm_set], max_allowed_prob[arm_set])
    while cur_index != -1: 
        # Determine which action this index corresponds to
        arm_number = arm_set[cur_index]

        #calculate the excess mass to redistribute
        excess =  sampling_dist[arm_number] - max_allowed_prob[arm_number]
        
        # cap this arms probability and remove it from things that are getting more probability
        sampling_dist[arm_number] = max_allowed_prob[arm_number]
        arm_set.pop(cur_index)
        
        # redistribute extra mass
        # Note: Talked to Yun-En, author on original paper, and original has error on lines
        # 10-12 ; should include original arm I as one that's getting mass added to it
        arm_set_sum = sum(sampling_dist[arm_set]) + sampling_dist[target_index]
        if arm_set_sum == 0:
            print("problem - arm_set_sum is zero")
        for a in arm_set:
            sampling_dist[a] = sampling_dist[a] * (arm_set_sum + excess) / arm_set_sum # Ratio scales each remaining arm equally
        # redistribute to target_index arm as well
        sampling_dist[target_index] = sampling_dist[target_index] * (arm_set_sum + excess) / arm_set_sum 
        # now get the next index to modify
        cur_index = get_first_exceeded_index(sampling_dist[arm_set], max_allowed_prob[arm_set])
    #print(sampling_dist)
    return sampling_dist
        
    
def get_first_exceeded_index(array1, array2):
    """
    Returns the index of the first item in array1 that is larger than the
    corresponding item in array2. E.g., if array1 = [3, 4, 2] and array2 = [10,3,1],
    it returns 1 as 1 is the smallest i such that array1[i] > array2[i]. Returns -1
    if no such i exists.
    """
    for i in range(len(array1)):
        if array1[i] > array2[i]:
            return i
        
    return -1

def read_reward_file(source, num_actions = 3):
    """
    Reads in a file that has the rewards for each arm on a sample by sample basis
    source - filename to read from
    """
    samples = []
    with open(source, newline='') as inf:
        reader = csv.DictReader(inf)
        for row in reader:
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            samples.append(observed_rewards)
    return samples

def switch_bandit_queue(immediate_input, true_input, immediate_output, true_output, 
    time_step_switch, total_time_steps, num_actions = 3):
    
    #Reward info for samples
    samples_with_true_reward = read_reward_file(true_input, num_actions)
    samples_with_immediate_reward = read_reward_file(immediate_input, num_actions)
    
    # Store the samples we have but that haven't yet arrived
    samples = []
    cur_sample_number = 0
    
    # Keep track of what actions are chosen so we can write out the results at the end
    chosen_actions = []
    sampling_distributions = []
    
    #Queue algorithm variables
    num_samples = 1
    mixing_weight = .01 # User defined mixing weight for how much to trust heuristic Policy (alpha in paper)
    queues = [[] for _ in range(num_actions)] # queues for holding samples
    max_queue_size = 1 # limit on queue size (B in paper)
    queue_sizes = np.zeros(num_actions)
    delays = [] # records how long it is between sample being selected and arriving (L in paper) 
    models_heuristic = [beta_bernoulli.BetaBern(success = 1, failure = 1) for _ in range(num_actions)] # Thompson sampling stats for heuristic(immediate); h in paper
    models_base = [beta_bernoulli.BetaBern(success = 1, failure = 1) for _ in range(num_actions)] # Thompson sampling stats for base(delayed)
    heuristic_dist = get_thompson_sample_distribution(models_heuristic) # approx distribution over arms for heuristic; h in paper
    #base_dist = get_thompson_sample_distribution(models_base) # approx distribution over arms for base; p in paper
    
    arm_choice = get_thompson_arm_choice(models_base) #Draw first action choice from base distribution (I in paper)
    while cur_sample_number < total_time_steps:
        while len(queues[arm_choice]) != 0:
            reward = queues[arm_choice].pop(0) # Get the new reward
            queue_sizes[arm_choice] -= 1
            # update base model with this reward
            # converted to range {-1,1} - this is based on standard thompson sampling doing this, although I don't see why it needs to 
            #(looking at BetaBernoulli code, it doesn't need to)
            models_base[arm_choice].update_posterior(0, 2 * reward - 1)
            
            # resample arm_choice
            arm_choice = get_thompson_arm_choice(models_base) 
            
        # resample base arm distribution
        base_dist = get_thompson_sample_distribution(models_base, arm_choice=arm_choice)
        heuristic_dist = get_thompson_sample_distribution(models_heuristic) # approx distribution over arms for heuristic; h in paper
    
#         for base_model,i  in zip(models_base, range(len(models_base))):
#             print("Base model", i, "successes", base_model.success, "failures", base_model.failure)
#         for heuristic_model,i  in zip(models_heuristic, range(len(models_heuristic))):
#             print("Heur model", i, "successes", heuristic_model.success, "failures", heuristic_model.failure)
#             
    
    
        sampling_dist = get_sampling_dist(heuristic_dist, base_dist, num_actions, arm_choice, queue_sizes, max_queue_size, mixing_weight)
        if sum(sampling_dist) < .995:
            print("Sampling dist not a probability distribution")
            sampling_dist = get_sampling_dist(heuristic_dist, base_dist, num_actions, arm_choice, queue_sizes, max_queue_size, mixing_weight)
        # sample from environment (paper notes one if online updates or else for one batch)
        for _ in range(num_samples):
            i = np.argmax(nprand.multinomial(1, sampling_dist))
            # need to get this sample and put it in our samples list
            # sample stores when it was selected, the time step it will arrive, the arm choice, the immediate reward, and the final reward
            sample = (cur_sample_number, max(time_step_switch, cur_sample_number), i, 
                      2*samples_with_immediate_reward[cur_sample_number][i]-1,  2*samples_with_true_reward[cur_sample_number][i] - 1)
            # observe the reward for the heuristic bandit if we aren't yet past the switch time (this isn't part of the paper)
            # we always observe it here because we'll always remove it when the sample arrives
            models_heuristic[i].update_posterior(0, sample[IMMEDIATE_REWARD])
            
            # store the sample so we'll be able to check it when it arrives
            samples.append(sample)
            
            # store the action so we can write it out to report results
            chosen_actions.append(i)
            sampling_distributions.append(sampling_dist)
            
            # increment queue size and sample counts based on sample
            queue_sizes[i] += 1
            cur_sample_number += 1
            

            
        # now need to find out which of the samples have arrived - i.e., delayed reward has come in
        samples_to_remove = []
        for sample in samples:
            # check if the arrival time is here
            if sample[ARRIVAL_TIME] <= cur_sample_number:
                # this sample has arrived - need to update the heuristic model, put it in a queue and mark it for removal
                queues[sample[ARM_INDEX]].append(sample[FINAL_REWARD])
                samples_to_remove.append(sample)
                # update the heuristic by dropping immediate reward from model and adding final reward
                models_heuristic[sample[ARM_INDEX]].remove_from_model(0, sample[IMMEDIATE_REWARD])
                models_heuristic[sample[ARM_INDEX]].update_posterior(0, sample[FINAL_REWARD])
                
                # record the delay of this sample
                delays.append(cur_sample_number - sample[SAMPLE_TIME])
        
        # remove any of the samples that arrived this time around
        for sample in samples_to_remove:
            samples.remove(sample)
            
        # set max_queue_size to maximum delay
        # Problem: can't take the max of an empty list
        # Bigger semantic problem: what should the maximum delay be if nothing has yet arrived?
        if len(delays) != 0:
            max_queue_size = max(delays)
        else:
            max_queue_size += num_samples # increment max queue size base on how many we've seen so far
    
    # At the end, we write out 
    writeOutFile(true_input, true_output, chosen_actions, num_actions, sampling_distributions)
    writeOutFile(immediate_input, immediate_output, chosen_actions, num_actions, sampling_distributions)

def create_headers(field_names, num_actions):
    # Construct output column header names
    field_names_out = field_names[:]
    field_names_out.extend([H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD, H_ALGO_MATCH_OPTIMAL,
                            H_ALGO_SAMPLE_REGRET, H_ALGO_SAMPLE_REGRET_CUMULATIVE,
                            H_ALGO_REGRET_EXPECTED, H_ALGO_REGRET_EXPECTED_CUMULATIVE])
        
    # not important, store the position to write high level header to output file
    group_header_parameters_index = len(field_names_out)

    for a in range(num_actions):
        field_names_out.append(H_ALGO_ESTIMATED_PROB.format(a + 1))


    # print group-level headers for readability
    group_header = ['' for _ in range(len(field_names_out))]
    group_header[0] = "Input Data"
    group_header[len(field_names)] = "Algorithm's Performance"
    group_header[group_header_parameters_index] = "Model Parameters"

    return field_names_out, group_header


def writeOutFile(infile, outfile, chosen_actions, num_actions, sampling_distributions):
    with open(infile, newline='') as inf, open(outfile, 'w', newline='') as outf:
        reader = csv.DictReader(inf)
        # Construct output column header names
        field_names = reader.fieldnames
        field_names_out, group_header = create_headers(field_names, num_actions)
        
        print(','.join(group_header), file=outf)

        writer = csv.DictWriter(outf, fieldnames=field_names_out)
        writer.writeheader()
        
        sample_number = 0
        cumulative_sample_regret = 0
        
        for row in reader:
            # copy the input data to output file
            out_row = {}

            for i in range(len(reader.fieldnames)):
                out_row[reader.fieldnames[i]] = row[reader.fieldnames[i]]
                
            ''' write performance data (e.g. regret) for this sample'''
            action = chosen_actions[sample_number]
            observed_rewards = [int(row[HEADER_ACTUALREWARD.format(a + 1)]) for a in range(num_actions)]
            sampling_dist = sampling_distributions[sample_number]
            reward = observed_rewards[action]
                
            optimal_action = int(row[HEADER_OPTIMALACTION]) - 1
            optimal_action_reward = observed_rewards[optimal_action]
            sample_regret = optimal_action_reward - reward
            cumulative_sample_regret += sample_regret

            out_row[H_ALGO_ACTION] = action + 1
            out_row[H_ALGO_OBSERVED_REWARD] = reward
            out_row[H_ALGO_MATCH_OPTIMAL] = 1 if optimal_action == action else 0
            out_row[H_ALGO_SAMPLE_REGRET] = sample_regret
            out_row[H_ALGO_SAMPLE_REGRET_CUMULATIVE] = cumulative_sample_regret
            
            for a in range(num_actions):
                out_row[H_ALGO_ESTIMATED_PROB.format(a + 1)] = sampling_dist[a]
           

            writer.writerow(out_row)
            sample_number += 1
        
def main():
    # Primarily for debugging
    immediate_input_file = "/Users/arafferty/git/banditalgorithms/src/louie_experiments/imm/input/gauss_single_bandit_input_immediate_cor-0.25sample1.csv"
    true_input_file = "/Users/arafferty/git/banditalgorithms/src/louie_experiments/true/input/gauss_single_bandit_input_true_cor-0.25sample1.csv"
    #immediate_input_file = "/Users/rafferty/banditalgorithms/data/gauss_single_bandit_input_immediate_cor0.75sample1.csv"
    #true_input_file = "/Users/rafferty/banditalgorithms/data/gauss_single_bandit_input_true_cor0.75sample1.csv"
    immediate_output_file = "/Users/arafferty/git/banditalgorithms/src/louie_experiments/true/input/gauss_single_bandit_output_debug3_immediate_cor-0.25sample1.csv"
    true_output_file = "/Users/arafferty/git/banditalgorithms/src/louie_experiments/true/input/gauss_single_bandit_output_debug3_true_cor-0.25sample1.csv"

    switch_bandit_queue(immediate_input_file, true_input_file, immediate_output_file, true_output_file, 120, 240, 3)


if __name__ == "__main__":
    main()
    