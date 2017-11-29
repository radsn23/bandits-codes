import sys
import csv
import random
import math
import numpy


def constant_policy(num_actions):
    ''' 1-based action '''
    return 1


def getEpsilonGreedyAction(eps, num_actions, default_policy):
    '''
    Performs epsilon greedy exploration with specified default policy.
    This works as follows: in epsilon of the time, action index is
    uniformly chosen from [1, num_actions]. In the rest 1 - epsilon
    time, action is chosen by calling default policy.
    '''
    default_policy_action = default_policy(num_actions)
    chosen_action = -1
    chosen_action_prob = 0

    if random.random() < eps:
        # 1-based uniform action index
        chosen_action = random.randint(1, num_actions)

        # update probability if the default policy would have
        # chosen the same action
        if chosen_action == default_policy_action:
            chosen_action_prob = 1 - eps + (eps / num_actions)
        else:
            chosen_action_prob = eps / num_actions
    else:
        # choose action from default policy
        chosen_action_prob = 1 - eps + (eps / num_actions)
        chosen_action = default_policy_action

    return (chosen_action, chosen_action_prob)


def calculateEpsilonGreedyPolicy(source, dest, eps=0.1):
    '''
    Calculate epsilon greedy on the source dataset.
    :params source: The input source dataset (e.g. simulated_data_files_input.csv).
    :param dest: The output destination dataset.
    :param eps: Epsilon parameter.
    '''
    numActions = 3
    numMooclets = 3
    with open(source, newline='') as inf, open(dest, 'w', newline='') as outf:
        reader = csv.DictReader(inf)
        fieldNamesOut = reader.fieldnames[0:3]
        
        #output the conditions chosen
        fieldNamesOut.append('MOOClet1')
        fieldNamesOut.append('MOOClet2')
        fieldNamesOut.append('MOOClet3')
        
        #output our samples drawn
        fieldNamesOut.append('RewardMOOClet1')
        fieldNamesOut.append('RewardMOOClet2')
        fieldNamesOut.append('RewardMOOClet3')

        writer = csv.DictWriter(outf, fieldnames=fieldNamesOut)
        writer.writeheader()
        sampleNumber = 0
        for row in reader:
            sampleNumber += 1
            #get the user vars
            ageQuartile = int(row['agequartilesUSER']);
            #user 0 instead of -1 for age quartiles
            if ageQuartile==-1:
              ageQuartile=0;
            
            nDaysAct = int(row['ndaysactUSER']);
                
            #choose a random action
            actions = []
            for i in range(numMooclets):
                a, p = getEpsilonGreedyAction(eps, numActions, constant_policy)
                actions.append(a)

            # get reward signals
            rewards = []
            for i in range(numMooclets):
                row_key = 'MOOClet{}{}{}'.format(i + 1, chr(ord('A') + i), actions[i])
                rewards.append(int(row[row_key]))


            #write out some of the inputs, which versions we chose, samples
            writer.writerow({'SampleNumber' : sampleNumber, 'agequartilesUSER': ageQuartile, 'ndaysactUSER' : nDaysAct,
             'MOOClet1' : actions[0], 'MOOClet2' : actions[1], 'MOOClet3' : actions[2],
                             'RewardMOOClet1' : rewards[0], 'RewardMOOClet2' : rewards[1], 'RewardMOOClet3' : rewards[2]})

def main():
  if len(sys.argv) == 4: 
    calculateEpsilonGreedyPolicy(sys.argv[1], sys.argv[2], sys.argv[3])
  else:
    calculateEpsilonGreedyPolicy('simulated_data_files_input.csv', 'testEpsilonGreedy_simData.csv', eps=0.1)

if __name__ == "__main__":
  main()
