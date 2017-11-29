import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, cross_validation
from bandit_data_format import *

'''
Experimental code using scikit-learn logistic regression API.
http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
'''


def load_synthesized_single_bandit_data(source):
    '''
    Load data from a single-bandit file, while synthesizing
    context features (such as age group etc...) in a
    manner correlated with the observed reward to test with
    regression. For example, if the reward is 1 then they 
    should be generated such that the sum of the features 
    are greater than some threshold.
    :param source: The single-bandit source file to load.
    '''

    rewards = []
    
    with open(source, newline='') as inf:
        reader = csv.DictReader(inf)

        for row in reader:
            rewards.append(np.array([int(row[HEADER_ACTUALREWARD.format(
                a + 1)]) for a in range(num_actions)]))
            pass
    pass

    rewards = np.array(rewards)

    # TODO: generate random features for now (e.g. age group, # days active).
    # Use actual 50 feature set later.
    age_group = np.random.randint(1, 6, size = rewards.shape[0])
    
    # correlate features with rewards for an action in the following way:
    # if (age group + # days active >= 7): reward = 1 else 0
    days_act  = np.zeros(rewards.shape[0])
    days_act[rewards[:, 1] > 0] = 7 - age_group[rewards[:, 1] > 0]
    days_act[days_act > 4] = 4

    features = np.vstack((age_group, days_act)).T

    return features, rewards


def load_mooclet_data(source, num_mooclets, num_actions):
    '''
    Load data from a multiple-bandit MOOClet file and returns
    (features, rewards) where features are the user context
    features, rewards is a 2-d array whose columns represent
    the observed rewards for one action in one mooclet. So if
    there are 3 mooclets each having 3 actions, rewards is then
    a (N x 9) 2-d array, where N is the number of samples.
    :param source: The MOOClet file to load.
    :param num_mooclets: Number of mooclets (or bandits) stored in source file.
    :param num_actions: Number of actions (or arms) per each mooclet.
    '''

    age_group = []
    days_act = []
    rewards = [[] for _ in range(num_mooclets * num_actions)]
    
    reward_names = np.array([[HEADER_MOOCLET_REWARD.format(
        m + 1, chr(ord('A') + m), a + 1) for a in range(
        num_actions)] for m in range(num_mooclets)]).ravel()

    with open(source, newline='') as inf:
        reader = csv.DictReader(inf)

        for row in reader:
            age_group.append(max(0, int(row[HEADER_AGEGROUP])))
            days_act.append(int(row[HEADER_DAYSACTIVE]))

            for i in range(len(reward_names)):
                rewards[i].append(int(row[reward_names[i]]))
            pass
    pass

    age_group = np.array(age_group)
    days_act = np.array(days_act)
    rewards = np.array(rewards).T

    features = np.vstack((age_group, days_act)).T

    return features, rewards


def cross_validate(X, Y, C, num_folds = 10):
    '''
    Cross-validate logistic regression with K-fold method.
    :param X: The input features.
    :param Y: The output labels.
    :param C: Inverse of regularization strength.
    :param num_folds: Number of folds.
    '''

    rmse_trains = []
    accuracy_trains = []
    rmse_tests = []
    accuracy_tests = []

    for fold in range(num_folds):
        
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, Y, test_size = 1.0 / num_folds)

        logreg = linear_model.LogisticRegression(C=C)
        logreg.fit(X_train, y_train)

        # predict on training set
        Z_train = logreg.predict(X_train)
        diff_train = Z_train - y_train
        rmse_train = np.sqrt(np.sum(diff_train ** 2))
        accuracy_train = 100.0 - np.count_nonzero(diff_train) * 100.0 / len(diff_train)

        # predict on training set
        Z_test = logreg.predict(X_test)
        diff_test = Z_test - y_test
        rmse_test = np.sqrt(np.sum(diff_test ** 2))
        accuracy_test = 100.0 - np.count_nonzero(diff_test) * 100.0 / len(diff_test)

        # add to list for computing average
        rmse_trains.append(rmse_train)
        rmse_tests.append(rmse_test)
        accuracy_trains.append(accuracy_train)
        accuracy_tests.append(accuracy_test)
        
        pass
    
    return np.mean(rmse_trains), np.mean(accuracy_trains), np.mean(rmse_tests), np.mean(accuracy_tests)


def cross_validate_C(X, Y, C_values):
    '''
    Cross-validate logistic regression with K-fold method
    to select the best regularization value.
    :param X: The input features.
    :param Y: The output labels.
    :param C_values: Inverse of regularization strength
        as listed here
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
    '''

    best_rmse_test = sys.maxsize
    best_C = -1

    for C in C_values:
        rmse_train, accuracy_train, rmse_test, accuracy_test = cross_validate(X, Y, C)

        if best_rmse_test > rmse_test:
            best_C = C
            best_rmse_test = rmse_test
            best_rmse_train = rmse_train
            best_acc_train = accuracy_train
            best_acc_test = accuracy_test

    print('MOOClet {}, Action {}\n'
            'C = {}\n'
            'err-train = {}, acc-train = {} %\n'
            'err-test = {}, acc-test = {} %\n'.format(
        mooclet + 1, action + 1, 
        best_C, 
        best_rmse_train, best_acc_train, 
        best_rmse_test, best_acc_test))

    return best_C


def plot_email_data(features, rewards, feature_labels = "", title = ""):
    '''
    Plot email data, only works with 2-d features for now.
    '''

    plt.figure()
    plt.scatter(features[:, 0], features[:, 1], c = rewards, edgecolors='k', cmap=plt.cm.Paired)

    if feature_labels != "":
        plt.xlabel(feature_labels[0])
        plt.ylabel(feature_labels[1])
        pass

    plt.title(title)

    plt.show()

    pass

''' DRIVER CODE '''

def main():
    np.random.seed(0)
    num_mooclets = 3
    num_actions = 3
    single_bandit_source = 'simulated_single_bandit_input.csv'
    mooclet_source = 'simulated_data_files_input.csv'

    #X, rewards = load_email_data(source)
    X, rewards = load_mooclet_data(mooclet_source, num_mooclets, num_actions)

    for mooclet in range(num_mooclets):
        for action in range(num_actions):
            Y = rewards[:, mooclet * num_actions + action]

            # plot input data for visual aid
            #plot_email_data(X, Y)

            best_C = cross_validate_C(X, Y, C_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2])

if __name__ == "__main__":
    main()
