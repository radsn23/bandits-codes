import numpy as np
from base_model import *
from output_format import *


class BetaBern(BaseModel):
    '''
    Beta-Bernoulli model for Thompson Sampling.
    This model does not consider the context.
    '''

    def __init__(self, success, failure):
        self.success = success
        self.failure = failure

        self.last_success = success
        self.last_failure = failure


    def update_posterior(self, x, y):
        # update success/failure counts per observed reward
        if y == 1:
            self.success += 1
        else:
            self.failure += 1
    

    def draw_expected_value(self, x, num_samples = 1):
        
        if num_samples > 1:
            success_tile = self.success * np.ones(num_samples)
            failure_tile = self.failure * np.ones(num_samples)
            return np.random.beta(success_tile, failure_tile)

        # draw a sample from Beta posterior which is
        # same as expected reward given this model.
        return np.random.beta(self.success, self.failure)

    def remove_from_model(self, x, y):
        """
        Removes 1 success or failure based on the value of y - return 
        model to state as if you'd never observed y. (Does not incorporate
        context variable x; present because the other methods
        in this class include it.)
        """
        if y == 1:
            self.success -= 1
        else:
            self.failure -= 1
        if self.success == 0 or self.failure == 0:
            print("success/failure zero")

    def write_parameters(self, out_row, action):
        # success count for each action
        out_row[H_ALGO_ACTION_SUCCESS.format(action + 1)] = self.success
                
        # failure count for each action
        out_row[H_ALGO_ACTION_FAILURE.format(action + 1)] = self.failure

        # estimated reward probability for each arm is simply the
        # mean of the current beta distribution
        out_row[H_ALGO_ESTIMATED_PROB.format(action + 1)] = \
            self.success / float(self.success + self.failure)


    def save_state(self):
        self.last_success = self.success
        self.last_failure = self.failure


    def restore_state(self):
        self.success = self.last_success
        self.failure = self.last_failure


    def reset_state(self):
        self.success = 1
        self.failure = 1
        self.last_success = self.success
        self.last_failure = self.failure