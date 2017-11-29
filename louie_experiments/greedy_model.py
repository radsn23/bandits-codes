import numpy as np
from base_model import *
from output_format import *


class Greedy(BaseModel):
    '''
    Greedy model for Epsilon Greedy to keep track of best arm so far.
    This model does not consider the context.
    '''

    def __init__(self):
        self.reward = 0.5
        self.num_selected = 1
        self.last_reward = 0


    def update_posterior(self, x, y):
        # update success/failure counts per observed reward
        if y == 1:
            self.reward += 1
        self.num_selected += 1


    def draw_expected_value(self, x, num_samples = 1):
        return self.reward / float(self.num_selected)


    def save_state(self):
        self.last_reward = self.reward


    def restore_state(self):
        self.reward = self.last_reward


    def reset_state(self):
        self.reward = 0.5
        self.num_selected = 1
        self.last_reward = 0
