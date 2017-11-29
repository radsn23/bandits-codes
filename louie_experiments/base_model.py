

class BaseModel(object):
    '''
    Base model API.
    '''

    def update_posterior(self, x, y):
        pass


    def draw_expected_value(self, x, num_samples = 1):
        pass


    def predict(self, X):
        pass


    def write_parameters(self, out_row, action):
        pass


    def save_state(self):
        pass


    def restore_state(self):
        pass

    def reset_state(self):
        pass