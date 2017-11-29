'''
Regularized Logistic Regression as listed in Algorithm 3
of "An Empirical Evaluation of Thompson Sampling"
'''

import autograd.numpy as np
from autograd import value_and_grad
import scipy.optimize as scopt
from scipy.special import expit
from base_model import *
from output_format import *


class RLogReg(BaseModel):


    def __init__(self, D, Lambda, online = True):

        # True if posterior is updated in an online way,
        # i.e. one one sample at a time. Otherwise, samples
        # are accumulated and posteriors are updated on
        # all samples up to current time.
        self.online = False

        # number of features
        self.D = D

        # initial value for inverse variance of prior
        self.Lambda = Lambda

        # number of iterations for optimization
        self.NumOptimizeIterations = 100

        # initialize parameters:
        # weight values = 0
        # prior means = 0
        # prior inverse variance = Lambda
        self.w = np.zeros(D)
        self.m = np.zeros(D)
        self.q = np.ones(D) * Lambda

        # get a function which computes the value
        # and gradient of the log posterior
        self.grad_logpost = value_and_grad(self.log_prob_and_gradient)

        # features of size (N x D)
        self.X = None

        # labels of size (N x 1)
        # values should be in {-1,1}
        self.Y = None

        self.has_data = False

        # storing states of the model
        self.last_w = self.w
        self.last_m = self.m
        self.last_q = self.q
        self.last_X = self.X
        self.last_Y = self.Y


    def log_prob_and_gradient(self, w):

        # first term of the objective function
        c1 = 0.5 * np.sum( self.q * (w - self.m)**2 )

        # second term
        c2 = np.sum(np.logaddexp(0, -self.Y * np.dot(self.X, w)))

        return c1 + c2


    def update_data(self, x, y):
        if self.online:
            self.X = x
            self.Y = y
        else:
            self.X = np.vstack((self.X, x)) if self.has_data else np.array([x])
            self.Y = np.hstack((self.Y, y)) if self.has_data else np.array([y])
        self.has_data = True


    def set_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.has_data = True


    def compute_map(self):
        # estimate minimum with scipy optimize
        result = scopt.minimize(self.grad_logpost, self.w,
            method='l-bfgs-b', jac=True, options={
            'maxiter': self.NumOptimizeIterations, 'disp': False})

        self.w = result.x


    def update_posterior(self, x, y):
        # update data and posterior for next round
        self.update_data(x, y)
        
        # compute MAP value of w
        self.compute_map()

        # update m_i = w_i
        self.m = self.w

        # update q_i by Laplace approximation
        P = expit(np.dot(self.X, self.w))
        self.q += np.dot(P * (1 - P), self.X ** 2)


    def draw_expected_value(self, x, num_samples = 1):
        mean = self.m
        std  = np.sqrt(1 / self.q)

        if num_samples <= 1:
            # draw samples from approximate posterior distribution
            w = np.random.normal(mean, std)

            # compute expected value
            u = np.dot(w, x)

            return expit(u) - expit(-u)
        else:
            # generate many samples from same distribution
            mean_tile = np.tile(mean, [num_samples, 1])
            std_tile  = np.tile(std , [num_samples, 1])
            W = np.random.normal(mean_tile, std_tile)
            U = np.dot(W, x)
            return expit(U) - expit(-U)


    def get_mean_variance(self):
        return self.m, 1.0 / self.q


    def predict(self, X):
        Y = expit(np.dot(X, self.w))
        Y[Y >= 0.5] =  1
        Y[Y <  0.5] = -1
        return Y


    def save_state(self):
        self.last_w = self.w
        self.last_m = self.m
        self.last_q = self.q
        self.last_X = self.X
        self.last_Y = self.Y


    def restore_state(self):
        self.w = self.last_w
        self.m = self.last_m
        self.q = self.last_q
        self.X = self.last_X
        self.Y = self.last_Y


    def write_parameters(self, out_row, action):
        # TODO: write out model parameters for debugging
        pass


    def reset_state(self):
        self.w = np.zeros(self.D)
        self.m = np.zeros(self.D)
        self.q = np.ones(self.D) * self.Lambda
        self.X = None
        self.Y = None
        self.has_data = False
        self.last_w = self.w
        self.last_m = self.m
        self.last_q = self.q
        self.last_X = self.X
        self.last_Y = self.Y