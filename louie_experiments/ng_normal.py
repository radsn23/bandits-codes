import numpy as np
import scipy
from scipy.stats import invgamma
from scipy.stats import gamma

from base_model import *
from output_format import *


class NGNormal(BaseModel):
    '''
       Similar to the NIG Normal in that we'll use it for the posterior of a model
       where we assume normally distributed random rewards, but we parameterize the
       model base on mean and precision (inverse of variance) rather than mean and
       variance. This uses a normal gamma rather than a normal inverse gamma prior,
       where it's easier to find the posterior predictive distributions. 
    '''

    init_mu = 0
    init_k = 1
    init_alpha = 1
    init_beta = 1

    def __init__(self, mu, k, alpha, beta):
        self.mu = mu  # distribution mean
        self.k = k  # related to variance of the normal part of the prior, likely in a 1/k way
        self.alpha = alpha  # shape
        self.beta = beta  # rate (inverse scale)
        self.arm_precision = 0 # variance for the posterior marginal on precision 
        self.n = 0  # number of samples
        self.x_avg = 0  # sample mean

        # for calculating \sum{(x_i - x_avg)^2} = \sum{x_i^2} - \sum{x_i}*2*x_avg + n*x_avg^2
        self.sum_x_i_sqr = 0
        self.sum_x = 0  # sample sum

        self.last_mu = mu  # distribution mean
        self.last_k = k  # related to variance of the normal part of the prior, likely in a 1/k way
        self.last_alpha = alpha  # shape
        self.last_beta = beta  # rate (inverse scale)
        self.last_arm_precision = 0
        self.last_n = 0  # number of samples
        self.last_x_avg = 0  # sample mean

        # for calculating \sum{(x_i - x_avg)^2} = \sum{x_i^2} - \sum{x_i}*2*x_avg + n*x_avg^2
        self.last_sum_x_i_sqr = 0
        self.last_sum_x = 0  # sample sum

    def update_posterior(self, x, y):
        # update all parameters per sample
        # @param y: reward output

        # for each reward, update sample count
        self.n += 1
        # update sample mean
        self.x_avg = ((self.n - 1) * self.x_avg + y) / self.n
        # update sample sum
        self.sum_x = self.sum_x + y
        self.sum_x_i_sqr = self.sum_x_i_sqr + y * y

        # update posterior parameters
        self.mu = (NGNormal.init_k * NGNormal.init_mu + self.n * self.x_avg) / (NGNormal.init_k + self.n)
        self.k = NGNormal.init_k + self.n
        self.alpha = self.alpha + self.n * 0.5
        self.beta = self.beta + 0.5 * (self.sum_x_i_sqr \
                                       - 2 * self.x_avg * self.sum_x \
                                       + self.n * self.x_avg * self.x_avg) \
                    + (self.n * NGNormal.init_k / (NGNormal.init_k + self.n)) * (self.x_avg - NGNormal.init_mu) * (self.x_avg - NGNormal.init_mu) * 0.5
        self.arm_precision = self.alpha/self.beta#self.beta * (self.v + 1) / (self.v * self.alpha)

    def draw_expected_value(self, x, num_samples=1):
        # draw a sample from normal innvgamma posterior which is
        # same as expected reward given this model.
        # TODO: check if it is actually the same with sampling from a student's t distribution

        if num_samples > 1:
            mu_tile = self.mu * np.ones(num_samples)
            precision_tile = []
            for i in range(num_samples):
                precision_tile[i] = gamma.rvs(self.alpha, scale = 1/self.beta)
            return np.random.normal(mu_tile, precision_tile)

        # first sample sigma^2 with inverse gamma
        precision = gamma.rvs(self.alpha, scale = 1/self.beta)
        # then sample x from normal with self.mean and variance= 1/(k*tau)
        return np.random.normal(self.mu, 1/(self.k*precision))

    def remove_from_model(self, x, y):
        """
            Just reverse update_posterior?
        """
        print("NOT IMPLEMENTED!!!")

    def write_parameters(self, out_row, action):
        out_row[H_ALGO_ESTIMATED_MU.format(action + 1)] = self.mu
        out_row[H_ALGO_ESTIMATED_V.format(action + 1)] = self.k
        out_row[H_ALGO_ESTIMATED_ALPHA.format(action + 1)] = self.alpha
        out_row[H_ALGO_ESTIMATED_BETA.format(action + 1)] = self.beta
        if self.arm_precision == 0:
            out_row[H_ALGO_ESTIMATED_ARM_VARIANCE.format(action + 1)] = float('nan')
        else:
            out_row[H_ALGO_ESTIMATED_ARM_VARIANCE.format(action + 1)] = 1/self.arm_precision

    def save_state(self):
        self.last_mu = self.mu  # distribution mean
        self.last_k = self.k  # related to variance
        self.last_alpha = self.alpha  # shape
        self.last_beta = self.beta  # rate
        self.last_arm_precision = self.arm_precision # posterior marginal on precision
        self.last_n = self.n  # number of samples
        self.last_x_avg = self.x_avg  # sample mean
        self.last_sum_x_i_sqr = self.sum_x_i_sqr
        self.last_sum_x = self.sum_x  # sample sum

    def restore_state(self):
        self.mu = self.last_mu
        self.k = self.last_k
        self.alpha = self.last_alpha
        self.beta = self.last_beta
        self.arm_precision = self.last_arm_precision
        self.n = self.last_n
        self.x_avg = self.last_x_avg
        self.sum_x_i_sqr = self.last_sum_x_i_sqr
        self.sum_x = self.last_sum_x

    def reset_state(self):
        self.mu = 0
        self.k = 1
        self.alpha = 1
        self.beta = 1
        self.arm_precision = 0

        self.n = 0
        self.x_avg = 0
        self.sum_x_i_sqr = 0
        self.sum_x = 0

        self.last_mu = self.mu
        self.last_k = self.k
        self.last_alpha = self.alpha
        self.last_beta = self.beta
        self.last_arm_precision = 0
        self.last_n = 0
        self.last_x_avg = 0

        self.last_sum_x_i_sqr = 0
        self.last_sum_x = 0
