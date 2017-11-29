import numpy as np
import scipy
from scipy.stats import invgamma
from base_model import *
from output_format import *


class NIGNormal(BaseModel):
    '''
    Normal-inverse-Gamma_Normal model for Thompson Sampling.
    This model does not consider the context.
    '''

    init_mu = 0
    init_v = 1
    init_alpha = 1
    init_beta = 1

    def __init__(self, mu, v, alpha, beta):
        self.mu = mu  # distribution mean
        self.v = v  # variance
        self.alpha = alpha  # shape
        self.beta = beta  # scale
        self.arm_variance = 0 # variance for the posterior predictive student's T distro
        self.n = 0  # number of samples
        self.x_avg = 0  # sample mean

        # for calculating \sum{(x_i - x_avg)^2} = \sum{x_i^2} - \sum{x_i}*2*x_avg + n*x_avg^2
        self.sum_x_i_sqr = 0
        self.sum_x = 0  # sample sum

        self.last_mu = mu  # distribution mean
        self.last_v = v  # variance
        self.last_alpha = alpha  # shape
        self.last_beta = beta  # scale
        self.last_arm_variance = 0
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
        mu_0 = NIGNormal.init_mu
        v_0 = NIGNormal.init_v
        self.mu = (v_0 * mu_0 + self.n * self.x_avg) / (v_0 + self.n)
        self.v = v_0 + self.n
        self.alpha = NIGNormal.init_alpha + self.n * 0.5
        self.beta = NIGNormal.init_beta + 0.5 * (self.sum_x_i_sqr \
                                       - 2 * self.x_avg * self.sum_x \
                                       + self.n * self.x_avg * self.x_avg) \
                    + (self.n * v_0 / (v_0 + self.n)) * (self.x_avg - mu_0) * (self.x_avg - mu_0) * 0.5
        self.arm_variance = self.beta * (self.v + 1) / (self.v * self.alpha)

    def draw_expected_value(self, x, num_samples=1):
        # draw a sample from normal innvgamma posterior which is
        # same as expected reward given this model.
        # TODO: check if it is actually the same with sampling from a student's t distribution

        if num_samples > 1:
            mu_tile = self.mu * np.ones(num_samples)
            sigma_sqr_tile = []
            for i in range(num_samples):
                sigma_sqr_tile[i] = invgamma.rvs(self.alpha, scale=self.beta)
            return np.random.normal(mu_tile, sigma_sqr_tile)

        # first sample sigma^2 with inverse gamma
        sigma_sqr = invgamma.rvs(self.alpha, scale=self.beta)
        # then sample x from normal with self.mean and sigma^2/nu
        return np.random.normal(self.mu, sigma_sqr/self.v)

    def remove_from_model(self, x, y):
        """
            Just reverse update_posterior?
        """
        print("NOT IMPLEMENTED!!!")

    def write_parameters(self, out_row, action):
        out_row[H_ALGO_ESTIMATED_MU.format(action + 1)] = self.mu
        out_row[H_ALGO_ESTIMATED_V.format(action + 1)] = self.v
        out_row[H_ALGO_ESTIMATED_ALPHA.format(action + 1)] = self.alpha
        out_row[H_ALGO_ESTIMATED_BETA.format(action + 1)] = self.beta
        out_row[H_ALGO_ESTIMATED_ARM_VARIANCE.format(action + 1)] = self.arm_variance

    def save_state(self):
        self.last_mu = self.mu  # distribution mean
        self.last_v = self.v  # variance
        self.last_alpha = self.alpha  # shape
        self.last_beta = self.beta  # scale
        self.last_arm_variance = self.arm_variance # predictive posterior for student's T distro
        self.last_n = self.n  # number of samples
        self.last_x_avg = self.x_avg  # sample mean
        self.last_sum_x_i_sqr = self.sum_x_i_sqr
        self.last_sum_x = self.sum_x  # sample sum

    def restore_state(self):
        self.mu = self.last_mu
        self.v = self.last_v
        self.alpha = self.last_alpha
        self.beta = self.last_beta
        self.arm_variance = self.last_arm_variance
        self.n = self.last_n
        self.x_avg = self.last_x_avg
        self.sum_x_i_sqr = self.last_sum_x_i_sqr
        self.sum_x = self.last_sum_x

    def reset_state(self):
        self.mu = 0
        self.v = 1
        self.alpha = 1
        self.beta = 1
        self.arm_variance = 0

        self.n = 0
        self.x_avg = 0
        self.sum_x_i_sqr = 0
        self.sum_x = 0

        self.last_mu = self.mu
        self.last_v = self.v
        self.last_alpha = self.alpha
        self.last_beta = self.beta
        self.last_arm_variance = 0
        self.last_n = 0
        self.last_x_avg = 0

        self.last_sum_x_i_sqr = 0
        self.last_sum_x = 0
