import numpy as np


class Simulator(object):

    def __init__(self, num_arms, arm_means):
        """
        Arguments:
            num_arms: total number of arms
            arm_means: vector of arm means
        """
        self.n = num_arms
        self.means = arm_means
        self.num_pulls = np.zeros(num_arms)

    def pull(self, arm):
        score = np.random.binomial(1, self.means[arm])
        self.num_pulls[arm] += 1
        return score

    def num(self):
        return self.n

    def pull_num_array(self):
        return self.num_pulls

    def pull_num(self, arm):
        return self.num_pulls[arm]

    def total_pull_num(self):
        return np.sum(self.num_pulls)


class Simulator(object):

    def __init__(self, num_arms, arm_means):
        """
        Arguments:
            num_arms: total number of arms
            arm_means: vector of arm means
        """
        self.n = num_arms
        self.means = arm_means
        self.num_pulls = np.zeros(num_arms)
        self.condition=0

    def pull(self, arm):
        score = np.random.binomial(1, self.means[arm])
        self.num_pulls[arm] += 1
        return score

    def num(self):
        return self.n

    def pull_num_array(self):
        return self.num_pulls

    def pull_num(self, arm):
        return self.num_pulls[arm]

    def total_pull_num(self):
        return np.sum(self.num_pulls)
