import numpy as np
from scipy.special import erf

class normal_distribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x):
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sigma ** 2))

    def cdf(self, x):
        return 0.5 * (1 + erf((x - self.mu) / (self.sigma * np.sqrt(2))))

    def sample(self, n):
        return np.random.normal(self.mu, self.sigma, n)
    
class uniform_distribution:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        return 1 / (self.b - self.a)

    def cdf(self, x):
        return (x - self.a) / (self.b - self.a)

    def sample(self, n):
        return np.random.uniform(self.a, self.b, n)