import numpy

class MackeyGlass:
    def __init__(self, tau, beta, gamma, n, x0, numPoints):
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.x0 = x0
        self.numPoints = numPoints

    def generate(self):
        x = numpy.zeros(self.numPoints)
        x[0] = self.x0
        for i in range(0, self.numPoints - 1):
            x[i + 1] = self.gamma * x[i] + (self.beta * x[i - self.tau]) / (1 + x[i - self.tau] ** self.n) - 0.1 * x[i]

            # equation comes from
            # https://www.mathworks.com/help/ident/ref/mackeyglass.html
            # https://www.scholarpedia.org/article/Mackey-Glass_equation
            # https://en.wikipedia.org/wiki/Mackey-Glass_equations

        return x
    
    def generate_noisy(self, sigma):
        x = self.generate()
        return x + numpy.random.normal(0, sigma, self.numPoints)
    

# class PoissonBoltzmann:
#     def __init__(self, alpha):
#         self.alpha = alpha

#     def generate(self, x):
#         # x is a numpy array