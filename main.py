import numpy as np


class BlackBoxRegression:

    def __init__(self, lam=0.0):
        self.lam = lam
        self.std_coef = None
        self.x_mean = None
        self.x_std = None

    def fit(self, x, y):
        x = self._standardize(x)
        y_bar = np.mean(y)
        x_beta = (np.dot(x, y) / (np.dot(x, x) + self.lam))
        self.std_coef = np.array([y_bar, x_beta])
        return self

    def _standardize(self, x):
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        return (x - self.x_mean) / self.x_std

    def predict(self, x):
        x = (x - self.x_mean) / self.x_std
        return self.std_coef[0] + self.std_coef[1] * x


def generate_data(b=0, m=1, N=100, sd=0.6):
    x = np.random.normal(0.0, 1.0, size=N)
    x = x / (np.sqrt(500) * np.std(x))
    y = (m * x + b
         + np.random.normal(0.0, sd, size=N)
         + (np.random.choice([0, 1], size=N, replace=True, p=[0.85, 0.15]) *
            np.random.normal(0.0, 2 * sd, size=N))
         + (np.random.choice([0, 1], size=N, replace=True, p=[0.85, 0.15]) *
            np.random.normal(0.0, 2 * sd, size=N)))
    return x, y


def plot_predictions(ax, bbm, **kwargs):
    y_hat = bbm.predict(t)
    ax.plot(t, y_hat, **kwargs)


def add_random_noise(x, sd=0.5):
    return x * np.random.normal(1.0, sd, size=len(x))


def fit_bbm_data(x, y, sd=0.25):
    x_rand = add_random_noise(x, sd=sd)
    r = BlackBoxRegression(lam=0.0).fit(x_rand, y)
    return r