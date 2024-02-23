import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

x = np.arange(1, 10)

plt.figure(figsize=(13,8))
plt.plot(x, poisson.pmf(x, 0.6), 'ko-', ms=8, label='Poisson mu = 0.6')
plt.plot(x, poisson.pmf(x, 0.3), 'bo-', ms=8, label='Poisson mu = 0.3')
plt.plot(x, poisson.pmf(x, 1.2), 'ro-', ms=8, label='Poisson mu = 1.2')
plt.xlabel('x')
plt.xlabel('PMF')
plt.legend()
plt.show()

plt.figure(figsize=(13,8))
plt.plot(x, poisson.cdf(x, 0.6), 'ko-', ms=8, label='Poisson mu = 0.6')
plt.plot(x, poisson.cdf(x, 0.3), 'bo-', ms=8, label='Poisson mu = 0.3')
plt.plot(x, poisson.cdf(x, 1.2), 'ro-', ms=8, label='Poisson mu = 1.2')
plt.xlabel('x')
plt.xlabel('CDF')
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
y = poisson.rvs(0.6, size=10000)
plt.hist(y)
plt.show()
