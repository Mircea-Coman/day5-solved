import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, num = 1000)

plt.figure(figsize=(13,8))
plt.plot(x, norm.pdf(x), 'ko-', ms=8, label='Normal PMF')
plt.xlabel('x')
plt.xlabel('PMF')
plt.legend()
plt.show()

plt.figure(figsize=(13,8))
plt.plot(x, norm.cdf(x), 'ko-', ms=8, label='Normal CDF')
plt.xlabel('x')
plt.xlabel('CDF')
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
y = norm.rvs(size=10000)
plt.hist(y, density = True)
plt.plot(x, norm.pdf(x), 'r-')
plt.show()
