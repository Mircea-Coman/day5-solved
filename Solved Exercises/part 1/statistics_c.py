import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, logistic, uniform, ttest_ind

size = 10000000
y_1 = norm.rvs(size=size)
y_2 = norm.rvs(size=size)

# y_2 = uniform.rvs(size=size)

print(ttest_ind(y_1, y_2, equal_var = False))

plt.figure(figsize=(13,8))
plt.hist(y_1, color = 'r', alpha = 0.5, density = True)
plt.hist(y_2, color = 'b', alpha = 0.5, density = True)
plt.show()
