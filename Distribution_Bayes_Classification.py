import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

N = 10000

# %% gaussian distribution N(0, 1) and white noise
x = np.random.normal(loc=0, scale=1, size=(1, N))

# %% Even Distribution [a,b]=[0,1]
y = np.random.uniform(low=0.0, high=1.0, size=(1, N))

# %% Classification
# Gaussian probability density
pdf_x = norm.pdf(x, loc=0, scale=1)
# Uniform probability density
pdf_y = uniform.pdf(y, loc=0.0, scale=1.0)
# Confussion Matrix
CM = np.zeros((2, 2))
# First group check
for i in range(0, N):
    num_x = norm.pdf(x[0, i], loc=0, scale=1)
    num_y = uniform.pdf(x[0, i], loc=0.0, scale=1.0)
    if num_x >= num_y:
        CM[0, 0] += 1
    else:
        CM[0, 1] += 1
# Second group check
for i in range(0, N):
    num_x = norm.pdf(y[0, i], loc=0, scale=1)
    num_y = uniform.pdf(y[0, i], loc=0.0, scale=1.0)
    if num_x < num_y:
        CM[1, 1] += 1
    else:
        CM[1, 0] += 1
    # Error Calculation
Accuracy = (CM[0, 0] + CM[1, 1]) / np.sum(CM)
Error_Rate = 1 - Accuracy
Precision = CM[0, 0] / np.sum(CM, axis=1)[0]
Recall = CM[0, 0] / np.sum(CM, axis=0)[0]
Specifisity = CM[1, 1] / np.sum(CM, axis=0)[1]
print("Accuracy = ", Accuracy)
print("Error_Rate = ", Error_Rate)
print("Precision = ", Precision)
print("Recall = ", Recall)
print("Specifisity = ", Specifisity)
print(CM)
# %% Graph the distributions
t = np.linspace(-2., 2., N)
pdf_x = norm.pdf(t, loc=0, scale=1)
pdf_y = uniform.pdf(t, loc=0.0, scale=1.0)
plt.plot(t, pdf_x)
plt.plot(t, pdf_y)
plt.show()
