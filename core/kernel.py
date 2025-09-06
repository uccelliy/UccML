import numpy as np


# K(x1,x2) = x1^T * x2
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# K(x1,x2) = (beta + gamma * x1^T * x2)^d
def ploy_kernel(x1, x2, d=3):
    return (1 + np.dot(x1, x2)) ** d

# K(x1,x2) = exp(-gamma * ||x1-x2||^2)
#          = sum(exp(-x1^2)sqrt(2^n/n!)x1^n * exp(-x2^2)sqrt(2^n/n!)x2^n)
def rbf_kernel(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# K(x1,x2) = tanh(alpha * x1^T * x2 + beta)
def sigmoid_kernel(x1, x2):
    return np.tanh(np.dot(x1, x2) + 1)

def chi2_kernel(x1, x2):
    return np.exp(-np.sum((x1 - x2) ** 2 / (x1 + x2 + 1e-9)))

