"""
    Classify the iris dataset using a polynomial model with Gaussian basis functions.
    A single polynomial model is generated and each iris type is applied to the model.
    Much like a deep learning classification model, a value will be generated for each iris type.
    The highest value will be the predicted iris type.
"""

import torch
from sklearn import datasets

T = torch.tensor
T_ONES = torch.ones
STACK = torch.stack
CAT = torch.cat
LSTSQ = torch.linalg.lstsq
T_ZERO = T(0.)
T_ONE = T(1.)
T_TWO = T(2.)
T_MEAN = T([.5, ])
T_SIGMA = T([.5, ])
WHERE = torch.where
COUNT_NONZERO = torch.count_nonzero

iris = datasets.load_iris()

X = T(iris.data).float().T
y = T(iris.target)

# Scale X to have mean 0 and standard deviation 1
mean = X.mean(dim=1, keepdim=True)
std = X.std(dim=1, keepdim=True)
X = (X - mean) / std

A = CAT((
    # Axis intercepts for each feature combined into a single column
    STACK((T_ONES(X.size(dim=1)),)),
    # Linear term for each feature (4 columns)
    X,
    # First Gaussian basis function, centered with a mean of 0 and a standard deviation of 1 for each feature
    (-(X ** 2) / T_TWO).exp(),
))

# Add Gaussian basis functions for each feature.  The mean and standard deviation are specified in T_MEAN and T_SIGMA.
# This step reduces the error rate from 2.66% to 2% but could be overfitting the model. (8 columns)
for i in range(T_MEAN.size(dim=0)):
    A = CAT((
        A,
        # Gaussian basis function left of the mean for each feature (4 columns)
        (-((X - T_MEAN[i]) ** 2) / (T_TWO * T_SIGMA[i] ** 2)).exp(),
        # Gaussian basis function right of the mean for each feature (4 columns)
        (-((X + T_MEAN[i]) ** 2) / (T_TWO * T_SIGMA[i] ** 2)).exp()),
    )
A = A.T
print("polynomial shape:", A.shape)

Y = WHERE(y == T([[0], [1], [2]]), T_ONE, T_ZERO).T

# Solve for the polynomial coefficients using the Least Squares for each iris type (3 sets of coefficients)
P = LSTSQ(A, Y).solution

print("coefficient sums (larger numbers can suggest over fitting):", P.abs().sum(dim=0))

# Apply the polynomial model to the iris data
Z = A @ P

# Select the iris type with the highest value
max_value_indices = Z.max(dim=1).indices

# Calculate the error rate
error_count = COUNT_NONZERO(max_value_indices - y)
print(error_count.item() / y.size(0) * 100, 'percent errors,', error_count.item(), 'errors')

