import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *

plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing input data
data = pd.read_csv('C:/Users/gante/Documents/KAMPUS/AI - KECERDASAN BUATAN/no1.csv', ';')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X))  # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

print(m, c, '\n')

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

urut = data.sort_values(['Experience'])

print(data, '\n')

print('\tThe data is sorted by Experience level')
print(urut, '\n')

print('\t\tPredicted')
print(Y_pred, '\n')

print('The Mean Square Error is:')
print(mean_squared_error(Y, Y_pred), '\n')

sal = urut['Salary'].values
print('The R² score for those model is:')
print(r2_score(sal, Y_pred))
