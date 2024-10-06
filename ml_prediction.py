
# ## Wine quality prediction: White Wine dataset
# 
# The task is to implement, describe and present regression and/or
# classification models to predict the quality of white wines given a range of
# their features. The goal is to develop regression and/or classification models using any
# number of the variables provided, which describe wines’ features, to predict their quality
# (measured as a score from 0 to 10 based on sensory data from three experts).

# **Importing Packages**

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('winequality-white.csv',sep=';')
df

# Explore the distribution of the target variable 'quality'
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Examine summary statistics for each feature
summary_stats = df.describe()

# Visualize summary statistics using a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, palette='viridis')
plt.title('Boxplot of Features in the Dataset')
plt.xticks(rotation=45, ha='right')
plt.show()

# Display the summary statistics
print("Summary Statistics for Each Feature:")
print(summary_stats)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)


# Visualize outliers in the target variable 'quality'
plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', data=df, palette='viridis')
plt.title('Boxplot of Wine Quality')
plt.xlabel('Quality')
plt.show()


# Visualize the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Extract the correlation values for the target variable, 'quality'
correlation_with_quality = correlation_matrix['quality'].sort_values(ascending=False)

# Printing the correlations
print("Correlation with Quality:")
print(correlation_with_quality)

# Visualize the top correlations using a bar plot
plt.figure(figsize=(10, 6))
correlation_with_quality.drop('quality').plot(kind='bar', color='skyblue')
plt.title('Correlation of Features with Wine Quality')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.show()


# #### Splitting data into test and train set

# Setting y as target variable, 'quality'
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Setting the random seed for reproducibility
np.random.seed(42)

# Define the proportion for the test set
test_size = 0.2

# Determine the number of samples for the test set
num_test_samples = int(test_size * len(X))

# Randomly shuffle indices to create a random split
indices = np.arange(len(X))
np.random.shuffle(indices)

# Splitting the indices into training and testing sets
train_indices, test_indices = indices[num_test_samples:], indices[:num_test_samples]

# Create the training and testing sets
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Displaying the shape of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# ## Linear Regression


# Adding a column of ones to X_train for the intercept term
X_train_lr = np.column_stack((np.ones(len(X_train)), X_train))

# Use the normal equation to calculate the coefficients
W = np.linalg.inv(X_train_lr.T @ X_train_lr) @ X_train_lr.T @ y_train

# Extract the intercept and coefficients
intercept = W[0]
coefficients = W[1:]

# Display the coefficients
print("Intercept:", intercept)
print("Coefficients:", coefficients)


# Adding a column of ones to X_test for the intercept term
X_test_lr = np.column_stack((np.ones(len(X_test)), X_test))

# Make predictions using the linear regression coefficients
y_test_pred = X_test_lr @ W

# Display the predicted values
print("Predicted values on the test data:")
print(y_test_pred)


# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_test_pred)**2)

# Display MSE
print("Mean Squared Error (MSE) on the test data:", mse)


# #### Scatter plot to calculate the residual

# Calculate the residuals (difference between actual and predicted values)
residuals = y_test - y_test_pred

# Plotting the residuals
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residuals')
plt.title('Residuals Plot for Linear Regression')
plt.xlabel('Data Points')
plt.ylabel('Residuals')
plt.legend()
plt.show()


# ## Ridge regression


class RidgeRegression:
    def __init__(self, alpha=0.01, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y):
        # Standardize features
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean) / self.X_std

        n_samples, n_features = X.shape
        X = np.column_stack((np.ones(n_samples), X))  # Add a column of ones for intercept
        self.coef_ = np.zeros(n_features + 1)  # Include intercept term

        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Exclude regularization for the intercept term

        for iteration in range(self.max_iter):
            prev_coef = np.copy(self.coef_)
            y_pred = X @ self.coef_
            residuals = y - y_pred
            gradient = -2 * X.T @ residuals / n_samples
            ridge_penalty = 2 * self.alpha * self.coef_
            gradient += ridge_penalty

            # Add a small positive value to the diagonal for stability
            Hessian = 2 * X.T @ X / n_samples + self.alpha * I

            self.coef_ -= np.linalg.inv(Hessian) @ gradient

            if np.linalg.norm(self.coef_ - prev_coef, ord=2) < self.tol:
                break

        return self

    def predict(self, X):
        # Standardize features using the mean and std from the training set
        X = (X - self.X_mean) / self.X_std
        n_samples = X.shape[0]
        X = np.column_stack((np.ones(n_samples), X))  # Adding a column of ones for intercept
        return X @ self.coef_


# Set hyperparameters
alpha = 0.01
max_iter = 1000

# Instantiate and train Ridge Regression
ridge_regressor = RidgeRegression(alpha=alpha, max_iter=max_iter)
X_train_rid = X_train
Y_train_rid = y_train
ridge_regressor.fit(X_train_rid, Y_train_rid)


# Make predictions
y_test_pred = ridge_regressor.predict(X_test)

# Display the predicted values
print("Predicted values on the test data:")
print(y_test_pred)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_test_pred)**2)

# Display MSE
print("Mean Squared Error (MSE) on the test data:", mse)


# ## KNN Regression

# Function to calculate euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Function to predict the values of target variable
def knn_regression_predict(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_targets = y_train[k_nearest_indices]

        # Simple average for regression
        prediction = np.mean(k_nearest_targets)
        predictions.append(prediction)

    return np.array(predictions)

# Set the value of k (number of neighbors)
k_value = 5

# Make predictions on the test set
y_test_pred = knn_regression_predict(X_train, y_train, X_test, k_value)

# Display the predicted values
print("Predicted values on the test data:")
print(y_test_pred)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_test_pred)**2)

# Display MSE
print("Mean Squared Error (MSE) on the test data:", mse)


# ## Using Sci-kit learn

# ### Linear Regression using Sci-kit learn


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Drop the target variable ('quality')
X = df.drop('quality', axis=1)
y = df['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate Linear Regression model
linear_regressor = LinearRegression()

# Fit the model to the training data
linear_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = linear_regressor.predict(X_test)

# Display the predicted values
print("Predicted values on the test data:")
print(y_test_pred)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")


# ### Ridge regression using sci-kit learn

# In[18]:


from sklearn.linear_model import Ridge

# Set hyperparameters
alpha = 0.01

# Instantiate and train Ridge Regression
ridge_regressor = Ridge(alpha=alpha)
ridge_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = ridge_regressor.predict(X_test)

# Display the predicted values
print("Predicted values on the test set:")
print(y_test_pred)

# Clculating Mean Squared Error
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error (MSE) on the test set: {mse}")


# ### k-NN Regression using Sci-kit learn

from sklearn.neighbors import KNeighborsRegressor

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the KNN Regression model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Fit the model on the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = knn_model.predict(X_test)

# Display the predicted values
print("Predicted values on the test data:")
print(y_test_pred)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error: {mse}")
