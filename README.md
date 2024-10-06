# ml-prediction

The goal of this project is to implement **regression models** to predict the quality of white wines based on various features. The dataset provides a range of variables that describe the properties of each wine, and the task is to develop models that use these variables to predict the wine quality, which is scored from 0 to 10 based on sensory evaluations by three experts.

## Regression Models

The following regression techniques were implemented to predict wine quality:
- **Linear Regression**
- **Ridge Regression**
- **k-Nearest Neighbors (kNN) Regression**

The models were trained using a combination of the wine features provided in the dataset. The goal was to explore the relationship between these features and the quality scores, and identify the best-performing model.

## Formulas Used

### Linear Regression

The normal equation used to calculate the coefficients in linear regression is as follows:

$$ \mathbf{X}^{\top}\mathbf{X} \hat{\mathbf{W}} = \mathbf{X}^{\top}\mathbf{Y} $$

Here $\mathbf{X}$  is the matrix of feature values.
and $\mathbf{Y}$ is the vector of target values, while $\hat{\mathbf{W}}$ is the weights/coefficients of the linear regression.

The **Mean Squared Error(MSE)** is calculated using the formula,

$$ \mathrm{MSE} = \frac{1}{2s} \left\|\mathbf{X}\mathbf{W} - \mathbf{Y} \right\|^2, $$

where $\mathbf{X}$  is the matrix of feature values.
and $\mathbf{Y}$ is the vector of target values, while $\hat{\mathbf{W}}$ is the weights/coefficients of the linear regression.

### Ridge regression

Ridge Regression is a linear regression variant that introduces regularization to prevent overfitting. The objective function for ridge regression is given by ,

$$ W = \frac{1}{2s} \left\| Xw - y \right\|^2 + \frac{\alpha}{2} \left\| w \right\|^2 $$

### KNN Regression

In k-NN regression, the prediction for a new data point is determined by averaging the target values of its k nearest neighbours. For each test point, the Euclidean distance to all training points is calculated by,

$$ d(x_1, x_2) = \sqrt{\sum_{i=1}^{n} (x_{1i} - x_{2i})^2} $$


The effectiveness of each model is evaluated based on error metrics such as **Mean Squared Error (MSE)** and **R-Squared**.
