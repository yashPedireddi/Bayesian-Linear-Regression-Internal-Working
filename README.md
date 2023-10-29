# Bayesian-Linear-Regression-Internal-Working
“Bayesian linear regression” is a probabilistic perspective of the “Linear Regression Model.” In the
traditional linear regression model, we have a point estimate of the weight parameters and the predicted
value. In the probabilistic perspective of machine learning, we try to estimate a probability distribution
of the model’s weights and the predicted value instead of the point estimate. In this project, our focus
is to estimate probability distribution of the weights of our model.
A linear model with known weights (w0 and w1) will map the relation between simulated input features
and output values. The aim is to predict a distribution of weights close to the initial weights(w0 and
w1). We initially consider a “Prior” distribution and “Posterior” distribution of weights a “Gaussian”
to achieve this. Then, the data points will be fed sequentially to the ” Posterior Distribution equation.”
The mean and covariance of the posterior depend on the prior mean, prior covariance, and the simulated
data..
