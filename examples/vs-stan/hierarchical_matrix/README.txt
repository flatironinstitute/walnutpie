Hierarchical regression model with varying intercepts and slopes

6 May 2026

Data are simulated.

This is a regression problem wth 4 predictors, 10 groups, and coefficients that vary by group; thus, we are estimating a 10 x 4 matrix of coefficients, along with a sd of the data model, a mean vector of length 4, and a 4 x 4 covariance matrix for the distributon of coefficients.

This model is challenging because in Stan it can't be encapsulated.  We'd like to express it as a two-liner, something like this:
y ~ normal(X*b[group], s_y);
b ~ multi_normal(mu_b, Sigma_b);
But we can't do this in Stan because the declarations and transformations required to set up an efficient computation are all over the place.

lkj_rng.stan:  Utility program used to simulate correlation matrices from the LKJ prior.

hierarchical_matrix_1.stan:  Stan program implementing a multivariate normal model (with flat prior on the means, weakly informative exponential(1) priors on the scale parameters, and LKJ(2) prior on the correlation matrix) for K coefficients that vary by group.

hierarchical_matrix_2.stan:  Same model using the more efficient Cholesky parameterization.

hierarchical_matrix.R:  R script that simulates the data and fits the models.  The data structure is J=20 groups, with 1, 2, ..., 20 observations per group, thus N=210 observations in total.  Data are simulated from a regression model with K=4 predictors:  a constant term, a pre-treatment predictor x, a treatment indicator z, and their interaction, with x and z simulated in a way that is correlated with the group ID.

