Multilevel regression from survey data

24 Apr 2026

Data are from the 2018 Cooperative Congressional Election Study.

Background on the problem and link to original data are at MRP Case Studies:  https://bookdown.org/jl5522/MRP-case-studies/

data_all.csv:  Data from 59,660 survey respondents.  Respondents with missing data were excluded.

state_predictors.csv:  State-level predictors (standardized Republican vote in previous election, and an index for region of the country)

multilevel_regression.stan:  Stan program implementing a particular multilevel linear regression

multilevel_regression_logit.stan:  Stan program implementing a particular multilevel logistic regression

multilevel_regression.R:  R script that sets up the data and fits the mulilevel linear and logistic regressions in Stan and rstanarm.  The rstanarm versions are the same as the Stan models and are included just for convenience.

To make the computation run faster, we fit to a random subset of 1000 people.
