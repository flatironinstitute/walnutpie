data {
  int<lower=0> D;
}
parameters {
  real log_sigma_sq;
  vector[D - 1] alpha;
}
transformed parameters {
  real<lower=0> sigma = exp(log_sigma_sq / 2);
}
model {
  log_sigma_sq ~ normal(0, 3);
  alpha ~ normal(0, sigma);
}