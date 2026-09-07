data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x_star;
  real<lower=0> sigma_x_star;
}
parameters {
  real a, b, mu_x;
  real<lower=0> sigma, sigma_x;
  vector[N] x;
}
model {
  x ~ normal(mu_x, sigma_x);
  y ~ normal(a + b*x, sigma);
  x_star ~ normal(x, sigma_x_star);
}
