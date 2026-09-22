data {
  int<lower=0> D;
}
transformed data {
  vector<lower=0>[D] sigma = linspaced_vector(D, 1, D);
}
parameters {
  vector[D] alpha;
}
model {
  alpha ~ normal(0, sigma);
}
