data {
  int J, K, N;
  vector[N] y;
  array[N] int<lower=1, upper=J> group;
  matrix[N,K] X;
}
parameters {
  real<lower=0> s_y;
  matrix[K,J] z_b;
  vector[K] mu_b;
  vector<lower=0>[K] s_b;
  cholesky_factor_corr[K] L_b;
}
transformed parameters {
  matrix[K,J] b = rep_matrix(mu_b, J) + diag_pre_multiply(s_b, L_b) * z_b;
  matrix[K,K] Omega_b = L_b * L_b';
}
model {
  s_b ~ exponential(1);
  L_b ~ lkj_corr_cholesky(2);
  to_vector(z_b) ~ normal(0, 1);
  {
    vector[N] y_hat;
    for (n in 1:N) {
      y_hat[n] = X[n,]*b[,group[n]];
    }
    y ~ normal(y_hat, s_y);
  }
}

