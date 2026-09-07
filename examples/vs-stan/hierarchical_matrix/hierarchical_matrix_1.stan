data {
  int J, K, N;
  vector[N] y;
  array[N] int<lower=1, upper=J> group;
  matrix[N,K] X;
}
parameters {
  real<lower=0> s_y;
  array[J] vector[K] b;
  vector[K] mu_b;
  vector<lower=0>[K] s_b;
  corr_matrix[K] Omega_b;
}
model {
  s_b ~ exponential(1);
  Omega_b ~ lkj_corr(2);
  {
    matrix[K,K] Sigma_b = quad_form_diag(Omega_b, s_b);
    b ~ multi_normal(mu_b, Sigma_b);
  }
  {
    vector[N] y_hat;
    for (n in 1:N) {
      y_hat[n] = X[n,]*b[group[n],];
    }
    y ~ normal(y_hat, s_y);
  }
}

