/**
 * Model from Eric Potash, 2026
 * For brms version, see:  https://github.com/potash/spectroscopy-lmm
 * Downloaded from: https://discourse.mc-stan.org/t/open-data-and-brms-code-for-soil-carbon-spectroscopy-prediction/41667/8
 */
data {
  int I; // number of observations
  vector[I] y; // outcomes
  
  int K; // number of predictors
  matrix[I, K] X; // spectroscopy predictors
  
  int L; // number of group-level predictors
  matrix[I, L] Z; // group-level predictors
  matrix[I, L*K] XZ; // group-level interactions
  
  int N; // number of groupings
  array[N] int<lower=1> M_n; // number of groups M_n in grouping n
  array[N, I] int m_ni; // m_ni[n, i] is the group number in grouping n of observation i
  
  // prediction set
  int I_pred;
  matrix[I_pred, K] X_pred;
  matrix[I_pred, L] Z_pred;
  matrix[I_pred, L*K] XZ_pred; // group-level interactions
  array[N, I_pred] int m_ni_pred;
}
transformed data {
  // total number of groups across groupings
  // used for ragged arrays (vectors) of random intercepts and slopes
  int sum_M_n = sum(M_n); 
}
parameters {
  real alpha1; // intercept
  vector[K] alpha_raw; // slopes
  
  real<lower=0> sigma_alpha; // slope hyperprior
  real<lower=0> sigma; // residual SD
  
  vector[L] gamma1; // group-level slopes
  vector[L*K] gamma_raw; // interaction slopes (effect of group-level predictors on main slopes)
  real<lower=0> sigma_gamma;
  
  // random intercepts
  vector[sum_M_n] nu1_raw;
  vector<lower=0>[N] sigma_n;
  
  // random slopes
  matrix[sum_M_n, K] nu_raw;
  vector<lower=0>[N] theta_n;
}
// we use non-centered parameters above, transform to centered
transformed parameters {
  vector[K] alpha = alpha_raw * sigma_alpha;
  vector[L*K] gamma = gamma_raw * sigma_gamma;
  vector[sum_M_n] nu1;
  matrix[sum_M_n, K] nu;
  
  {
    int pos = 1;
    for(n in 1:N) {
      nu1[pos:(pos + M_n[n] - 1)] = segment(nu1_raw, pos, M_n[n]) * sigma_n[n];
      nu[pos:(pos + M_n[n] - 1),:] = nu_raw[pos:(pos + M_n[n] - 1),:] * theta_n[n];
      
      pos = pos + M_n[n];
    }
  }
}
model {
  int pos = 1;
  vector[I] intercepts = rep_vector(0, I);
  matrix[I,K] slopes = rep_matrix(to_row_vector(alpha), I);
  for(n in 1:N) {
    vector[M_n[n]] nu1_n = segment(nu1, pos, M_n[n]);
    intercepts = intercepts + nu1_n[m_ni[n]];

    matrix[M_n[n], K] nu_n = nu[pos:(pos+M_n[n] - 1)];
    slopes = slopes + nu_n[m_ni[n]];

    pos = pos + M_n[n];
  }
  y ~ normal(alpha1 + intercepts + rows_dot_product(X, slopes) + Z*gamma1 + XZ*gamma, sigma);
  
  // priors
  alpha1 ~ normal(0, 2.5);
  alpha_raw ~ std_normal();
  sigma ~ exponential(1);
  gamma1 ~ std_normal();
  gamma_raw ~ std_normal();
  
  sigma_alpha ~ std_normal();
  sigma_gamma ~ std_normal();
  
  sigma_n ~ std_normal();
  theta_n ~ std_normal();
  
  nu1_raw ~ std_normal();
  to_vector(nu_raw) ~ std_normal();
}

generated quantities {
  array[I_pred] real y_pred;
  {
    int pos = 1;
    vector[I_pred] intercepts_pred = rep_vector(0, I_pred);
    matrix[I_pred,K] slopes_pred = rep_matrix(to_row_vector(alpha), I_pred);
    for(n in 1:N) {
      vector[M_n[n]] nu1_n = segment(nu1, pos, M_n[n]);
      intercepts_pred = intercepts_pred + nu1_n[m_ni_pred[n]];
      
      matrix[M_n[n], K] nu_n = nu[pos:(pos+M_n[n] - 1)];
      slopes_pred = slopes_pred + nu_n[m_ni_pred[n]];
    
      pos = pos + M_n[n];
    }
    y_pred = normal_rng(alpha1 + intercepts_pred + rows_dot_product(X_pred, slopes_pred) + Z_pred*gamma1 + XZ_pred*gamma, sigma);
  }
}
