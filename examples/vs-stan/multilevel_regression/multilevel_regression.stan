data {
  int<lower=0> N, N_male, N_state, N_region, N_age, N_eth, N_educ;
  array[N] real y;
  array[N] int state, age, eth, educ, male;
  array[N_state] int region;
  array[N_state] real repvote;
}
parameters {
  real b_0, b_male, b_repvote;
  real<lower=0> s_y;
  real<lower=0> s_state, s_region, s_age, s_eth, s_educ, s_male_eth, s_educ_age, s_educ_eth;
  array[N_state] real<multiplier=s_state> a_state;
  array[N_region] real<multiplier=s_region> a_region;
  array[N_age] real<multiplier=s_age> a_age;
  array[N_eth] real<multiplier=s_eth> a_eth;
  array[N_educ] real<multiplier=s_educ> a_educ;
  array[N_male,N_eth] real<multiplier=s_male_eth> a_male_eth;
  array[N_educ,N_age] real<multiplier=s_educ_age> a_educ_age;
  array[N_educ,N_eth] real<multiplier=s_educ_eth> a_educ_eth;
}
model {
  array[N] real linpred;
  a_state ~ normal(0, s_state);
  a_region ~ normal(0, s_region);
  a_age ~ normal(0, s_age);
  a_eth ~ normal(0, s_eth);
  a_educ ~ normal(0, s_educ);
  to_array_1d(a_male_eth) ~ normal(0, s_male_eth);
  to_array_1d(a_educ_age) ~ normal(0, s_educ_age);
  to_array_1d(a_educ_eth) ~ normal(0, s_educ_eth);
  for (n in 1:N){
    linpred[n] = b_0 + b_male*male[n] + b_repvote*repvote[state[n]] +
      a_state[state[n]] + a_region[region[state[n]]] + a_age[age[n]] + a_eth[eth[n]] + a_educ[educ[n]] +
      a_male_eth[male[n]+1,eth[n]] + a_educ_age[educ[n],age[n]] + a_educ_eth[educ[n],eth[n]];
  }
  y ~ normal(linpred, s_y);
}

