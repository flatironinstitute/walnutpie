data {
  int<lower=0> D;
}
parameters {
  vector[D] alpha;
}
model {
  alpha ~ std_normal();
}
