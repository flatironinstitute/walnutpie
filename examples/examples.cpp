#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include <walnutpie.hpp>
#include "handlers.hpp"

static double total_time = 0.0;
static std::size_t count = 0;

static std::size_t DIM = 100;
static std::size_t WARMUP = 1000;
static std::size_t SAMPLE = 10000;

// p(y) = normal(y | 0, I)
static void std_normal(const Eigen::VectorXd& x, double& logp,
                       Eigen::VectorXd& grad) {
  logp = -0.5 * x.dot(x);
  grad = -x;
}

// p(y) = normal(0, diag(sigma)), sigma[d] = d
static void ill_normal(const Eigen::VectorXd& x, double& logp,
                       Eigen::VectorXd& grad) {
  const auto D = x.size();
  grad = Eigen::VectorXd::Zero(D);
  logp = 0;
  for (auto d = 0; d < D; ++d) {
    double sigma = d + 1;
    double sigma_sq = sigma * sigma;
    logp += -0.5 * x[d] * x[d] / sigma_sq;
    grad[d] = -x[d] / sigma_sq;
  }
}

// p(y) = normal(y | 0, Sigma), with Sigma[i, j] = rho^abs(i - j)
__attribute__((noinline)) static void rw1(const Eigen::VectorXd& y, double& logp, Eigen::VectorXd& grad) {
  double rho = 0.99;
  Eigen::Index D = y.size();
  double sigma_sq = 1.0 - rho * rho;
  double inv_sigma_sq = 1.0 / sigma_sq;
  grad.setZero(D);
  logp = -0.5 * y[0] * y[0];
  grad[0] -= y[0];
  for (Eigen::Index n = 1; n < D; ++n) {
    double r = y[n] - rho * y[n - 1];
    double w = r * inv_sigma_sq;
    logp -= 0.5 * r * w;
    grad[n] -= w;
    grad[n - 1] += rho * w;
  }
}

static void summarize(const Eigen::MatrixXd& draws) {
  auto D = draws.cols();
  auto N = draws.rows();
  for (auto d = 0; d < D; ++d) {
    if (d >= 3 && d < D - 3) {
      if (d == 4) {
        std::cout << "... elided " << (D - 6) << " rows ..." << std::endl;
      }
      continue;
    }
    auto mean = draws.col(d).mean();
    auto var = (draws.col(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << "dim " << d << ": mean = " << mean << ", stddev = " << stddev
              << "\n";
  }
}

static void summarize(const std::vector<Eigen::VectorXd>& draws) {
  Eigen::MatrixXd draws_mat(draws.size(), draws[0].size());
  for (std::size_t n = 0; n < draws.size(); ++n) {
    draws_mat.row(n) = draws[n].transpose();
  }
  return summarize(draws_mat);
}

template <typename F>
static void run_adaptive_walnuts(F& target_logp_grad) {
  std::cout << "\nRun adaptive Walnuts" << std::endl;

  unsigned int seed = 876254;
  std::mt19937 rng(seed);

  std::size_t num_chains = 1;
  std::size_t D = DIM;

  auto init_cfg = walnutpie::InitConfigBuilder(num_chains, D)
                      .positions(rng, 1.0)
                      .masses(target_logp_grad, 0.01)
                      .build();

  auto warmup_cfg = walnutpie::WarmupConfigBuilder()
                        .min_max_iter(WARMUP, WARMUP)
                        .mass_converge_tol(1.0)
                        .step_size_converge_tol(0.1)
                        .mass_init_count(4.0)
                        .step_accept_rate_target(0.8)
                        .step_learning_rate(0.05)
                        .step_gradient_decay(0.8)
                        .step_sq_gradient_decay(0.9)
                        .step_stabilization(1e-4)
                        .step_learn_rate_decay(0.95)
                        .build();

  auto sampling_cfg = walnutpie::SamplingConfigBuilder()
                          .min_max_iter(SAMPLE, SAMPLE)
                          .min_micro_steps(1)
                          .max_trajectory_doublings(8)
                          .max_step_halvings(5)
                          .rhat_converge_tol(1.001)
                          .build();

  std::cout << "Initialization configuration:\n" << init_cfg << std::endl;
  std::cout << "Warmup configuration:\n" << warmup_cfg << std::endl;
  std::cout << "Sampling configuration:\n" << sampling_cfg << std::endl;

  ChainStore handler;
  walnutpie::AdaptiveWalnuts adapt(rng, handler, target_logp_grad,
                                   init_cfg.init_chain_config(0), warmup_cfg,
                                   sampling_cfg);

  for (std::size_t n = 0; n < warmup_cfg.max_iter(); ++n) {
    adapt();
  }
  auto sample = adapt.sampler();
  for (std::size_t n = 0; n < sampling_cfg.max_iter(); ++n) {
    sample();
  }

  summarize(handler.draws());
  std::cout << std::endl;
  std::cout << "Micro step size = " << adapt.step_size() << std::endl;
  std::cout << "Min micro steps per macro step = " << adapt.min_micro_steps()
            << std::endl;
  std::cout << "Inverse mass matrix = " << std::fixed << std::setprecision(2)
            << adapt.inv_mass().transpose() << std::endl;
}

int main(int argc, char** argv) {
  if (argc > 1) DIM = std::stoul(argv[1]);
  if (argc > 2) WARMUP = std::stoul(argv[2]);
  if (argc > 3) SAMPLE = std::stoul(argv[3]);
  run_adaptive_walnuts(rw1);
  return 0;
}
