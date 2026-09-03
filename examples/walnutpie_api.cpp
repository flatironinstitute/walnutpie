#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

// overincludes, but it's what a client would call
#include <walnutpie.hpp>
#include "handlers.hpp"

double geom_mean_step(const std::vector<ChainStore>& handlers) {
  if (handlers.size() == 0) {
    return 0.0;
  }
  double sum = 0;
  for (const auto& handler : handlers) {
    sum += std::log(handler.step_size());
  }
  return std::exp(sum / handlers.size());
}

Eigen::VectorXd geom_mean_inv_mass(const std::vector<ChainStore>& handlers) {
  if (handlers.size() == 0) {
    return {};
  }
  Eigen::VectorXd sum =
      Eigen::VectorXd::Zero(handlers[0].diag_inv_mass().rows());
  for (const auto& handler : handlers) {
    sum += handler.diag_inv_mass().array().log().matrix();
  }
  return (sum / handlers.size()).array().exp().matrix();
}

// 0) TARGET DENSITY ===========================================================
static void std_normal(const Eigen::VectorXd& x, double& lp,
                       Eigen::VectorXd& grad) {
  lp = -0.5 * x.dot(x);
  grad = -x;
}

int main() {
  // 1) CONFIGUFRE =============================================================
  auto logp_grad = std_normal;

  std::size_t seed = 48;
  std::seed_seq seed_seq_for_init{seed, static_cast<std::size_t>(0)};
  std::mt19937 rng{seed_seq_for_init};
  std::size_t num_chains = 4;
  std::size_t dims = 100;

  CppInterruptCallback interrupt_callback;
  GlobalStore global_handler;
  std::vector<ChainStore> chain_handlers;
  chain_handlers.reserve(num_chains);

  for (std::size_t i = 0; i < num_chains; i++) {
    chain_handlers.emplace_back(ChainStore(true));
  }

  auto init_cfg =
      walnutpie::InitConfigBuilder(num_chains, dims)
          .step_sizes(100.2)  // test that adapt_step works with absurd init
          .adapt_step_build(rng, logp_grad);

  auto warmup_cfg =
      walnutpie::WarmupConfigBuilder().min_max_iter(50, 2000).build();

  auto sampling_cfg =
      walnutpie::SamplingConfigBuilder().min_max_iter(50, 1000).build();

  // std::cout << init_cfg << "\n\n";  // too verbose with multi-chain
  std::cout << warmup_cfg << "\n\n";
  std::cout << sampling_cfg << "\n\n";

  // 2) SAMPLE =================================================================
  // output sent to handlers
  walnutpie::WalnutsConfig config{std::move(init_cfg), std::move(warmup_cfg),
                                  std::move(sampling_cfg)};
  walnutpie::walnuts<std::mt19937_64>(seed, chain_handlers, global_handler,
                                      interrupt_callback, logp_grad, config);

  // 3) SUMMARIZE ==============================================================
  std::cout << "ADAPTATION RESULT: " << "\n";
  std::cout << "  geom_mean(step_size) = " << geom_mean_step(chain_handlers)
            << "\n";
  std::cout << "  geom_mean(inv_mass) = "
            << geom_mean_inv_mass(chain_handlers).transpose() << "\n\n";

  std::cout << "PER-CHAIN STATISTICS: " << "\n";
  for (size_t m = 0; m < num_chains; ++m) {
    std::cout
        << "  Chain " << m << "; step size = " << chain_handlers[m].step_size()
        << "; ||mass|| = "
        << chain_handlers[m].diag_inv_mass().array().inverse().matrix().norm()
        << "; # warmup_draws = " << chain_handlers[m].warmup_draws().size()
        << "; # draws = " << chain_handlers[m].draws().size() << "\n";
  }
  std::cout << "\n";

  std::cout << "NUMBER OF R-HAT EVALS: " << global_handler.r_hats().size()
            << ";  FINAL R-HAT: " << global_handler.r_hats().back() << "\n\n";

  std::cout << "WRITING BINARY TO FILES: step_size.wal, mass_matrix.wal, "
               "sample.wal\n\n";

  write_step_size("step_size.wal", chain_handlers);
  write_mass_matrix("mass_matrix.wal", chain_handlers);
  write_sample("sample.wal", chain_handlers);

  std::cout << "FINISHED NORMALLY." << std::endl << std::endl;
}
