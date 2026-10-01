#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <bridgestan.h>
#include <Eigen/Dense>
#include <walnutpie.hpp>
#include <walnutpie/load_stan.hpp>

#include "errors.hpp"
#include "export.h"
#include "handlers.hpp"
#include "interrupts.hpp"

namespace walnutpy {

void run_sampler(const walnutpie::LogpGrad auto& logp, int num_params,
                 walnutpie::InitConfigBuilder& init_cfg_builder, auto& handlers,
                 auto& global, size_t num_chains, unsigned int seed,
                 unsigned int id, const double* init_inv_metric,
                 int min_warmup_iter, int max_warmup_iter,
                 int min_sampling_iter, int max_sampling_iter,
                 int max_trajectory_doublings, int max_step_halvings,
                 int min_micro_steps, double max_hamiltonian_error,
                 double step_size_converge_tol, double mass_converge_tol,
                 double rhat_converge_tol, double mass_init_count,
                 double mass_additive_smoothing, double max_macro_steps_target,
                 double step_accept_rate_target, double step_learning_rate,
                 double step_gradient_decay, double step_sq_gradient_decay,
                 double step_stabilization, double step_learn_rate_decay) {
  interrupt::walnutpy_interrupt_handler interrupt;

  walnutpie::WarmupConfig warmup_cfg =
      walnutpie::WarmupConfigBuilder()
          .min_max_iter(min_warmup_iter, max_warmup_iter)
          .step_size_converge_tol(step_size_converge_tol)
          .mass_converge_tol(mass_converge_tol)
          // .publish_stride(publish_stride)
          // .yield_period(yield_period)
          .mass_init_count(mass_init_count)
          .mass_additive_smoothing(mass_additive_smoothing)
          .max_macro_steps_target(max_macro_steps_target)
          .step_accept_rate_target(step_accept_rate_target)
          .step_learning_rate(step_learning_rate)
          .step_gradient_decay(step_gradient_decay)
          .step_sq_gradient_decay(step_sq_gradient_decay)
          .step_stabilization(step_stabilization)
          .step_learn_rate_decay(step_learn_rate_decay)
          .build();

  walnutpie::SamplingConfig sample_cfg =
      walnutpie::SamplingConfigBuilder()
          .min_max_iter(min_sampling_iter, max_sampling_iter)
          .rhat_converge_tol(rhat_converge_tol)
          .max_trajectory_doublings(max_trajectory_doublings)
          .max_step_halvings(max_step_halvings)
          .max_hamiltonian_error(max_hamiltonian_error)
          .min_micro_steps(min_micro_steps)
          .build();

  if (init_inv_metric != nullptr) {
    std::vector<Eigen::VectorXd> mass_inits(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      mass_inits[i] = Eigen::Map<const Eigen::VectorXd>(
          init_inv_metric + (i * num_params), num_params);
    }
    init_cfg_builder.masses(mass_inits);
  } else {
    init_cfg_builder.masses(logp, mass_additive_smoothing);
  }

  std::seed_seq ss{seed, 2u};
  std::mt19937_64 init_rng(ss);

  walnutpie::WalnutsConfig walnuts_cfg{
      init_cfg_builder.adapt_step_build(init_rng, logp), std::move(warmup_cfg),
      std::move(sample_cfg)};

  walnutpie::walnuts<std::mt19937_64>(seed + id + num_chains, handlers, global,
                                      interrupt, logp, walnuts_cfg);
}

walnutpie::MarkovChainsUnified make_chains(const double* draws, int num_draws,
                                           int num_params, const int* lengths,
                                           int num_chains) {
  Eigen::Map<const Eigen::MatrixXd> draws_map(draws, num_draws, num_params);
  std::vector<std::size_t> lengths_vec(num_chains);
  for (int i = 0; i < num_chains; i++) {
    lengths_vec[i] = lengths[i];
  }
  return walnutpie::MarkovChainsUnified(draws_map, lengths_vec);
}

std::vector<std::string> split(const char* str, char sep) {
  if (str == nullptr) {
    return {};
  }

  std::vector<std::string> res;

  const char* cursor = str;
  const char* current = str;
  while (*cursor != '\0') {
    if (*cursor == sep) {
      res.emplace_back(current, cursor - current);
      current = cursor + 1;
    }
    cursor++;
  }

  if (cursor - current > 1) {
    res.emplace_back(current, cursor - current);
  }

  return res;
}

}  // namespace walnutpy

using namespace walnutpy;

extern "C" {

/** Function pointer to a log density function
 * This is intentionally a mirror of `RawLogpFunc` in
 * nutpie (https://github.com/pymc-devs/nutpie/blob/main/src/pymc.rs#L23)
 */
typedef int (*LOGP_CFUNC)(size_t theta_size, const double* theta, double* grad,
                          double* lp, void* data);

WALNUTPY_EXPORT int walnutpie_sample_cfunc(
    LOGP_CFUNC logp_c, void* data, int num_params, const double* inits,
    size_t num_chains, unsigned int seed, unsigned int id, double init_radius,
    const double* init_inv_metric, int min_warmup_iter, int max_warmup_iter,
    int min_sampling_iter, int max_sampling_iter, int max_trajectory_doublings,
    int max_step_halvings, int min_micro_steps, double max_hamiltonian_error,
    double step_size_converge_tol, double mass_converge_tol,
    double rhat_converge_tol, double mass_init_count,
    double mass_additive_smoothing, double max_macro_steps_target,
    double step_size_init, double step_accept_rate_target,
    double step_learning_rate, double step_gradient_decay,
    double step_sq_gradient_decay, double step_stabilization,
    double step_learn_rate_decay, bool save_warmup, double* out,
    size_t out_size, int* final_lengths, double* stepsize_out,
    double* inv_metric_out, int refresh, PRINT_CALLBACK print,
    WalnutpyError** err) {
  return error::catch_exceptions(err, [&]() {
    error::check_nonnegative("refresh", refresh);

    int draws_offset =
        num_params * (max_sampling_iter + max_warmup_iter * save_warmup);
    if (out_size < num_chains * draws_offset) {
      std::stringstream ss;
      ss << "Output buffer too small. Expected at least " << num_chains
         << " chains of " << draws_offset << " doubles, got " << out_size;
      throw std::runtime_error(ss.str());
    }

    auto logp = [&](const Eigen::VectorXd& x, double& logp,
                    Eigen::VectorXd& grad) {
      grad.resizeLike(x);
      int ret = logp_c(x.size(), x.data(), grad.data(), &logp, data);
      if (ret != 0) {
        throw std::runtime_error("logp failed with code " +
                                 std::to_string(ret));
      }
    };

    auto init_cfg_builder =
        walnutpie::InitConfigBuilder{num_chains,
                                     static_cast<std::size_t>(num_params)}
            .step_sizes(step_size_init);
    if (inits != nullptr) {
      std::vector<Eigen::VectorXd> theta_inits(num_chains);

      for (size_t i = 0; i < num_chains; ++i) {
        if (inits != nullptr) {
          theta_inits[i] = Eigen::Map<const Eigen::VectorXd>(
              inits + i * num_params, num_params);
        }
      }
      init_cfg_builder.positions(theta_inits);
    } else {
      std::seed_seq ss{seed, 1u};
      std::mt19937_64 rng(ss);
      init_cfg_builder.positions(rng, init_radius);
    }

    std::vector<PythonPrinter> printers;
    std::vector<BufferHandler> handlers;
    printers.reserve(num_chains);
    handlers.reserve(num_chains);
    for (size_t i = 0; i < num_chains; ++i) {
      printers.emplace_back(print, i + 1, refresh);
      handlers.emplace_back(
          out + draws_offset * i,
          stepsize_out != nullptr ? stepsize_out + i : nullptr,
          inv_metric_out != nullptr ? inv_metric_out + i * num_params : nullptr,
          save_warmup, printers[i]);
    }
    GlobalHandler global(printers[0]);
    run_sampler(
        logp, num_params, init_cfg_builder, handlers, global, num_chains, seed,
        id, init_inv_metric, min_warmup_iter, max_warmup_iter,
        min_sampling_iter, max_sampling_iter, max_trajectory_doublings,
        max_step_halvings, min_micro_steps, max_hamiltonian_error,
        step_size_converge_tol, mass_converge_tol, rhat_converge_tol,
        mass_init_count, mass_additive_smoothing, max_macro_steps_target,
        step_accept_rate_target, step_learning_rate, step_gradient_decay,
        step_sq_gradient_decay, step_stabilization, step_learn_rate_decay);

    for (size_t i = 0; i < num_chains; ++i) {
      final_lengths[i] = handlers[i].written_warmup();
      final_lengths[i + num_chains] = handlers[i].written_sampling();
    }

    return 0;
  });
}

static constexpr const char SEPARATOR = '\x1C';  ///< ASCII file separator
WALNUTPY_EXPORT char walnutpie_separator_char() { return SEPARATOR; }

WALNUTPY_EXPORT int walnutpie_sample_bridgestan(
    const char* bs_dll, const char* json_data, STREAM_CALLBACK callback,
    unsigned int model_seed, const char* inits, size_t num_chains,
    unsigned int seed, unsigned int id, double init_radius,
    const double* init_inv_metric, int min_warmup_iter, int max_warmup_iter,
    int min_sampling_iter, int max_sampling_iter, int max_trajectory_doublings,
    int max_step_halvings, int min_micro_steps, double max_hamiltonian_error,
    double step_size_converge_tol, double mass_converge_tol,
    double rhat_converge_tol, double mass_init_count,
    double mass_additive_smoothing, double max_macro_steps_target,
    double step_size_init, double step_accept_rate_target,
    double step_learning_rate, double step_gradient_decay,
    double step_sq_gradient_decay, double step_stabilization,
    double step_learn_rate_decay, bool save_warmup, double* out,
    size_t out_size, int* final_lengths, double* stepsize_out,
    double* inv_metric_out, int refresh, PRINT_CALLBACK print,
    WalnutpyError** err) {
  using walnutpie::DynamicStanModel;
  using walnutpie::unique_bs_rng;

  return error::catch_exceptions(err, [&]() {
    error::check_nonnegative("refresh", refresh);

    DynamicStanModel stan_model(bs_dll, json_data, model_seed, callback);

    int draws_offset = stan_model.constrained_dimensions() *
                       (max_sampling_iter + max_warmup_iter * save_warmup);
    if (out_size < num_chains * draws_offset) {
      std::stringstream ss;
      ss << "Output buffer too small. Expected at least " << num_chains
         << " chains of " << draws_offset << " doubles, got " << out_size;
      throw std::runtime_error(ss.str());
    }

    auto logp = [&](auto&&... args) { stan_model.logp_grad(args...); };

    std::vector<unique_bs_rng> rngs;
    std::vector<Eigen::VectorXd> theta_inits;
    std::vector<PythonPrinter> printers;
    std::vector<StanBufferHandler> handlers;
    rngs.reserve(num_chains);
    theta_inits.reserve(num_chains);
    printers.reserve(num_chains);
    handlers.reserve(num_chains);
    {
      std::seed_seq ss{seed, 1u};
      std::vector<std::uint32_t> seeds(num_chains);
      ss.generate(seeds.begin(), seeds.end());

      std::vector<std::string> chain_inits = split(inits, SEPARATOR);
      if (!chain_inits.empty() && chain_inits.size() != 1 &&
          chain_inits.size() != num_chains) {
        throw std::invalid_argument("Number of parameter initializations "
                                    "provided must be 0, 1, or match "
                                    "the number of chains");
      }

      for (size_t i = 0; i < num_chains; ++i) {
        rngs.push_back(stan_model.make_rng(seeds[i]));

        if (chain_inits.size() <= 1) {
          theta_inits.push_back(
              stan_model.initialize(inits, rngs[i], init_radius));
        } else {
          theta_inits.push_back(stan_model.initialize(chain_inits[i].c_str(),
                                                      rngs[i], init_radius));
        }

        printers.emplace_back(print, i + 1, refresh);
        handlers.emplace_back(
            stan_model, rngs[i], out + draws_offset * i,
            stepsize_out != nullptr ? stepsize_out + i : nullptr,
            inv_metric_out != nullptr
                ? inv_metric_out + i * stan_model.unconstrained_dimensions()
                : nullptr,
            save_warmup, printers[i]);
      }
    }

    auto init_cfg_builder =
        walnutpie::InitConfigBuilder{
            num_chains, static_cast<std::size_t>(theta_inits[0].size())}
            .step_sizes(step_size_init)
            .positions(theta_inits);

    GlobalHandler global(printers[0]);
    run_sampler(logp, stan_model.unconstrained_dimensions(), init_cfg_builder,
                handlers, global, num_chains, seed, id, init_inv_metric,
                min_warmup_iter, max_warmup_iter, min_sampling_iter,
                max_sampling_iter, max_trajectory_doublings, max_step_halvings,
                min_micro_steps, max_hamiltonian_error, step_size_converge_tol,
                mass_converge_tol, rhat_converge_tol, mass_init_count,
                mass_additive_smoothing, max_macro_steps_target,
                step_accept_rate_target, step_learning_rate,
                step_gradient_decay, step_sq_gradient_decay, step_stabilization,
                step_learn_rate_decay);

    for (size_t i = 0; i < num_chains; ++i) {
      final_lengths[i] = handlers[i].written_warmup();
      final_lengths[i + num_chains] = handlers[i].written_sampling();
    }

    return 0;
  });
}

WALNUTPY_EXPORT int walnutpie_ess(const double* draws, int num_draws,
                                  int num_params, const int* lengths,
                                  int num_chains, double* out,
                                  WalnutpyError** err) {
  return error::catch_exceptions(err, [&]() {
    auto chains =
        make_chains(draws, num_draws, num_params, lengths, num_chains);
    Eigen::Map<Eigen::RowVectorXd>(out, num_params) =
        walnutpie::effective_sample_size(chains);
    return 0;
  });
}

WALNUTPY_EXPORT int walnutpie_r_hat(const double* draws, int num_draws,
                                    int num_params, const int* lengths,
                                    int num_chains, double* out,
                                    WalnutpyError** err) {
  return error::catch_exceptions(err, [&]() {
    auto chains =
        make_chains(draws, num_draws, num_params, lengths, num_chains);
    Eigen::Map<Eigen::RowVectorXd>(out, num_params) = walnutpie::r_hat(chains);
    return 0;
  });
}

WALNUTPY_EXPORT int walnutpie_mcse(const double* draws, int num_draws,
                                   int num_params, const int* lengths,
                                   int num_chains, double* out,
                                   WalnutpyError** err) {
  return error::catch_exceptions(err, [&]() {
    auto chains =
        make_chains(draws, num_draws, num_params, lengths, num_chains);
    Eigen::Map<Eigen::RowVectorXd>(out, num_params) =
        walnutpie::monte_carlo_standard_error(chains);
    return 0;
  });
}

WALNUTPY_EXPORT const char* walnutpie_get_error_message(
    const WalnutpyError* err) {
  if (err == nullptr) {
    return "Something went wrong: No error found";
  }
  return err->msg.c_str();
}

WALNUTPY_EXPORT WalnutpyErrorType
walnutpie_get_error_type(const WalnutpyError* err) {
  if (err == nullptr) {
    return WalnutpyErrorType::generic;
  }
  return err->type;
}

WALNUTPY_EXPORT void walnutpie_destroy_error(WalnutpyError* err) {
  delete (err);
}
}  // extern "C"
