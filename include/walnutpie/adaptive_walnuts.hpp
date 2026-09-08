#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <utility>

#include <Eigen/Dense>

#include "walnutpie/adam.hpp"
#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"
#include "walnutpie/online_moments.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/walnuts.hpp"

namespace walnutpie::detail {

/**
 * @brief A mass matrix estimator based on exponentially discounted draws
 * and scores (gradients of log densities).
 */
class MassEstimator {
 public:
  /**
   * @brief Construct a mass matrix estimator with the specified configuration,
   * at the specified initial position and gradient of the log density at the
   * position.
   *
   * The estimator observes positions and their gradients at given iterations
   * with the function `observe()`. At each step, the discount factor for
   * discounting past draws the online moment estimators is set to
   * ```
   * discount_factor = 1 - 1 / (iter_offset + iter)
   * ```
   * where `iter_offset` is the offset specified in the configuration and `iter`
   * is the iteration number for the observation.
   *
   * The initial estimate is additively smoothed by multiplying by one
   * minus the additive smoothing and adding the additive smoothing.
   *
   * The final estimate for the inverse mass matrix is given by the
   * geometric mean of the variance of the scores (the inverse
   * variance estimator) and the inverse variance of the draws (the
   * variance estimator).
   *
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] init_cfg The initialization configuration.
   * @throw std::invalid_argument If the position and gradient are not the same
   * size.
   */
  MassEstimator(const WarmupConfig& warmup_cfg, const InitChainConfig& init_cfg)
      : warmup_cfg_(warmup_cfg) {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(init_cfg.position().size());
    score_var_estimator_ =
        OnlineMoments(warmup_cfg.mass_init_count(), zero, init_cfg.mass());
    draw_var_estimator_ =
        OnlineMoments(warmup_cfg.mass_init_count(), zero,
                      init_cfg.mass().array().inverse().matrix());
  }

  /**
   * @brief Update the estimate for the specified iteration with the
   * observation and gradient.
   *
   * @param[in] theta The position observed.
   * @param[in] grad The gradient of the log density at the position.
   * @param[in] iteration The iteration number.
   * @pre theta.size() = grad.size()
   * @pre iteration >= 0
   */
  void observe(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad,
               std::size_t iteration) {
    double discount_factor = 1.0 - 1.0 / (warmup_cfg_.mass_init_count() +
                                          static_cast<double>(iteration));
    draw_var_estimator_.discount_observe(discount_factor, theta);
    score_var_estimator_.discount_observe(discount_factor, grad);
  }

  /**
   * @brief Return an estimate of the inverse mass matrix. The result
   * is the geometric average of the variance of the draws and the
   * inverse variance of the scores.
   *
   * @return The inverse mass matrix estimate.
   */
  Eigen::VectorXd inv_mass_estimate() const {
    return (draw_var_estimator_.variance().array() /
            score_var_estimator_.variance().array())
        .sqrt()
        .matrix();
  }

 private:
  /** The warmup configuration for adaptive Walnutpie. */
  WarmupConfig warmup_cfg_;

  /** The online variance estimator for draws. */
  OnlineMoments draw_var_estimator_;

  /** The online inverse variance estimator for scores. */
  OnlineMoments score_var_estimator_;
};

/**
 * @brief The adaptation handler for the minimum number of micro steps
 * per macro step.
 *
 * After being constructed with a target number of macro steps, this
 * class is given observations of the number of micro steps taken and
 * adjusts the minimum number of micro steps per macro step in order
 * to achieve the target expected number of macro steps historically.
 * There is slight regularization of a single observation at depth 2,
 * but otherwise it just uses the floor of an average and thus rounds
 * down.
 */
class MinMicroStepsAdaptHandler {
 public:
  /**
   * Construct a minimum number of micro steps per macro step handler.
   *
   * @param[in] target_macro_steps Target number of expected macro steps.
   * @param[in] min_micro_steps The minimum number of micro steps to return.
   */
  MinMicroStepsAdaptHandler(double target_macro_steps,
                            std::size_t min_micro_steps)
      : target_macro_steps_(target_macro_steps),
        min_micro_steps_(min_micro_steps),
        total_macro_steps_(2.0),
        count_(1.0) {}

  /**
   * @brief Observe the specified number of macro steps in a Nuts trajectory.
   *
   * @param[in] macro_steps The number of macro steps used in a trajectory.
   */
  void observe(std::size_t macro_steps) {
    total_macro_steps_ += static_cast<double>(macro_steps);
    ++count_;
  }

  /**
   * @brief Return the estimated minimum number of micro steps.
   *
   * This estimate is designed to achieve the expected number of macro steps
   * per iteration.
   *
   * @return The minimum number of micro steps to use per macro step.
   */
  std::size_t min_micro_steps() const noexcept {
    double mean_micro = total_macro_steps_ / count_;
    double min_micro_steps = mean_micro / target_macro_steps_;
    return std::max(min_micro_steps_,
                    static_cast<std::size_t>(std::lround(min_micro_steps)));
  }

 private:
  const double target_macro_steps_;
  const std::size_t min_micro_steps_;
  double total_macro_steps_;
  double count_;
};

}  // namespace walnutpie::detail

namespace walnutpie {

/**
 * @brief The adaptive Walnuts sampler.
 *
 * The adaptive Walnuts sampler is configured in the constructor, then
 * provides a functor method `operator()()` for returning the next
 * state in warmup. Warmup re-estimates step size and mass matrix
 * each iteration, exponentially discounting the past.
 *
 * @tparam F Type of log density/gradient function.
 * @tparam RNG Type of base random number generator.
 * @tparam Handler Type of adaptation and sampling event handler.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG, ChainHandler H>
class AdaptiveWalnuts {
 public:
  /**
   * @brief Construct an adaptive Walnuts sampler.
   *
   * The configuration objects, the base random number generator, and
   * the log density/gradient function are held by reference. The RNG
   * changes state every time a random number is generated. The
   * target depth specifies the expected Nuts tree depth, which is
   * controlled through the minimum number of micro steps per macro
   * step and adjusted with a mean estimator to achieve this average.
   *
   * @param[in] rng The base random number generator, stored by reference and
   * modifed.
   * @param[in,out] handler Event handler for adaptation and sampling, stored by
   * reference and called back.
   * @param[in] logp_grad The target log density and gradient function.
   * @param[in] init_chain_cfg The initialization configuration for a single
   * chain.
   * @param[in] warmup_cfg The warmup configuration.
   * @param[in] sampling_cfg The sampling configuration.
   * @param[in] endpoint_reuse Deterministic reuse requires a stable,
   * position-only target, including across warmup and completion callbacks.
   * Invalidate after changes. The returned sampler inherits this policy and
   * any finite endpoint; zero warmup leaves the cache empty.
   */
  AdaptiveWalnuts(RNG& rng, H& handler, const F& logp_grad,
                  const InitChainConfig& init_chain_cfg,
                  const WarmupConfig& warmup_cfg,
                  const SamplingConfig& sampling_cfg,
                  EndpointReuse endpoint_reuse = EndpointReuse::Disabled)
      : warmup_cfg_(std::cref(warmup_cfg)),
        sampling_cfg_(std::cref(sampling_cfg)),
        rand_(rng),
        handler_(handler),
        logp_grad_(logp_grad, handler),
        theta_(init_chain_cfg.position()),
        endpoint_reuse_(endpoint_reuse),
        iteration_(0),
        adam_(init_chain_cfg.step_size(), warmup_cfg.step_accept_rate_target(),
              warmup_cfg.step_learning_rate(), warmup_cfg.step_gradient_decay(),
              warmup_cfg.step_sq_gradient_decay(),
              warmup_cfg.step_stabilization(),
              warmup_cfg.step_learn_rate_decay()),
        mass_estimator_(warmup_cfg, init_chain_cfg),
        min_micro_estimator_(warmup_cfg.max_macro_steps_target(),
                             sampling_cfg.min_micro_steps()) {}

  /**
   * @brief Generate the next state for adaptation and the handler.
   *
   * This method should be called a number of time equal to the number
   * of warmup iterations desired. These warmup draws are *not* drawn
   * from a Markov chain and are not valid for inference. After
   * warmup, call `sampler()` to return a sampler that fixes the
   * tuning parameters and provides a proper Markov chain.
   */
  void operator()() {
    Eigen::VectorXd inv_mass = mass_estimator_.inv_mass_estimate();
    Eigen::VectorXd chol_mass = inv_mass.array().inverse().sqrt().matrix();
    Eigen::VectorXd grad_select;
    double logp_select;
    std::size_t depth;
    theta_ = detail::transition_w_impl(
        rand_, logp_grad_, inv_mass, chol_mass, adam_.step_size(),
        sampling_cfg_.get().max_trajectory_doublings(),
        sampling_cfg_.get().max_step_halvings(),
        min_micro_estimator_.min_micro_steps(),
        sampling_cfg_.get().max_hamiltonian_error(), std::move(theta_), depth,
        grad_select, logp_select, adam_,
        endpoint_reuse_ == EndpointReuse::Deterministic ? &endpoint_cache_
                                                        : nullptr);
    mass_estimator_.observe(theta_, grad_select, iteration_);
    min_micro_estimator_.observe(1 << depth);
    if (endpoint_reuse_ == EndpointReuse::Deterministic) {
      endpoint_cache_.store(std::move(grad_select), logp_select);
    }
    handler_.get().on_warmup(theta_, logp_select, step_size(), inv_mass);
    ++iteration_;
  }

  /**
   * @brief Return a Walnuts sampler with the current tuning parameter
   * estimates.
   *
   * The returned sampler forms a proper Markov chain. The method passes
   * along the compound random number generator and log density function and
   * is hence not marked `const`.
   *
   * @return The Walnuts sampler with current tuning parameter estimates.
   */
  WalnutsSampler<F, RNG, H> sampler() {
    handler_.get().on_warmup_complete(step_size(), inv_mass());
    WalnutsSampler<F, RNG, H> result(
        rand_.rng(), handler_, logp_grad_.logp_grad_, theta_, inv_mass(),
        step_size(), sampling_cfg_.get().max_trajectory_doublings(),
        sampling_cfg_.get().max_step_halvings(),
        min_micro_estimator_.min_micro_steps(),
        sampling_cfg_.get().max_hamiltonian_error(), endpoint_reuse_);
    result.endpoint_cache_ = endpoint_cache_;
    return result;
  }

  /** Discard the endpoint after changing target data between draws. */
  void invalidate_endpoint_cache() noexcept { endpoint_cache_.valid = false; }

  /**
   * @brief Return the diagonal of the diagonal inverse mass matrix.
   *
   * @return The diagonal of the inverse mass matrix.
   */
  Eigen::VectorXd inv_mass() const {
    return mass_estimator_.inv_mass_estimate();
  }

  /**
   * @brief Return the step size.
   *
   * @return The step size.
   */
  double step_size() const { return adam_.step_size(); }

  /**
   * @brief Return the minimum number of micro steps per macro step.
   *
   * @return The minimum number of micro steps per macro step.
   */
  std::size_t min_micro_steps() const {
    return min_micro_estimator_.min_micro_steps();
  }

  /**
   * @brief Return the number of dimensions of the position.
   *
   * @return The number of dimensions.
   */
  std::size_t dim() const noexcept {
    return static_cast<std::size_t>(theta_.size());
  }

  /**
   * @brief Return the natural logarithm of the step size.
   *
   * @return The log of the step size.
   */
  double log_step_size() const noexcept { return std::log(step_size()); }

  /**
   * @brief Return the natural logarithm of the diagonal of the
   * diagonal mass matirx.
   *
   * @return The log of the diagonal of the mass matrix.
   */
  Eigen::VectorXd log_mass() const {
    // equiv. inv_mass().array().inverse().log().matrix();
    return -inv_mass().array().log().matrix();
  }

  /**
   * @brief Return the current iteration.
   *
   * @return The iteration.
   */
  std::size_t iter() const noexcept { return iteration_; }

 private:
  /** The warmup configuration. */
  std::reference_wrapper<const WarmupConfig> warmup_cfg_;

  /** The Walnuts sampler configuration. */
  std::reference_wrapper<const SamplingConfig> sampling_cfg_;

  /** The random number generator required for Nuts. */
  detail::Random<RNG> rand_;

  /** The adaptation and sampling event handler. */
  std::reference_wrapper<H> handler_;

  /** The target log density/gradient function. */
  const detail::NoExceptLogpGrad<F, H> logp_grad_;

  /** The current state. */
  Eigen::VectorXd theta_;

  EndpointReuse endpoint_reuse_;
  detail::EndpointCache endpoint_cache_;

  /** The current iteration. */
  std::size_t iteration_;

  /** The Adam optimizer for step size adaptation.
   */
  detail::Adam adam_;

  /** The estimator for the mass matrix. */
  detail::MassEstimator mass_estimator_;

  /** The estimator for the minimum number of micro steps per macro step. */
  detail::MinMicroStepsAdaptHandler min_micro_estimator_;
};

}  // namespace walnutpie
