#pragma once

#include <cstddef>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/validate.hpp"

namespace walnutpie {

/**
 * @brief The initialization configuration for a single Markov chain.
 *
 * The initialization configuration specifies a step size, initial position,
 * and initial mass matrix.
 */
class InitChainConfig {
 public:
  /**
   * @brief Construct an initialization configuration.
   *
   * @param[in] step_size The initial step size.
   * @param[in] position The initial position.
   * @param[in] mass The initial mass matrix (diagonal).
   */
  InitChainConfig(double step_size, const Eigen::VectorXd& position,
                  const Eigen::VectorXd& mass)
      : step_size_(step_size), position_(position), mass_(mass) {}

  /**
   * @brief Return the initial step size.
   *
   * @return The step size.
   */
  double step_size() const noexcept { return step_size_; }

  /**
   * @brief Return the initial position.
   *
   * @return The position.
   */
  const Eigen::VectorXd& position() const noexcept { return position_; }

  /**
   * @brief Return the initial mass matrix.
   *
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass() const noexcept { return mass_; }

 private:
  double step_size_;
  Eigen::VectorXd position_;
  Eigen::VectorXd mass_;
};

/**
 * @brief The initialization configuration for multiple Markov chains.
 *
 * Rather than a public constructor, it is built using an
 * `InitConfigBuilder` instance.
 *
 * The initialization configuration specifies a step size, initial
 * position, and initial mass matrix.
 */
class InitConfig {
 public:
  /**
   * @brief Return the number of chains.
   *
   * @return The number of chains.
   */
  std::size_t num_chains() const noexcept { return step_sizes_.size(); }

  /**
   * @brief Return the dimensionality of the positions.
   *
   * If the initialization is empty, 0 is returned.
   *
   * @return The dimensionality.
   */
  std::size_t dims() const noexcept {
    return positions_.empty()
               ? 0u
               : static_cast<std::size_t>(positions_.front().size());
  }

  /**
   * @brief Return the initial step sizes.
   *
   * @return The step sizes.
   */
  const std::vector<double>& step_sizes() const noexcept { return step_sizes_; }

  /**
   * @brief Return the initial step size for the specified chain.
   *
   * @param[in] n The chain identifier.
   * @return The step size.
   */
  double step_size(std::size_t n) const noexcept { return step_sizes_[n]; }

  /**
   * @brief Return the initial positions for all chains.
   *
   * @return The positions.
   */
  const std::vector<Eigen::VectorXd>& positions() const noexcept {
    return positions_;
  }

  /**
   * @brief Return the initial position for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The positions.
   */
  const Eigen::VectorXd& position(std::size_t n) const noexcept {
    return positions_[n];
  }

  /**
   * @brief Return the initial diagonal mass matrices for all chains.
   *
   * @return The mass matrix diagonals.
   */
  const std::vector<Eigen::VectorXd>& masses() const noexcept {
    return masses_;
  }

  /**
   * @brief Return the mass matrix for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The mass matrix.
   */
  const Eigen::VectorXd& mass(std::size_t n) const noexcept {
    return masses_[n];
  }

  /**
   * @brief Return the initialization configuration for the specified chain.
   *
   * @param[in] n The chain index.
   * @return The indexed chain's initialization configuration.
   */
  InitChainConfig init_chain_config(std::size_t n) const {
    return InitChainConfig(step_size(n), position(n), mass(n));
  }

 private:
  friend class InitConfigBuilder;

  /**
   * @brief Construct an initialization configuration.
   *
   * This constructor does not validate arguments because it is only
   * called internally. It only implements rvalue moves because that
   * is the only way it is called.
   *
   * @param[in] step_sizes The step sizes.
   * @param[in] positions The positions.
   * @param[in] masses The diagonals of the diagonal mass matrixes.
   */
  InitConfig(std::vector<double>&& step_sizes,
             std::vector<Eigen::VectorXd>&& positions,
             std::vector<Eigen::VectorXd>&& masses)
      : step_sizes_(std::move(step_sizes)),
        positions_(std::move(positions)),
        masses_(std::move(masses)) {}

  InitConfig() = default;

  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;
};

/**
 * @brief The builder for initialization configurations.
 *
 * The usage to return an `InitConfig` is
 * `InitConfigBuilder(4, 20).step_sizes(0.5).build();`
 * with any number of config methods
 * chained between the construction and call to build.
 */
class InitConfigBuilder {
 public:
  /**
   * @brief Construct an initialization builder of the given sizes.
   *
   * @param[in] num_chains The number of Markov chains.
   * @param[in] dims The dimensionality of each chain.
   */
  InitConfigBuilder(std::size_t num_chains, std::size_t dims)
      : num_chains_(num_chains),
        dims_(dims),
        step_sizes_(std::vector<double>(num_chains, 0.1)),
        positions_(std::vector<Eigen::VectorXd>(
            num_chains,
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dims)))),
        masses_(std::vector<Eigen::VectorXd>(
            num_chains,
            Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dims)))) {}

  /**
   * @brief Set the step sizes to all be the specified value.
   *
   * @param[in] v The step size.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the step size is not finite and positive.
   */
  InitConfigBuilder& step_sizes(double v) {
    detail::validate_finite_positive(v, "step size");
    step_sizes_ = std::vector<double>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Set the step sizes to all be the specified values.
   *
   * @param[in] v The step sizes.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If any of the step sizes are not finite
   * positive.
   * @throw std::invalid_argument If the number of chains doesn't match the
   * number specified in the constructor.
   */
  InitConfigBuilder& step_sizes(const std::vector<double>& v) {
    detail::validate_size(v, num_chains_, "step_sizes", "num_chains");
    detail::validate_finite_positive(v, "step_size");
    step_sizes_ = v;
    return *this;
  }

  /**
   * @brief Randomly initialization the positions.
   *
   * Initialization is independent in each dimension with values drawn
   * from a zero-centered normal distribution with the specified
   * scale.
   *
   * @tparam RNG The type of the base random number generator.
   * @param[in,out] rng The base random number generator.
   * @param[in] init_scale The scale of the normal initial values.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the initial scale is not finite and
   * positive.
   */
  template <std::uniform_random_bit_generator RNG>
  InitConfigBuilder& positions(RNG& rng, double init_scale) {
    detail::validate_finite_positive(init_scale, "init_scale");
    detail::Random<RNG> rand(rng);
    positions_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      rand.standard_normal(static_cast<Eigen::Index>(dims_), positions_[c]);
      positions_[c] *= init_scale;
    }
    return *this;
  }

  /**
   * @brief Initialize the positions all to the same value.
   *
   * @param[in] v The initial position.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   */
  InitConfigBuilder& positions(const Eigen::VectorXd& v) {
    detail::validate_size(v, dims_, "position", "dims");
    detail::validate_finite(v, "position");
    positions_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values.
   *
   * @param[in] vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(const std::vector<Eigen::VectorXd>& vs) {
    detail::validate_size(vs, num_chains_, "positions", "num_chains");
    detail::validate_finite(vs, "positions");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "position", "dims");
    }
    positions_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the positions to the specified values via move.
   *
   * @param[in] vs The initial positions.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial positions doesn't
   * match the number of chains specified in the constructor.
   * @throw std::invalid_argument If any of the initial positions contains
   * non-finite values.
   * @throw std::invalid_argumet If any of the initial positions has a
   * dimensionality that does not match the dimensionality specified in the
   * constructor.
   */
  InitConfigBuilder& positions(std::vector<Eigen::VectorXd>&& vs) {
    detail::validate_size(vs, num_chains_, "positions", "num_chains");
    detail::validate_finite(vs, "positions");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "position", "dims");
    }
    positions_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Initialize the masses using the Nutpie outer product strategy.
   *
   * Following Nutpie, the initialization uses a smoothed negative
   * outer product of gradient, which is the absolute value of the
   * outer product of gradients linearly interpolated with a unit
   * matrix with weight `mass_smoothing` on the unit matrix and `1 -
   * mass_smoothing` on the regularized outer product.
   *
   * If the flag `average_masses` is `true`, then each chain's mass
   * matrix is set to the geometric average of the per-chain mass
   * matrixes.
   *
   * See: Seyboldt, Adrian and Carlson, Eliot and Carpenter,
   * Bob. 2026. [Preconditioning Hamiltonian Monte Carlo by
   * minimizing Fisher
   * divergence](https://arxiv.org/abs/2603.18845v1). arXiv
   * 2603.18845.
   *
   * @tparam LPG The type of the log density and gradient function.
   * @param[in] logp_grad The log density and gradient function, called back.
   * @param[in] mass_smoothing The additive smoothing for mass matrices.
   * @param[in] average_masses Set to `true` to geometrically average mass
   * matrices.
   * @throw std::invalid_argumet If the mass smoothing is not in (0, 1).
   * @return A reference to this builder for chaining.
   */
  template <LogpGrad F>
  InitConfigBuilder& masses(const F& logp_grad, double mass_smoothing,
                            bool average_masses = false) {
    detail::validate_probability(mass_smoothing, "mass_smoothing");
    Eigen::VectorXd grad;
    masses_.resize(num_chains_);
    for (std::size_t c = 0; c < num_chains_; ++c) {
      double lp_to_discard;
      logp_grad(positions_[c], lp_to_discard, grad);
      masses_[c] = (1 - mass_smoothing) * grad.array().abs() + mass_smoothing;
    }
    if (average_masses) {
      Eigen::Index D = masses_[0].size();
      Eigen::VectorXd sum_log_mass = Eigen::VectorXd::Zero(D);
      for (const auto& mass : masses_) {
        sum_log_mass += mass.array().log().matrix();
      }
      auto avg_log_mass = sum_log_mass / num_chains_;
      auto geom_mean_mass = avg_log_mass.array().exp().matrix();
      masses_ = std::vector<Eigen::VectorXd>(num_chains_, geom_mean_mass);
    }
    return *this;
  }

  /**
   * @brief Initialize the mass matrices all to the same value.
   *
   * @param[in] v The initial diagonal mass matrix.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the dimensionality doesn't match
   * that specified during construction.
   * @throw std::invalid_argument If any of the initial mass matrix
   * diagonals contains non-finite or non-positive values.
   */
  InitConfigBuilder& masses(const Eigen::VectorXd& v) {
    detail::validate_size(v, dims_, "masses", "dims");
    detail::validate_finite_positive(v, "masses");
    masses_ = std::vector<Eigen::VectorXd>(num_chains_, v);
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values.
   *
   * @param[in] vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(const std::vector<Eigen::VectorXd>& vs) {
    detail::validate_size(vs, num_chains_, "masses", "num_chains");
    detail::validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = vs;
    return *this;
  }

  /**
   * @brief Initialize the mass matrices to the specified values via
   * move.
   *
   * @param[in] vs The initial mass matrices.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the number of initial mass
   * matrices doesn't match the number of chains specified in the
   * constructor.
   * @throw std::invalid_argument If any of the initial mass matrices
   * contains non-finite values.
   * @throw std::invalid_argumet If any of the initial mass matrices
   * has a dimensionality that does not match the dimensionality
   * specified in the constructor.
   */
  InitConfigBuilder& masses(std::vector<Eigen::VectorXd>&& vs) {
    detail::validate_size(vs, num_chains_, "masses", "num_chains");
    detail::validate_finite_positive(vs, "masses");
    for (const auto& v : vs) {
      detail::validate_size(v, dims_, "all masses", "dims");
    }
    masses_ = std::move(vs);
    return *this;
  }

  /**
   * @brief Return the initialization configuration.
   *
   * @return The initialization configuration.
   */
  InitConfig build() {
    return InitConfig{std::move(step_sizes_), std::move(positions_),
                      std::move(masses_)};
  }

  /**
   * @brief Heuristically adapt the initial step sizes, then return
   * the initialization configuration.
   *
   * @tparam RNG Type of the base random number generator.
   * @tparam F Type of the log density and gradient function.
   * @param[in] rng The base random number generator.
   * @param[in] logp_grad The log density and gradient function.
   */
  template <std::uniform_random_bit_generator RNG, LogpGrad F>
  InitConfig adapt_step_build(RNG& rng, const F& logp_grad) {
    for (std::size_t c = 0; c < num_chains_; ++c) {
      step_sizes_[c] = detail::adapt_step(rng, logp_grad, positions_[c],
                                          masses_[c], step_sizes_[c], dims_);
    }
    return build();
  }

 private:
  std::size_t num_chains_;
  std::size_t dims_;
  std::vector<double> step_sizes_;
  std::vector<Eigen::VectorXd> positions_;
  std::vector<Eigen::VectorXd> masses_;
};

/**
 * @internal @brief Write a dump of the initial configurations to the
 * specified stream.
 *
 * @param[in,out] out Stream to which configuration is written.
 * @param[in] cfg The configuration to write.
 * @return A reference to the output stream for chaining.
 */
inline std::ostream& operator<<(std::ostream& out, const InitConfig& cfg) {
  out << "InitConfigs (by chain)\n";
  for (std::size_t n = 0; n < cfg.step_sizes().size(); ++n) {
    if (n > 0) {
      out << "\n";
    }
    out << "  chain         = " << n << "\n"
        << "    num_chains  = " << cfg.num_chains() << "\n"
        << "    step_size   = " << cfg.step_sizes()[n] << "\n"
        << "    position    = " << cfg.positions()[n].transpose() << "\n"
        << "    mass        = " << cfg.masses()[n].transpose() << "\n";
  }
  return out;
}

/**
 * @brief The warmup configuration object. The object supplies methods
 * for all of the tuning parameters for warmup.
 */
class WarmupConfig {
 public:
  /**
   * @brief Return the minimum number of warmup iterations.
   *
   * @return Minimum warmup iterations.
   */
  std::size_t min_iter() const { return min_iter_; }

  /**
   * @brief Return the maximum number of warmup iterations.
   *
   * @return Maximum warmup iterations.
   */
  std::size_t max_iter() const { return max_iter_; }

  /**
   * @brief Return the step-size convergence tolerance.
   *
   * @return The step-size convergence tolerance.
   */
  double step_size_converge_tol() const { return step_size_converge_tol_; }

  /**
   * @brief Return the mass matrix convergence tolerance. The
   * tolerance is for the L2-norm of the diagonal.
   *
   * @return The mass matrix  convergence tolerance.
   */
  double mass_converge_tol() const { return mass_converge_tol_; }

  /**
   * @brief Return the initial count for the mass matrix estimator.
   *
   * @return The initial count for the mass matrix estimator.
   */
  double mass_init_count() const { return mass_init_count_; }

  /**
   * @brief Return the additive smoothing for the mass matrix estimator.
   *
   * @return The additive smoothing for the mass matrix estimator.
   */
  double mass_additive_smoothing() const { return mass_additive_smoothing_; }

  /**
   * @brief Return the target number of macro steps.
   *
   * @return The target number of macro steps.
   */
  double max_macro_steps_target() const { return max_macro_steps_target_; }

  /**
   * @brief Return the target acceptance rate.
   *
   * @return The target acceptance rate.
   */
  double step_accept_rate_target() const { return step_accept_rate_target_; }

  /**
   * @brief Return the step-size learning rate for the Adam optimizer.
   *
   * @return The step-size learning rate for the Adam optimizer.
   */
  double step_learning_rate() const { return step_learning_rate_; }

  /**
   * @brief Return the gradient decay rate for the Adam optimizer.
   *
   * @return The gradient decay rate for the Adam optimizer.
   */
  double step_gradient_decay() const { return step_gradient_decay_; }

  /**
   * @brief Return the squared gradient decay rate for the Adam optimizer.
   *
   * @return The squared gradient decay rate for the Adam optimizer.
   */
  double step_sq_gradient_decay() const { return step_sq_gradient_decay_; }

  /**
   * @brief Return the step-size stabilization.
   *
   * @return The step-size stabilization.
   */
  double step_stabilization() const { return step_stabilization_; }

  /**
   * @brief Return the step-size learning rate decay factor.
   *
   * @return The step-size learning rate decay factor.
   */
  double step_learn_rate_decay() const { return step_learn_rate_decay_; }

  /**
   * @brief Return the stride for publishing updates for convergence monitoring.
   *
   * @return The stride for publishing updates for convergence monitoring.
   */
  std::size_t publish_stride() const { return publish_stride_; }

  /**
   * @brief Return the period at which threads for chains yield.
   *
   * @return The the period at which threads for chains yield.
   */
  std::size_t yield_period() const { return yield_period_; }

 private:
  friend class WarmupConfigBuilder;

  WarmupConfig() = default;

  std::size_t min_iter_ = 50;
  std::size_t max_iter_ = 1000;
  double step_size_converge_tol_ = 0.1;
  double mass_converge_tol_ = 1.0;
  double mass_init_count_ = 4.0;
  double mass_additive_smoothing_ = 1e-5;
  double max_macro_steps_target_ = 15.0;
  double step_accept_rate_target_ = 0.8;
  double step_learning_rate_ = 0.05;
  double step_gradient_decay_ = 0.8;
  double step_sq_gradient_decay_ = 0.9;
  double step_stabilization_ = 1e-4;
  double step_learn_rate_decay_ = 0.5;
  std::size_t publish_stride_ = 5;
  std::size_t yield_period_ = 32;
};

/**
 * @brief The builder for `WarmupConfig` objects.
 */
class WarmupConfigBuilder {
 public:
  /**
   * @brief Set the minimum and maximum number of warmup iterations.
   *
   * @param[in] min_iter The minimum number of warmup iterations.
   * @param[in] max_iter The maximum number of warmup iterations.
   * @return This builder for chaining.
   * @throw std::invalid_argument If `min_iter` > `max_iter`.
   */
  WarmupConfigBuilder& min_max_iter(std::size_t min_iter,
                                    std::size_t max_iter) {
    if (min_iter > max_iter) {
      throw std::invalid_argument(
          "min_iter cannot be greater than than max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
    return *this;
  }

  /**
   * @brief Set the step size convergence tolerance.
   *
   * @param[in] v The step size convergence tolerance.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and positive.
   */
  WarmupConfigBuilder& step_size_converge_tol(double v) {
    detail::validate_finite_positive(v, "step_size_converge_tol");
    cfg_.step_size_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix L2-norm convergence tolerance.
   *
   * @param[in] v The mass matrix L2-norm convergence tolerance.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and positive.
   */
  WarmupConfigBuilder& mass_converge_tol(double v) {
    detail::validate_finite_positive(v, "mass_converge_tol");
    cfg_.mass_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix estimator initial count.
   *
   * @param[in] v The mass matrix estimator initial count.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the initial count is not finite and
   * greater than one.
   */
  WarmupConfigBuilder& mass_init_count(double v) {
    detail::validate_finite_gt1(v, "mass_init_count");
    cfg_.mass_init_count_ = v;
    return *this;
  }

  /**
   * @brief Set the mass matrix estimator additive smoothing.
   *
   * @param[in] v The mass matrix estimator additive smoothing.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the smoothing is not finite and positive.
   */
  WarmupConfigBuilder& mass_additive_smoothing(double v) {
    detail::validate_finite_positive(v, "mass_additive_smoothing");
    cfg_.mass_additive_smoothing_ = v;
    return *this;
  }

  /**
   * @brief Set the target number of macro steps.
   *
   * @param[in] v The target number of macro steps.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the target is not finite and positive.
   */
  WarmupConfigBuilder& max_macro_steps_target(double v) {
    detail::validate_finite_positive(v, "max_macro_steps_target");
    cfg_.max_macro_steps_target_ = v;
    return *this;
  }

  /**
   * @brief Set the accept-rate target for step-size estimation.
   *
   * @param[in] v The accept-rate target.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the accept rate is not in (0, 1).
   */
  WarmupConfigBuilder& step_accept_rate_target(double v) {
    detail::validate_probability(v, "step_accept_rate_target");
    cfg_.step_accept_rate_target_ = v;
    return *this;
  }

  /**
   * @brief Set the step size learning rate.
   *
   * @param[in] v The step size learning rate.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the learning rate is not finite and
   * positive.
   */
  WarmupConfigBuilder& step_learning_rate(double v) {
    detail::validate_finite_positive(v, "step_learning_rate");
    cfg_.step_learning_rate_ = v;
    return *this;
  }

  /**
   * @brief Set the gradient decay for the step-size estimator.
   *
   * @param[in] v The gradient decay for the step-size estimator.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay is not in (0, 1).
   */
  WarmupConfigBuilder& step_gradient_decay(double v) {
    detail::validate_probability(v, "step_gradient_decay");
    cfg_.step_gradient_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the squared gradient decay for the step-size estimator.
   *
   * @param[in] v The squared gradient decay for the step-size estimator.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay is not in (0, 1).
   */
  WarmupConfigBuilder& step_sq_gradient_decay(double v) {
    detail::validate_probability(v, "step_sq_gradient_decay");
    cfg_.step_sq_gradient_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the step-size estimator stabilization term.
   *
   * @param[in] v The step-size estimator stabilization term.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the stabilization is not finite and
   * positive.
   */
  WarmupConfigBuilder& step_stabilization(double v) {
    detail::validate_finite_positive(v, "step_stabilization");
    cfg_.step_stabilization_ = v;
    return *this;
  }

  /**
   * @brief Set the learning rate decay exponent for step size.
   *
   * @param[in] v The learning rate decay exponent.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the decay exponent is not in (0, 1).
   */
  WarmupConfigBuilder& step_learn_rate_decay(double v) {
    detail::validate_probability(v, "step_learn_rate_decay");
    cfg_.step_learn_rate_decay_ = v;
    return *this;
  }

  /**
   * @brief Set the stride between publishing statistics for convergence
   * monitoring.
   *
   * @param[in] v The stride for publishing statistics for convergence
   * monitoring.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the stride is not positive.
   */
  WarmupConfigBuilder& publish_stride(std::size_t v) {
    detail::validate_positive(v, "publish_stride");
    cfg_.publish_stride_ = v;
    return *this;
  }

  /**
   * @brief Set the iteration period between chain threads yielding.
   *
   * @param[in] v The iteration period between chain threads yielding.
   * @return This builder for chaining.
   * @throw std::invalid_argument If the yield period is not positive.
   */
  WarmupConfigBuilder& yield_period(std::size_t v) {
    detail::validate_positive(v, "yield_period");
    cfg_.yield_period_ = v;
    return *this;
  }

  /**
   * @brief Return the warmup configuration.
   *
   * @return The warmup configuration.
   */
  WarmupConfig build() { return cfg_; }

 private:
  WarmupConfig cfg_;
};

/**
 * @internal @brief Print the tuning parameters specified by the
 * warmup configuration to the output stream.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The sampling configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const WarmupConfig& cfg) {
  out << "WarmupConfig\n"
      << "  min_iter                 = " << cfg.min_iter() << "\n"
      << "  max_iter                 = " << cfg.max_iter() << "\n"
      << "  step_size_converge_tol   = " << cfg.step_size_converge_tol() << "\n"
      << "  mass_converge_tol        = " << cfg.mass_converge_tol() << "\n"
      << "  mass_init_count          = " << cfg.mass_init_count() << "\n"
      << "  mass_additive_smoothing  = " << cfg.mass_additive_smoothing()
      << "\n"
      << "  max_macro_steps_target   = " << cfg.max_macro_steps_target() << "\n"
      << "  step_accept_rate_target  = " << cfg.step_accept_rate_target()
      << "\n"
      << "  step_learning_rate       = " << cfg.step_learning_rate() << "\n"
      << "  step_gradient_decay      = " << cfg.step_gradient_decay() << "\n"
      << "  step_sq_gradient_decay   = " << cfg.step_sq_gradient_decay() << "\n"
      << "  step_stabilization       = " << cfg.step_stabilization() << "\n"
      << "  step_learn_rate_decay    = " << cfg.step_learn_rate_decay() << "\n"
      << "  publish_stride           = " << cfg.publish_stride() << "\n"
      << "  yield_period             = " << cfg.yield_period() << "\n";
  return out;
}

/**
 * @brief A class to hold the configuration for the Walnuts sampler.
 */
class SamplingConfig {
 public:
  /**
   * @brief Return the minimum number of sampling iterations.
   *
   * @return The minimum number of sampling iterations.
   */
  std::size_t min_iter() const noexcept { return min_iter_; }

  /**
   * @brief Return the maximum number of sampling iterations.
   *
   * @return The maximum number of sampling iterations.
   */
  std::size_t max_iter() const noexcept { return max_iter_; }

  /**
   * @brief Return the maximum number of trajectory doublings
   * for Nuts.
   *
   * @return The maximum number of trajectory doublings.
   */
  std::size_t max_trajectory_doublings() const noexcept {
    return max_trajectory_doublings_;
  }

  /**
   * @brief Return the maximum number of stepsize halvings
   * for Nuts.
   *
   * @return The maximum number of trajectory doublings.
   */
  std::size_t max_step_halvings() const noexcept { return max_step_halvings_; }

  /**
   * @brief Return the maximum error in the Hamiltonian allowed for Walnutpie.
   *
   * @return The maximum error in the Hamiltonian allowed for Walnutpie.
   */
  double max_hamiltonian_error() const noexcept {
    return max_hamiltonian_error_;
  }

  /**
   * @brief Return the minimum number of micro steps per macro step.
   *
   * @return The minimum number of micro steps per macro step.
   */
  std::size_t min_micro_steps() const noexcept { return min_micro_steps_; }

  /**
   * @brief Return the convergence tolerance for the R-hat statistic.
   *
   * @return The convergence tolerance for the R-hat statistic.
   */
  double rhat_converge_tol() const noexcept { return rhat_converge_tol_; }

 private:
  friend class SamplingConfigBuilder;

  SamplingConfig() = default;

  std::size_t min_iter_ = 50;
  std::size_t max_iter_ = 1000;
  std::size_t max_trajectory_doublings_ = 5;
  std::size_t max_step_halvings_ = 5;
  double max_hamiltonian_error_ = 0.5;
  std::size_t min_micro_steps_ = 1;
  double rhat_converge_tol_ = 1.01;
};

/**
 * @brief The builder for sampling configurations.
 *
 * An example use would be:
 * @code
 * SampleConfigBuilder(50u, 100u)
 *     .max_step_halvings(4u)
 *     .min_micro_steps(2u)
 *     .build();
 * @endcode
 */
class SamplingConfigBuilder {
 public:
  /**
   * @brief Set the minimum and maximum number of iterations.
   *
   * @param[in] min_iter The minimum number of iterations.
   * @param[in] max_iter The maximum number of iterations.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the minimum number of iterations
   * is greater than the maximum number of iterations.
   */
  SamplingConfigBuilder& min_max_iter(std::size_t min_iter,
                                      std::size_t max_iter) {
    if (min_iter > max_iter) {
      throw std::invalid_argument("min_iter must be <= max_iter");
    }
    cfg_.min_iter_ = min_iter;
    cfg_.max_iter_ = max_iter;
    return *this;
  }

  /**
   * @brief Set the maximum number of trajectory doublings.
   *
   * @param[in] v The maximum number of trajectory doublings.
   * @return A reference to this builder for chaining.
   */
  SamplingConfigBuilder& max_trajectory_doublings(std::size_t v) noexcept {
    cfg_.max_trajectory_doublings_ = v;
    return *this;
  }

  /**
   * @brief Set the maximum number of step size halvings.
   *
   * @param[in] v The maximum number of step size halvings.
   * @return A reference to this builder for chaining.
   */
  SamplingConfigBuilder& max_step_halvings(std::size_t v) noexcept {
    cfg_.max_step_halvings_ = v;
    return *this;
  }

  /**
   * @brief Set the maximum error in the Hamiltonian for Walnutpie.
   *
   * @param[in] v The maximum error in the Hamiltonian for Walnutpie.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the error is not finite and positive.
   */
  SamplingConfigBuilder& max_hamiltonian_error(double v) {
    detail::validate_finite_positive(v, "max_hamiltonian_error");
    cfg_.max_hamiltonian_error_ = v;
    return *this;
  }

  /**
   * @brief Set the minimum number of micro steps per macro step.
   *
   * @param[in] v The minimum number of micro steps per macro step.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the minimum number of steps is not
   * positive.
   */
  SamplingConfigBuilder& min_micro_steps(std::size_t v) {
    detail::validate_positive(v, "min_micro_steps");
    cfg_.min_micro_steps_ = v;
    return *this;
  }

  /**
   * @brief Set the R-hat convergence tolerance.
   *
   * @param[in] v The R-hat convergence tolerance.
   * @return A reference to this builder for chaining.
   * @throw std::invalid_argument If the tolerance is not finite and > 1.
   */
  SamplingConfigBuilder& rhat_converge_tol(double v) {
    detail::validate_finite_gt1(v, "rhat_convergence_tol");
    cfg_.rhat_converge_tol_ = v;
    return *this;
  }

  /**
   * @brief Return the sampling configuration.
   *
   * @return The sampling configuration.
   */
  SamplingConfig build() { return cfg_; }

 private:
  SamplingConfig cfg_;
};

/**
 * @internal @brief Print the tuning parameters specified by the sampling
 * configuration to the output stream.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The sampling configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const SamplingConfig& cfg) {
  out << "SamplingConfig\n"
      << "  min_iter                   = " << cfg.min_iter() << "\n"
      << "  max_iter                   = " << cfg.max_iter() << "\n"
      << "  max_trajectory_doublings   = " << cfg.max_trajectory_doublings()
      << "\n"
      << "  max_step_halvings          = " << cfg.max_step_halvings() << "\n"
      << "  max_hamiltonian_error      = " << cfg.max_hamiltonian_error()
      << "\n"
      << "  min_micro_steps            = " << cfg.min_micro_steps() << "\n"
      << "  rhat_converge_tol          = " << cfg.rhat_converge_tol() << "\n";
  return out;
}

/**
 * @brief Encapsulated configuration for Walnutpie.
 *
 * Walnuts configurations include initialization, warmup, and sampling
 * configurations.
 */
class WalnutsConfig {
 public:
  /**
   * @brief Construct a Walnuts configuration given the component
   * configurations.
   *
   * The arguments will be moved if they are rvalues and copied if lvalues.
   *
   * @param[in] init The initialization configuration.
   * @param[in] warmup The warmup configuration.
   * @param[in] sampling The sampling configuration.
   */
  WalnutsConfig(InitConfig init, WarmupConfig warmup, SamplingConfig sampling)
      : init_(std::move(init)),
        warmup_(std::move(warmup)),
        sampling_(std::move(sampling)) {};

  /**
   * @brief Return the initialization configuration.
   *
   * @return The initialization configuration.
   */
  const InitConfig& init() const noexcept { return init_; }

  /**
   * @brief Return the warmup configuration.
   *
   * @return The warmup configuration.
   */
  const WarmupConfig& warmup() const noexcept { return warmup_; }

  /**
   * @brief Return the sampling configuration.
   *
   * @return The sampling configuration.
   */
  const SamplingConfig& sampling() const noexcept { return sampling_; }

 private:
  /** The initialization configuration for all chains. */
  InitConfig init_;

  /** The warmup configuration shared by all chains. */
  WarmupConfig warmup_;

  /** The sampling configuration shared by all chains for warmup and sampling.
   */
  SamplingConfig sampling_;
};

/**
 * @brief Print the Walnuts configuration.
 *
 * This just delegates to printing the initialization, warmup, and
 * sampling configurations separated by newlines.
 *
 * @param[in,out] out Output stream to which configuration is printed.
 * @param[in] cfg The Walnuts configuration.
 * @return The output stream for chained calls.
 */
inline std::ostream& operator<<(std::ostream& out, const WalnutsConfig& cfg) {
  out << cfg.init() << "\n" << cfg.warmup() << "\n" << cfg.sampling() << "\n";
  return out;
}

}  // namespace walnutpie
