#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"

namespace walnutpie::detail {

#if defined(__has_attribute) && __has_attribute(always_inline)
#define WALNUTPIE_STRONG_INLINE [[gnu::always_inline]] inline
#else
#define WALNUTPIE_STRONG_INLINE inline
#endif

#ifdef __APPLE__
#include <pthread/qos.h>
WALNUTPIE_STRONG_INLINE void interactive_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // best
}
WALNUTPIE_STRONG_INLINE void initiated_qos() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);  // next best
}
#else
WALNUTPIE_STRONG_INLINE void interactive_qos() {}
WALNUTPIE_STRONG_INLINE void initiated_qos() {}
#endif

/**
 * @brief A conservative constant destructive interference size.
 *
 * The std::hardware_destructive_interference_size is not universally supported
 * and can underreport when it is supported. 128 is safe for ARM and Intel
 * hardware.
 */
inline constexpr std::size_t FALSE_SHARING_GUARD_SIZE = 128;

/**
 * @brief Proposal update schemes for MCMC transitions.
 */
enum class Update {
  Barker,    /**< Use Barker's acceptance rule (proportional to density). */
  Metropolis /**< Use standard Metropolis acceptance rule */
};

/**
 * @brief Time direction of Hamiltonian simulation.
 */
enum class Direction {
  Backward, /**< Step backward in time. */
  Forward   /**< Step forward in time. */
};

/**
 * @brief A type definition for constructing `Direction::Backward` constants.
 */
using Backward_t = std::integral_constant<Direction, Direction::Backward>;

/**
 * @brief A type definition for constructing `Direction::Forward` constants.
 */
using Forward_t = std::integral_constant<Direction, Direction::Forward>;

/**
 * @brief A class encapsulating the randomizers needed for Hamiltonian Monte
 * Carlo.
 *
 * @tparam RNG The type of the base random number generator.
 */
template <std::uniform_random_bit_generator RNG>
class Random {
 public:
  /**
   * @brief Construct a randomizer with the specified base random number
   * generator.
   *
   * The base generator is held as a reference and used for all of the
   * generation. Thus it must be kept in scope as the instance constructed with
   * it is used. The base generator may be shared with other applications.
   *
   * @param[in,out] rng The base random number generator.
   */
  explicit Random(RNG& rng) noexcept
      : rng_(rng), unif_(0.0, 1.0), binary_(0.5), normal_(0.0, 1.0) {}

  /**
   * @brief Return a number between 0 and 1 generated uniformly at random.
   *
   * The base random number generator is used to generate from a
   * `uniform([0, 1])` distribution.
   *
   * @return A number between 0 and 1 generated uniformly at random.
   */
  double uniform_real_01() { return unif_(rng_); }

  /**
   * @brief Return `true` or `false` uniformly at random.
   *
   * The base random number generator is used to generate from
   * a `uniform({0, 1})` distribution.
   *
   * @return A boolean value generated uniformly at random.
   */
  bool uniform_binary() { return binary_(rng_); }

  /**
   * @brief Return a vector of values generated according to a
   * standard normal distribution.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @return An expression template for a standard normal vector.
   */
  auto standard_normal(Eigen::Index n) {
    return Eigen::VectorXd::NullaryExpr(
        n, [this](Eigen::Index /*i*/) { return this->normal_(rng_); });
  }

  /**
   * @brief Write a vector of random standard normal variables into the out
   * vector.
   *
   * The base random number generator is used to generate each
   * component independently from a `normal(0, 1)` distribution.
   *
   * @param[in] n The size of the vector generated.
   * @param[out] out The output vector.
   */
  void standard_normal(Eigen::Index n, Eigen::VectorXd& out) {
    out = standard_normal(n);
  }

  /**
   * @brief Return a reference to the base random number generator.
   *
   * @return The base random number generator.
   */
  RNG& rng() noexcept { return rng_; }

 private:
  /** The base random number generator reference. */
  RNG& rng_;  // not std::reference_wrapper to prevent copying

  /** The `uniform([0, 1])` random number generator. */
  std::uniform_real_distribution<double> unif_;

  /** The `uniform({0, 1})` random number generator. */
  std::bernoulli_distribution binary_;

  /** The `normal(0, 1)` random number generator. */
  std::normal_distribution<double> normal_;
};

/**
 * @brief Return the log of the sum of the exponentiated arguments.
 *
 * The mathematical definition is `log_sum_exp(x1, x2) = log(exp(x1) +
 * exp(x2))`. The implementation is high precision and numerically stable.
 *
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The log of the sum of the exponentiations of the arguments.
 */
inline double log_sum_exp(const double& x1, const double& x2) {
  auto m = std::fmax(x1, x2);
  if (std::isnan(x1) || std::isnan(x2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (std::isinf(m) || std::isnan(x1 + x2)) {
    return std::fmax(x1, x2);
  }
  return m + std::log(std::exp(x1 - m) + std::exp(x2 - m));
}

/**
 * @brief Return the log of the sum of the components of the argument
 * exponentiated.
 *
 * The mathematical definition is `log_sum_exp(v) = log(sum(exp(x))`.
 * The implementation is high precision and numerically stable.
 *
 * @param[in] x The vector argument.
 * @return The log of the sum of the exponentiation of the vector's components.
 */
inline double log_sum_exp(const Eigen::VectorXd& x) {
  using std::log;
  if (x.size() == 0) {  // Eigen triggers assert on empty .maxCoeff()
    return -std::numeric_limits<double>::infinity();
  }
  double m = x.maxCoeff();
  if (std::isinf(m)) {
    return m;  // x[i] all -inf or all +inf; ow NaN
  }
  return m + log((x.array() - m).exp().sum());
}

/**
 * @brief Return the unnormalized log density of the specified momentum given
 * the specified inverse mass matrix diagonal.
 *
 * The unnormalized log density is the negative kinetic energy.
 *
 * The formula is `-0.5 * rho' .* inv_mass * rho`, which for diagonals works
 * out to `-0.5 * rho**2 * inv_mass` elementwise.
 *
 * @param[in] rho Vector of momenta.
 * @param[in] inv_mass_diag The diagonal of the diagonal inverse mass matrix.
 * @return The log density of the momentum.
 */
inline double logp_momentum(const Eigen::VectorXd& rho,
                            const Eigen::VectorXd& inv_mass_diag) {
  return -0.5 * (inv_mass_diag.array() * rho.array().square()).sum();
}

/**
 * @brief Return the difference in log density (negative Hamiltonian)
 * between the intial position and momentum and the result of taking
 * one leapfrog step with the specified inverse mass matrix and step size.
 *
 * As an internal function called by an algorithm that controls the
 * inverse mass matrix and step size, there is no error checking to
 * enusre `theta`, `rho`, and `inv_M` are all the same size, that
 * `inv_M` has all finite positive entries, and that the step size is
 * finite and positive.
 *
 * @tparam F The type of the log density and gradient function.
 * @param[in] theta The starting position.
 * @param[in] rho The starting momentum.
 * @param[in] M The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The step size.
 */
template <LogpGrad F>
double leapfrog_error(const F& logp_grad, const Eigen::VectorXd& theta,
                      const Eigen::VectorXd& rho, const Eigen::VectorXd& inv_M,
                      double step) {
  Eigen::VectorXd grad;
  double logp;
  logp_grad(theta, logp, grad);
  logp += detail::logp_momentum(rho, inv_M);
  Eigen::VectorXd rho_star = rho + 0.5 * step * grad;
  Eigen::VectorXd theta_star =
      theta + step * (inv_M.array() * rho_star.array()).matrix();
  double logp_star;
  logp_grad(theta_star, logp_star, grad);
  rho_star = rho_star + 0.5 * step * grad;
  logp_star += detail::logp_momentum(rho_star, inv_M);
  double diff = logp_star - logp;
  return diff;
}

/**
 * @brief Return the adapted step size for the specified initial
 * step size, given an initial position and inverse mass matrix.
 *
 * The algorithm randomly generates a momentum. Then until
 * the Metropolis acceptance rate is below 0.9, it continues
 * to double the step size. Then until the Metropolis accept
 * rate is above 0.6, it continues to divide it by sqrt(2).
 * This is a slightly more fine-grained version of the heuristic
 * step size initialization used by Nuts.
 *
 * There is no error testing here for consistency because this
 * function is called from a controlled setting.
 *
 * @tparam RNG Type of the base random number generator.
 * @tparam F Type of the log density and gradient function.
 * @param[in] rng The base random number generator.
 * @param[in] logp_grad The log density and gradient function.
 * @param theta The initial position.
 * @param M The diagonal of the diagonal mass matrix.
 * @param step The initial step size guess.
 * @param D The dimensioanlity
 * @return The adapted step size.
 */
template <std::uniform_random_bit_generator RNG, LogpGrad F>
double adapt_step(RNG& rng, const F& logp_grad, const Eigen::VectorXd& theta,
                  const Eigen::VectorXd& M, double step, std::size_t D) {
  detail::Random<RNG> rand(rng);
  Eigen::VectorXd inv_M = M.array().inverse().matrix();
  Eigen::VectorXd rho =
      (rand.standard_normal(static_cast<Eigen::Index>(D)).array() *
       M.array().sqrt())
          .matrix();
  while (detail::leapfrog_error(logp_grad, theta, rho, inv_M, step) >
         std::log(0.9)) {
    step *= 2;
  }
  while (detail::leapfrog_error(logp_grad, theta, rho, inv_M, step) <
         std::log(0.6)) {
    step *= std::sqrt(0.5);
  }
  return step;
}

/**
 * @brief A wrapper for a log density and gradient function that traps
 * exceptions.
 *
 * @tparam F Type of underlying log density and gradient function.
 */
template <LogpGrad F, ErrorCallback H>
class NoExceptLogpGrad {
 public:
  /**
   * @brief Construct a log density and gradient function from a base
   * log density and gradient function.
   *
   * The log density and gradient function will be stored as a
   * constant reference.
   *
   * @param[in] logp_grad The base log density and gradient function, called
   * back.
   * @param[in] handler The sample handler, used when exceptions are thrown.
   */
  NoExceptLogpGrad(const F& logp_grad, H& handler)
      : logp_grad_(std::cref(logp_grad)), handler_(handler) {}

  /**
   * @brief Given the specified position, set the log density and
   * gradient.
   *
   * @param[in] x The position vector.
   * @param[out] logp The log density to set.
   * @param[out] grad The gradient to set.
   */
  void operator()(const Eigen::VectorXd& x, double& logp,
                  Eigen::VectorXd& grad) const noexcept {
    try {
      logp_grad_.get()(x, logp, grad);
    } catch (const std::exception& e) {
      handler_.get().on_logp_exception(x, e);
      // logp_grad failure equivalent to -inf log density
      logp = -std::numeric_limits<double>::infinity();
      grad.setZero(x.size());
    }
  }

  /** The log density and gradient function. */
  const std::reference_wrapper<const F> logp_grad_;
  const std::reference_wrapper<H> handler_;
};

/**
 * @brief Return the gradient of the log density at the specified position.
 *
 * @tparam F The type of the target log density/gradient function.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] theta The position at which to evaluate the gradient.
 * @return The gradient of the log density at `theta`.
 */
inline Eigen::VectorXd grad(const LogpGrad auto& logp_grad,
                            const Eigen::VectorXd& theta) {
  Eigen::VectorXd g;
  double logp;
  logp_grad(theta, logp, g);
  return g;
}

/**
 * @brief Returns the L2 relative distance between the two vectors
 * scaled by the second vector.

 * The computation is `norm((a - b) / b)`.
 *
 * @param[in] a The test vector.
 * @param[in] b The baseline vector.
 * @return The relative difference
 */
inline double l2_rel_diff(const Eigen::VectorXd& a,
                          const Eigen::VectorXd& b) noexcept {
  return ((a - b).array() / b.array()).matrix().norm();
}

/**
 * @brief Return the sum of the sizes in the vector.
 *
 * @param[in] xs The vector to sum.
 * @return The sum.
 */
inline std::size_t sum(const std::vector<std::size_t>& xs) noexcept {
  return std::transform_reduce(xs.begin(), xs.end(), std::size_t(0),
                               std::plus<>{}, std::identity());
}

/**
 * @brief Return the bias-adjusted sample variance estimate.
 *
 * @param[in] xs The vector whose variance is required.
 * @return The variance.
 */
inline double variance(const Eigen::VectorXd& xs) noexcept {
  return (xs.array() - xs.mean()).square().sum() /
         static_cast<double>((xs.size() - 1));
}

/**
 * @brief Return `true` if the log density and gradient function returns
 * a finite log density and a gradient with finite elements.
 *
 * @tparam F The type of the log density and gradient function.
 * @param logp_grad The log density and gradient function.
 * @param theta The position to evaluate.
 * @return `true` if the log density and gradients are finite at the position.
 */
template <LogpGrad F>  
inline bool logp_grad_is_finite(const F& logp_grad,
				const Eigen::VectorXd theta) {
  Eigen::VectorXd grad;
  double lp;
  logp_grad(theta, lp, grad);
  return std::isfinite(lp) && grad.allFinite();
}    

/** 
 * @brief Return the geometric mean of the specified sequence of vectors.
 *
 * @param xs The vectors to average.
 * @return The geometric mean of the vectors.
 */
inline Eigen::VectorXd geometric_mean(const std::vector<Eigen::VectorXd>& xs) {
  Eigen::ArrayXd sum_log = Eigen::ArrayXd::Zero(xs.front().size());
  for (const auto& x : xs) sum_log += x.array().log();
  return (sum_log / static_cast<double>(xs.size())).exp().matrix();
}

}  // namespace walnutpie::detail
