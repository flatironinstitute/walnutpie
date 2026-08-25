#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include "walnutpie/validate.hpp"

namespace walnutpie::detail {

/**
 * @brief Accumulator for online mean and smaple variance calculations.
 *
 * Welford's algorithm stores sufficient statistics with which to
 * compute a running mean and sample variance. The accumulator stores
 * only three sufficient statistics: a `std::size_t` and two `double`
 * values. The algorithm is more numerically stable for variance
 * calculations than the naive algorithm.
 */
class WelfordAccumulator {
 public:
  /**
   * @brief Construct an accumulator with no observed values.
   */
  WelfordAccumulator() : n_(0), mean_(0.0), M2_(0.0) {}

  /**
   * @brief Observe a value.
   *
   * @param[in] x The observed value.
   */
  void observe(double x) {
    ++n_;
    const double delta = x - mean_;
    mean_ += delta / static_cast<double>(n_);
    const double delta2 = x - mean_;
    M2_ += delta * delta2;
  }

  /**
   * @brief Return the number of values observed.
   *
   * @return The number of values observed.
   */
  std::size_t count() const { return n_; }

  /**
   * @brief Return the mean of all of the values observed, or 0
   * if no values have been observed.
   *
   * @return The mean of the observed values.
   */
  double mean() const { return mean_; }

  /**
   * @brief Return the sample variance of the observed values.
   *
   * The sample variance is the unbiased estimator of variance.
   * It divides by number of observations minus one. Thus if
   * there have been fewer than two observations, the sample
   * variance is undefined and `NaN` will be returned.
   *
   * @return The sample variance of the observed values.
   */
  double sample_variance() const {
    return n_ > 1 ? (M2_ / static_cast<double>(n_ - 1))
                  : std::numeric_limits<double>::quiet_NaN();
  }

  /**
   * @brief Reset the accumulator to its initial state of having seen
   * zero observations.
   */
  void reset() {
    n_ = 0;
    mean_ = 0.0;
    M2_ = 0.0;
  }

 private:
  std::size_t n_;
  double mean_;
  double M2_;
};

/**
 * @brief An accumulator estimating discounted means and variances online.
 *
 * The `observe()` method receives vector value updates and maintains a
 * running estimate of discounted means and variances. Historical counts
 * are discounted by the multiplying by the discount factor before each
 * new observation is added (with a count of one). 1 does no discounting
 * and 0 completely forgets the past.
 *
 * The implementation uses a weighted variant of Welford's algorithm that
 * discounts past observations. It requires a constant memory of size
 * proportional to the dimensionality of the observed vectors (i.e.,
 * O(`dim`)). Each of its methods runs in time proportional to the size
 * of the update vectors (i.e., O(`dim`)). Arithmetic is stable
 * following the original Welford accumulator, to which it reduces
 * when `discount_factor = 1`.
 *
 * After initialization and updating with `N` vectors
 * `y[0], ..., y[N - 1]`, the weight for vector `y[n]` is
 * ```
 * weight[n] = discount_factor^(N - n - 1).
 * ```
 *
 * The discounted mean is calculated in the usual way for weighted
 * averages,
 * ```
 * mean = sum(y .* weight) / sum(weight),
 * ```
 * where `.*` is elementwise product.
 *
 *
 * The discounted mean is then used to estimate the discounted variance,
 *
 * ```
 * var = sum(weight .* (y - mean)^2) / sum(weight).
 * ```
 */
class OnlineMoments {
 public:
  /**
   * @brief Construct a default online estimator of size zero.
   */
  OnlineMoments() : weight_(0) {}

  /**
   * @brief Construct an online estimator of moments with the
   * specified discount factor and initialization.
   *
   * The initialization specifies the initial mean and the initial
   * variance, assigning them a weight that is interpreted as if the
   * initial mean and variance were the result of a count of
   * `init_weight` observations.
   *
   * @param[in] init_weight Weight (in number of draws) of initial mean
   * and variance (positive).
   * @param[in] init_mean Initial mean.
   * @param[in] init_variance Initial variance.
   * @throw std::invalid_argument If `discount_factor` is not in (0, 1).
   * @throw std::invalid_argument If the initial weight is not finite and
   * positive.
   * @throw std::invalid_argument If the initial mean and variance are not
   * the same size.
   */
  OnlineMoments(double init_weight, const Eigen::VectorXd& init_mean,
                const Eigen::VectorXd& init_variance)
      : weight_(init_weight),
        mean_(init_mean),
        sum_sq_dev_(init_weight * init_variance) {
    detail::validate_positive(init_weight, "init_weight");
    detail::validate_same_size(init_mean, init_variance, "init_mean",
                               "init_variance");
  }

  /**
   * @brief Set the discount factor for previous observations to the specified
   * value.
   *
   * @param[in] discount_factor The discount factor.
   * @throw std::invalid_argument If the discount factor is not in (0, 1).
   */
  void set_discount_factor(double discount_factor) {
    detail::validate_probability_inclusive(discount_factor, "discount_factor");
    discount_factor_ = discount_factor;
  }

  /**
   * @brief Update this accumulator with the specified observation.
   *
   * The observed value `y` is assigned a weight (or count) of 1, and
   * the weights of the past observations are discounted by the discount
   * factor.
   *
   * @tparam Derived The type of matrix underlying the observation.
   * @param[in] y The observed vector.
   * @pre y.size() == mean().size()
   */
  template <typename Derived>
  void observe(const Eigen::MatrixBase<Derived>& y) {
    // Materialize the deviation BEFORE updating mean_: `auto delta = ...`
    // would bind a lazy expression referencing mean_, and the sum_sq_dev_
    // line would then evaluate (y - mean_new)^2 instead of the Welford
    // cross-term delta_old * (y - mean_new) this class documents.
    const Eigen::VectorXd delta = (y - mean_).eval();
    weight_ = discount_factor_ * weight_ + 1;
    mean_ += delta / weight_;
    sum_sq_dev_.noalias() =
        discount_factor_ * sum_sq_dev_ + delta.cwiseProduct(y - mean_);
  }

  /**
   * @brief Set the discount factor, then update with the specified observation.
   *
   * This is a convenience method to call `set_discount_factor(discount_factor)`
   * and `observe(y)`.
   *
   * @tparam Derived The type of matrix underlying the observation.
   * @param[in] discount_factor The discount factor.
   * @param[in] y The observed vector.
   * @throw discount_factor > 0 && discount_factor <= 1
   * @pre y.size() == mean().size()
   */
  template <typename Derived>
  void discount_observe(double discount_factor,
                        const Eigen::MatrixBase<Derived>& y) {
    set_discount_factor(discount_factor);
    observe(y);
  }

  /**
   * @brief Return the estimate of the mean.
   *
   * @return The mean estimate.
   */
  const Eigen::VectorXd& mean() const noexcept { return mean_; }

  /**
   * @brief Return the maximum likelihood estimate of the variance, or
   * a vector of 1 values if there have been no observations.
   *
   * @return The variance estimate.
   */
  Eigen::VectorXd variance() const {
    if (!(weight_ > 0)) {
      return Eigen::VectorXd::Ones(mean_.size());
    }
    return sum_sq_dev_ / weight_;
  }

 private:
  /** The discount factor applied to the weights of previous observations.
   *
   * Gets reset before it is used in discounted Welford algorithm.
   */
  double discount_factor_ = std::numeric_limits<double>::quiet_NaN();

  /** The combined weight of all previous observations. */
  double weight_;

  /** The current mean estimate */
  Eigen::VectorXd mean_;

  /** The sum of weighted squared deviations from the mean. */
  Eigen::VectorXd sum_sq_dev_;
};

}  // namespace walnutpie::detail
