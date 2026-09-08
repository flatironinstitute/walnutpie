#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include "walnutpie/concepts.hpp"

namespace walnutpie::detail {

template <typename Derived>
inline Eigen::RowVectorXd col_means(const Eigen::MatrixBase<Derived>& draws) {
  return draws.colwise().mean();
}

constexpr inline Eigen::Index strip_factor(Eigen::Index m,
                                           Eigen::Index factor) {
  while (m % factor == 0) {
    m /= factor;
  }
  return m;
}

/**
 * @brief Return smallest number greater than or equal to
 * input that is evenly divisible by 2, 3, and 5.
 *
 * @param[in] n Original size.
 * @return Original size padded for divisibility.
 */
constexpr inline Eigen::Index fft_next_good_size(Eigen::Index n) {
  if (n <= 2) {
    return 2;
  }
  for (; true; ++n) {
    auto m = n;
    m = strip_factor(m, 2);
    m = strip_factor(m, 3);
    m = strip_factor(m, 5);
    if (m <= 1) {
      return n;
    }
  }
}

// FFT not thread safe
inline void autocovariance_col(const Eigen::Ref<const Eigen::VectorXd>& y,
                               Eigen::Ref<Eigen::VectorXd> ac,
                               Eigen::FFT<double>& fft,
                               Eigen::VectorXd& padded_signal,
                               Eigen::VectorXcd& freq_vec,
                               Eigen::VectorXcd& ac_tmp) {
  // TODO: evaluate the following optimization
  // fft.SetFlag(fft.HalfSpectrum);
  Eigen::Index M2 = padded_signal.size();
  Eigen::Index N = y.size();
  padded_signal.tail(M2 - N).setZero();
  padded_signal.head(N) = y.array() - y.mean();
  fft.fwd(freq_vec, padded_signal);
  freq_vec = freq_vec.cwiseAbs2();
  fft.inv(ac_tmp, freq_vec);
  // biased estimate recommended by Geyer (1992) divides by N
  ac_tmp /= N;
  ac = ac_tmp.head(N).real();
}

// FFT not thread safe
inline Eigen::MatrixXd autocovariance_chain(
    const Eigen::Ref<const Eigen::MatrixXd>& chain, Eigen::FFT<double>& fft) {
  Eigen::Index N = chain.rows();
  Eigen::Index D = chain.cols();
  Eigen::MatrixXd acor(N, D);
  Eigen::Index M = detail::fft_next_good_size(N);
  Eigen::Index M2 = 2 * M;
  Eigen::VectorXd padded_signal(M2);
  Eigen::VectorXcd freq_vec(M2);
  Eigen::VectorXcd ac_tmp(M2);
  for (Eigen::Index d = 0; d < D; ++d) {
    autocovariance_col(chain.col(d), acor.col(d), fft, padded_signal, freq_vec,
                       ac_tmp);
  }
  return acor;
}

template <typename Derived>
inline Eigen::RowVectorXd sample_variance(
    const Eigen::MatrixBase<Derived>& draws, const Eigen::RowVectorXd& mean) {
  // would be numerically more robust with Welford's algorithm
  return ((draws.rowwise() - mean).array().square().colwise().sum() /
          (draws.rows() - 1));
}

template <typename Derived>
inline Eigen::RowVectorXd sample_variance(
    const Eigen::MatrixBase<Derived>& draws) {
  return sample_variance(draws, detail::col_means(draws));
}

}  // namespace walnutpie::detail

namespace walnutpie {

/**
 * @brief A sequence of Markov chains of possibly varying lengths
 * stored individually as matrices of draws.
 *
 * This implementation keeps a constant reference to the chains
 * provided in the constructor, so they must outlive the constructed
 * object's use.
 */
class MarkovChainsSplit {
 public:
  /**
   * @brief Construct a sequence of Markov chains.
   *
   * The result holds a constant reference to the chains, so the
   * argument `chains` must outlive the constructed object's use.
   *
   * @param[in] chains Markov chains.
   * @throw std::invalid_argument If the sequence of chains is length zero.
   * @throw std::invalid_argument If all chains do not contain at least one
   * draw.
   * @throw std::invalid_argument If all chains do not have the same number of
   * columns.
   */
  explicit MarkovChainsSplit(const std::vector<Eigen::MatrixXd>& chains)
      : chains_(chains) {
    if (chains.empty()) {
      throw std::invalid_argument("chains cannot be empty.");
    }
    Eigen::Index dims = chains[0].cols();
    for (const auto& chain : chains) {
      if (chain.rows() == 0) {
        throw std::invalid_argument("chains must have at least one draw.");
      }
      if (chain.cols() != dims) {
        throw std::invalid_argument(
            "all chains must have same number of columns.");
      }
    }
  }

  /**
   * @brief Return the number of chains.
   *
   * @return The number of chains.
   */
  std::size_t num_chains() const noexcept { return chains().size(); }

  /**
   * @brief Return the total number of draws across all chains.
   *
   * @return The total number of draws.
   */
  std::size_t num_draws() const noexcept {
    std::size_t sum = 0;
    for (const auto& chain : chains()) {
      sum += static_cast<std::size_t>(chain.rows());
    }
    return sum;
  }

  /**
   * @brief Return the dimensionality of the draws.
   *
   * @return The dimensionality of the draws.
   */
  std::size_t dims() const noexcept {
    auto num_dims = chains()[0].cols();
    return static_cast<std::size_t>(num_dims);
  }

  /**
   * @brief Return an immutable view of the specified chain.
   *
   * The return is an expression template that will hold a reference
   * to the chain managed by this class.
   *
   * @param[in] m The index of the chain.
   * @return A view of the specified chain.
   * @throw std::out_of_range If the index is greater than or equal to
   * the number of chains.
   */
  Eigen::Ref<const Eigen::MatrixXd> chain_view(std::size_t m) const {
    return chains().at(m);
  }

  /**
   * @brief Return the number of draws in the shortest chain.
   *
   * @return The length of the shortest chain.
   */
  Eigen::Index min_chain_size() const noexcept {
    Eigen::Index min_size = std::numeric_limits<Eigen::Index>::max();
    for (const auto& chain : chains()) {
      min_size = std::min(min_size, chain.rows());
    }
    return min_size;
  }

  /**
   * @brief Return all of the draws for the specified dimension.
   *
   * This implementation allocates a vector to return of the
   * appropriate size.
   *
   * @param[in] d The selected dimension.
   * @return The draws for the selecte dimension.
   * @throws std::out_of_range If the index is not between 0 and
   * the number of dimensions minus 1, inclusive.
   */
  Eigen::VectorXd draws(Eigen::Index d) const {
    if (d < 0 || static_cast<std::size_t>(d) >= dims()) {
      throw std::out_of_range("dimension index out of range");
    }
    Eigen::VectorXd drws(num_draws());
    Eigen::Index start = 0;
    for (const auto& chain : chains()) {
      Eigen::Index size = chain.rows();
      drws.segment(start, size) = chain.col(d);
      start += size;
    }
    return drws;
  }

 private:
  const std::vector<Eigen::MatrixXd>& chains() const noexcept {
    return chains_.get();
  }

  std::reference_wrapper<const std::vector<Eigen::MatrixXd>> chains_;
};

static_assert(MarkovChainSequence<MarkovChainsSplit>);

/**
 * @brief A sequence of Markov chains of possibly varying lengths with
 * a single underlying matrix of draws.
 *
 * The underlying single matrix will be contiguous and allows constant-time
 * access to columns and constant-time views of individual chain matrices.
 */
class MarkovChainsUnified {
 public:
  /**
   * @brief Construct an instance with the specified draws
   * and chain sizes.
   *
   * The implementation holds a constant reference to the matrix of
   * draws so that the `draws` matrix must outlive this instance.
   *
   * @param[in] draws The sequence of Markov chains states, one row per draw.
   * @param[in] chain_sizes The sizes of the Markov chains making up the
   * collection.
   * @throw std::invalid_argument If the sum of the chain sizes
   * is not equal to the number of draws.
   */
  MarkovChainsUnified(const Eigen::Ref<const Eigen::MatrixXd>& draws,
                      const std::vector<std::size_t>& chain_sizes)
      : draws_(draws),
        chain_sizes_(chain_sizes.size()),
        chain_starts_(chain_sizes.size()) {
    Eigen::Index total = 0;
    for (std::size_t m = 0; m < chain_sizes.size(); ++m) {
      chain_sizes_[m] = static_cast<Eigen::Index>(chain_sizes[m]);
      chain_starts_[m] = total;
      total += chain_sizes[m];
    }
    if (total != draws.rows()) {
      throw std::invalid_argument(
          "The number of rows in draws and sum of chain_sizes must be equal.");
    }
  }

  /**
   * @brief Return the number of chains.
   *
   * @return The number of chains.
   */
  std::size_t num_chains() const noexcept { return chain_sizes_.size(); }

  /**
   * @brief Return the total number of draws across all chains.
   *
   * @return The total number of draws.
   */
  std::size_t num_draws() const noexcept {
    return static_cast<std::size_t>(draws_.rows());
  }

  /**
   * @brief Return the dimensionality of the draws.
   *
   * @return The dimensionality of the draws.
   */
  std::size_t dims() const noexcept {
    return static_cast<std::size_t>(draws_.cols());
  }

  /**
   * @brief Return an immutable view of the specified chain.
   *
   * The return is an expression template that will hold a reference
   * to the draws.
   *
   * @param[in] m The index of the chain.
   * @return A view of the specified chain.
   * @throw std::out_of_range If the index is greater than or equal to
   * the number of chains.
   */
  Eigen::Ref<const Eigen::MatrixXd> chain_view(std::size_t m) const {
    return draws_.middleRows(chain_starts_.at(m), chain_sizes_.at(m));
  }

  /**
   * @brief Return the number of draws in the shortest chain.
   *
   * @return The length of the shortest chain.
   */
  Eigen::Index min_chain_size() const noexcept {
    return *std::ranges::min_element(chain_sizes_);
  }

  /**
   * @brief Return all of the draws for the specified dimension.
   *
   * The return is an expression template for a constant vector that
   * depends on the draws.
   *
   * @param[in] d The selected dimension.
   * @return The draws for the selecte dimension.
   * @throws std::out_of_range If the index is not between 0 and
   * the number of dimensions minus 1, inclusive.
   *
   */
  Eigen::Ref<const Eigen::VectorXd> draws(Eigen::Index d) const {
    if (d < 0 || d >= static_cast<Eigen::Index>(dims())) {
      throw std::out_of_range("Dimension index must be between 0 and"
                              "number of dimensions minus 1, inclusive.");
    }
    return draws_.col(d);
  }

 private:
  Eigen::Ref<const Eigen::MatrixXd> draws_;
  std::vector<Eigen::Index> chain_sizes_;
  std::vector<Eigen::Index> chain_starts_;
};

static_assert(MarkovChainSequence<MarkovChainsUnified>);

/**
 * @brief Return the sample means of the variables in the chains.
 *
 * The means are calculated for each variable (i.e., each
 * dimension).
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The sample means.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd mean(const MC& chains) {
  Eigen::RowVectorXd total =
      Eigen::RowVectorXd::Zero(static_cast<Eigen::Index>(chains.dims()));
  for (std::size_t m = 0; m < chains.num_chains(); ++m) {
    total += chains.chain_view(m).colwise().sum();
  }
  return total / static_cast<double>(chains.num_draws());
}

/**
 * @brief Return the sample variances of the variables in the chains.
 *
 * The variances are calculated for each variable (i.e., each
 * dimension). The formula divides by the number of draws minus one
 * and thus provides an unbiased estimate of the population variance
 * based on a small sample. If used to calculate the variance of an
 * entire population, it will be biased to the high side.
 *
 * If there is only one draw in a chain, this function will return
 * `Nan`.
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The variances.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd sample_variance(const MC& chains) {
  Eigen::RowVectorXd mu = mean(chains);
  Eigen::RowVectorXd sum_sq = Eigen::RowVectorXd::Zero(mu.size());
  for (std::size_t m = 0; m < chains.num_chains(); ++m) {
    const auto& chain = chains.chain_view(m);
    sum_sq += (chain.rowwise() - mu).array().square().colwise().sum().matrix();
  }
  return sum_sq / static_cast<double>(chains.num_draws() - 1);
}

/**
 * @brief Return the sample standard deviations of the variables in the chains.
 *
 * The standard deviations are calculated for each variable (i.e.,
 * each dimension). The formula divides by the number of draws minus
 * one. Unlike the sample variance estimate, sample standard
 * deviations are not unbiased estimates of population standard
 * deviations due to the nonlinearity of the square root operation.
 *
 * If there is only one draw in a chain, this function will return
 * `Nan`.
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The standard deviations.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd sample_standard_deviation(const MC& chains) {
  return sample_variance(chains).array().sqrt().matrix();
}

/**
 * @brief Return the empirical quantiles of the draws.
 *
 * The returned matrix has one quantile per row, with one dimension
 * per column.
 *
 * For each variable, the empirical quantile at probability `p` is
 * computed by sorting the variable's column and linearly
 * interpolating between an upper bounding and lower bounding value.
 * This function's behavior matches R's `stats::quantile(x, probs,
 * type = 7)` (the default `type`) and NumPy's `numpy.quantile(a, q,
 * method='linear')` (the default `method`).
 *
 * In pseudocode, where `column` is the column of values, the quantile for
 * probability `p` is calculated as follows.
 *
 * @code
 * sorted = sort(column)
 * idx = p * (N - 1)          // (N - 1) is last index
 * lb = floor(h)
 * ub = min(lb + 1, N - 1)    // don't go past last index
 * ub_frac = idx - lb         // distance toward upper bound
 * lb_frac = ub - idx         // lb_frac + ub_frac = 1
 * quantile = ub_frac * sorted[ub] + lb_frac * sorted[lb]
 * @endcode
 *
 * For example, if the values are `column = (9, 11, 5, 3)` and the
   * probability is `p = 0.6`, we have
   *
   * @code
   * sorted                        = (3, 5, 9, 11)
   * idx = 0.6 * (4 - 1)           = 1.8
   * lb = floor(1.8)               = 1
   * ub = min(1 + 1, 4 - 1)        = 2
   * ub_frac = 1.8 - 1             = 0.8
   * lb_frac = 2 - 1.8             = 0.2
   * sorted[1]                     = 5
   * sorted[2]                     = 9
   * quantile = 0.8 * 9 + 0.2 * 5  = 8.2
   * @endcode
 *
 * @see <a
 href="https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html">R
 * `stats:quantile`function, `type=7`</a>

 * @see <a
 href="https://numpy.org/doc/stable/reference/generated/numpy.quantile.html">NumPy
 * (Python) `numpy.quantile` function,`method='linear'`</a> (Python).
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @param[in] probs A vector of probabilities in [0, 1].
 * @return The quantiles with one row per quantile.
 * @throw std::invalid_argument If a value in `probs` is outside [0, 1].
 */
template <MarkovChainSequence MC>
inline Eigen::MatrixXd quantiles(const MC& chains,
                                 const Eigen::VectorXd& probs) {
  if (std::ranges::any_of(probs, [](double p) {
        if (!(p >= 0)) {
          return true;
        }
        if (!(p <= 1)) {
          return true;
        }
        return false;
      })) {
    throw std::invalid_argument("probs must be in [0, 1]");
  }
  const Eigen::Index N = static_cast<Eigen::Index>(chains.num_draws());
  const Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
  const Eigen::Index K = probs.size();
  Eigen::MatrixXd result(K, D);
  for (Eigen::Index d = 0; d < D; ++d) {
    Eigen::VectorXd col = chains.draws(d);  // chains const, sort copy
    std::sort(col.begin(), col.end());
    double n_minus_1 = static_cast<double>(N - 1);
    for (Eigen::Index k = 0; k < K; ++k) {
      const double h = probs(k) * n_minus_1;
      const Eigen::Index lo = static_cast<Eigen::Index>(std::floor(h));
      const Eigen::Index hi = std::min(lo + 1, N - 1);  // lo + 1 == ceil(h)
      const double frac = h - static_cast<double>(lo);
      result(k, d) = col(lo) + frac * (col(hi) - col(lo));
    }
  }
  return result;
}

/**
 * @brief Return the matrix of autocovariances at all lags for each of the
 * chains.
 *
 * The return will be of shape `chains.num_draws() x chains.dims()`.
 * The indexes will represent the lag, so that
 * `autocovariance(chains)[0]` is the variance, `...[1]` gives the
 * lag-1 autocovariance, and so on.
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] The Markov chains.
 * @return The matrix of autocovariances.
 */
template <MarkovChainSequence MC>
Eigen::MatrixXd autocovariance(const MC& chains) {
  Eigen::FFT<double> fft;
  Eigen::Index N = static_cast<Eigen::Index>(chains.num_draws());
  Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
  std::size_t M = chains.num_chains();
  Eigen::MatrixXd acov(N, D);
  Eigen::Index start = 0;
  // to parallelize, need to use one fft per chain
  for (std::size_t m = 0; m < M; ++m) {
    const auto& chain = chains.chain_view(m);
    acov.middleRows(start, chain.rows()) =
        detail::autocovariance_chain(chain, fft);
    start += chain.rows();
  }
  return acov;
}

/**
 * @brief Return the chain-balanced ragged R-hat statistics for the chains.
 *
 * The R-hat statistic weights the within-chain mean and variance of each
 * chain equally, no matter how long they are. The variance term used from
 * R-hat is derived from the Margossian (2025) R-hat estimator.
 *
 * The number of draws per chain may vary, so let `chain[k]` be the
 * `N[k] x D` matrix of draws for chain `k`. The means and variances
 * are taken column-wise as in the `walnutpie::mean` and
 * `walnutpie::sample_variance` functions. Sample variance divides by
 * `(N[k] - 1)` for an unbiased estimate of variance.
 *
 * @code
 * matrix[K, D] mu, sigma_sq;
 * mu[k, ] = mean(chain[k])  for k in 1:K
 * sigma_sq[k, ] = sample_variance(chain[k])
 * R-hat = sqrt(1 + sample_variance(mu) ./ mean(sigma_sq))
 * @endcode
 *
 * For the original R-hat, see: Gelman, Andrew and Rubin, Donald
 * B. 1992. [Inference from Iterative Simulation Using Multiple
 * Sequences](https://projecteuclid.org/journals/statistical-science/volume-7/issue-4/Inference-from-Iterative-Simulation-Using-Multiple-Sequences/10.1214/ss/1177011136.pdf).
 * Statistical Science.
 *
 * For the derivation of the equal length chain basis of the
 * definition here, see: Margossian, Charles C and Hoffman, Matthew D
 * and Sountsov, Pavel and Riou-Durand, Lionel and Vehtari, Aki and
 * Gelman, Andrew. 2025. [Nested R-hat: Assessing the convergence of
 * Markov chain Monte Carlo when running many short
 * chains](https://projecteuclid.org/journals/bayesian-analysis/volume-20/issue-4/Nested-Rˆ--Assessing-the-Convergence-of-Markov-Chain-Monte/10.1214/24-BA1453.full).
 * Bayesian Analysis.
 *
 * This function will throw an exception if there is a chain with
 * fewer than three draws. The number is because it requires at least
 * three draws to compute a lag-1 autocorrelation. In practice, we
 * require more than three draws per chain. 25 draws per chain is a
 * reasonable minimum for practical applications that mix well, but
 * slower mixing problems will require more.
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The R-hat statistic for each variable in the chain.
 * @throw std::invalid_argument If there are not at least two chains or if
 * any of the chains has fewer than 3 draws.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd r_hat(const MC& chains) {
  if (chains.num_chains() < 2) {
    throw std::invalid_argument("require at least two chains to compute R-hat");
  }
  for (std::size_t m = 0; m < chains.num_chains(); ++m) {
    const auto& chain_view = chains.chain_view(m);
    if (chain_view.rows() < 3) {
      throw std::invalid_argument("each chain must have at least 3 draws");
    }
  }
  std::size_t M = chains.num_chains();
  Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
  Eigen::MatrixXd mu(M, D);
  Eigen::MatrixXd sigma_sq(M, D);
  for (std::size_t m = 0; m < M; ++m) {
    const auto& chain = chains.chain_view(m);
    auto chain_mean = detail::col_means(chain);
    mu.row(static_cast<Eigen::Index>(m)) = chain_mean;
    sigma_sq.row(static_cast<Eigen::Index>(m)) =
        detail::sample_variance(chain, chain_mean);
  }
  return (1.0 + detail::sample_variance(mu).array() /
                    detail::col_means(sigma_sq).array())
      .sqrt()
      .matrix();
}

/**
 * @brief Return the effective sample size statistics for the chains.
 *
 * The effective sample size is adjusted downward when R-hat is
 * greater than 1 (Gelman et al. 2013) using the Margossian (2025)
 * estimator for the combined variance in R-hat. If only a single
 * chain is provided, there is no adjustment.
 *
 * This only uses a number of draws equal to the shortest chain.
 *
 * The algorithm is \f$\mathcal{O}(N log N)\f$ per dimension with N draws. The
 * computational bottleneck is that autocovariances are calculated
 * with Eigen's built-in not-so-fast Fourier transform (FFT).
 *
 * The implementation is based on the one in Stan.
 *
 * See Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson,
 * David B and Vehtari, Aki and Rubin, Donald B. 2014. [Bayesian Data
 * Analysis](https://sites.stat.columbia.edu/gelman/book/), Third
 * Edition. CRC Press.
 *
 * @see  Margossian, Charles C and Hoffman, Matthew D
 * and Sountsov, Pavel and Riou-Durand, Lionel and Vehtari, Aki and
 * Gelman, Andrew. 2025. [Nested R-hat: Assessing the convergence of
 * Markov chain Monte Carlo when running many short
 * chains](https://projecteuclid.org/journals/bayesian-analysis/volume-20/issue-4/Nested-Rˆ--Assessing-the-Convergence-of-Markov-Chain-Monte/10.1214/24-BA1453.full).
 * Bayesian Analysis.
 *
 * See: The Stan Development Team. 2026. (The Stan Reference
 * Manual)[https://mc-stan.org/docs/reference-manual/]. Version 2.39.
 * Stan Web Site.
 *
 * See the Stan C++ source code for [effective sample size
 * (ESS)](https://github.com/stan-dev/stan/blob/develop/src/stan/analyze/mcmc/ess.hpp)
 * and
 * [R-hat](https://github.com/stan-dev/stan/blob/develop/src/stan/analyze/mcmc/rhat.hpp).
 * GitHub.
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The effective sample size statistic for each variable in the chain.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd effective_sample_size(const MC& chains) {
  if (chains.num_draws() < 3) {
    throw std::invalid_argument("chains must have at least 3 draws");
  }
  Eigen::Index D = static_cast<Eigen::Index>(chains.dims());
  std::size_t K = chains.num_chains();
  std::size_t N_total = chains.num_draws();
  Eigen::Index min_len = chains.min_chain_size();

  Eigen::MatrixXd chain_means(K, D);
  Eigen::MatrixXd chain_vars(K, D);
  for (std::size_t k = 0; k < K; ++k) {
    const auto& chain = chains.chain_view(k);
    Eigen::RowVectorXd m = detail::col_means(chain);
    chain_means.row(static_cast<Eigen::Index>(k)) = m;
    chain_vars.row(static_cast<Eigen::Index>(k)) =
        detail::sample_variance(chain, m);
  }

  Eigen::RowVectorXd W = detail::col_means(chain_vars);
  Eigen::RowVectorXd var_plus = W;
  if (K > 1) {
    var_plus += detail::sample_variance(chain_means);
  }

  Eigen::MatrixXd acov = autocovariance(chains);

  Eigen::RowVectorXd result(D);
  for (Eigen::Index d = 0; d < D; ++d) {
    const double w_d = W(d);
    const double vp_d = var_plus(d);

    // mean_acov_at_lag(t): average over chains of acov
    auto mean_acov_at_lag = [&](Eigen::Index t) {
      double sum = 0.0;
      Eigen::Index start = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += acov(start + t, d);
        start += chains.chain_view(k).rows();
      }
      return sum / static_cast<double>(K);
    };

    Eigen::VectorXd rho_hat_t = Eigen::VectorXd::Zero(min_len);
    double rho_hat_even = 1.0;
    rho_hat_t(0) = rho_hat_even;

    double rho_hat_odd = 1.0 - (w_d - mean_acov_at_lag(1)) / vp_d;
    rho_hat_t(1) = rho_hat_odd;

    // Geyer's initial positive + monotone sequence on paired lags
    // min_len - 4 prevents reading beyond the end of chains
    Eigen::Index t = 1;
    while (t < min_len - 4 && (rho_hat_even + rho_hat_odd) > 0.0) {
      rho_hat_even = 1.0 - (w_d - mean_acov_at_lag(t + 1)) / vp_d;
      rho_hat_odd = 1.0 - (w_d - mean_acov_at_lag(t + 2)) / vp_d;

      if ((rho_hat_even + rho_hat_odd) >= 0.0) {
        rho_hat_t(t + 1) = rho_hat_even;
        rho_hat_t(t + 2) = rho_hat_odd;
      }
      // initial positive -> initial monotone
      if (rho_hat_t(t + 1) + rho_hat_t(t + 2) >
          rho_hat_t(t - 1) + rho_hat_t(t)) {
        rho_hat_t(t + 1) = (rho_hat_t(t - 1) + rho_hat_t(t)) / 2.0;
        rho_hat_t(t + 2) = rho_hat_t(t + 1);
      }
      t += 2;
    }

    Eigen::Index max_t = t;
    // antithetic-tail correction
    if (rho_hat_even > 0.0) {
      rho_hat_t(max_t + 1) = rho_hat_even;
    }

    double tau_hat =
        -1.0 + 2.0 * rho_hat_t.head(max_t).sum() + rho_hat_t(max_t + 1);

    // safety floor: tau >= 1/log10(N_total)
    tau_hat = std::max(tau_hat, 1.0 / std::log10(static_cast<double>(N_total)));

    result(d) = static_cast<double>(N_total) / tau_hat;
  }
  return result;
}

/**
 * @brief Return the Monte Carlo standard error for each dimension.
 *
 * The Monte Carlo standard error (MCSE) is just the sample standard deviation
 * divided by the square root of the effective sample size.
 *
 * @see effective_sample_size()
 * @see sample_standard_deviation()
 *
 * @tparam MC The type of the Markov chain sequence.
 * @param[in] chains The Markov chains.
 * @return The Monte Carlo standard error estimates.
 */
template <MarkovChainSequence MC>
inline Eigen::RowVectorXd monte_carlo_standard_error(const MC& chains) {
  auto ess = effective_sample_size(chains);
  auto sd = sample_standard_deviation(chains);
  return (sd.array() / ess.array().sqrt());
}

}  // namespace walnutpie
