#pragma once

#include <atomic>
#include <concepts>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include <walnutpie/validate.hpp>

namespace detail {

/**
 * @brief Internal flag with value `true` if C++ received `SIGINT`.
 */
std::atomic<bool> interrupted{false};

extern "C" void handle_sigint(int) { interrupted = true; }

/**
 * @brief Write the value in binary format as the specified type.
 *
 * @tparam T The type to which input is cast and written.
 * @tparam S The type of the input value.
 * @param[in] out The output stream.
 * @param[in] x The value that is written as type `T`.
 */
template <typename T, typename S>
  requires std::convertible_to<S, T> && std::is_trivially_copyable_v<T>
static void write_binary(std::ostream& out, S x) {
  auto y = static_cast<T>(x);
  auto bytes = reinterpret_cast<const char*>(&y);
  out.write(bytes, sizeof(y));
}

/**
 * @brief Write the vector in binary format.
 *
 * @param[in] os Output stream to which the vector is written.
 * @param[in] v The vector to write.
 */
static void write_vector(std::ostream& os, const Eigen::VectorXd& v) {
  const char* data = reinterpret_cast<const char*>(v.data());
  std::size_t dim = static_cast<std::size_t>(v.size());
  std::streamsize size = static_cast<std::streamsize>(dim * sizeof(double));
  os.write(data, size);
}

/**
 * @brief Validate that the specified stream is open.
 *
 * @tparam S The type of the stream.
 * @param[in] s The stream.
 * @param[in] name The name of the stream.
 * @throw invalid_argument If the stream is not open.
 */
template <typename S>
inline void validate_open(const S& s, const std::string& name) {
  if (s.is_open()) {
    return;
  }
  throw std::invalid_argument("could not open stream from: " + name);
}

}  // namespace detail

/**
 * @brief An interrupt callback for C++.
 */
class CppInterruptCallback {
 public:
  /**
   * @brief Construct an interrupt callback for C++.
   */
  CppInterruptCallback() { std::signal(SIGINT, detail::handle_sigint); }
  /**
   * @brief Throw an exception if C++ signaled an interrupt.
   */
  void throw_if_interrupted() const {
    if (detail::interrupted.load()) {
      throw std::runtime_error("C++ was interrupted with SIGINT");
    }
  }
};

/**
 * @brief A handler that stores global events.
 */
class GlobalStore {
 public:
  /**
   * @brief Handle the R-hat value.
   *
   * @param[in] r_hat The R-hat value.
   */
  void on_r_hat(double r_hat) { r_hats_.push_back(r_hat); }

  /**
   * @brief Return the R-hat values.
   *
   * @return The R-hat values.
   */
  const std::vector<double>& r_hats() const noexcept { return r_hats_; }

 private:
  std::vector<double> r_hats_;
};

/**
 * @brief A handler that stores chain-local events.
 */
class ChainStore {
 public:
  /**
   * @brief Construct a chain store that optionally saves warmup iterations.
   *
   * @param[in] save_warmup Set to `true` to save warmup iterations.
   */
  explicit ChainStore(bool save_warmup = false) : save_warmup_(save_warmup) {}

  /**
   * @brief Handle a warmup draw with meta-information.
   *
   * If `save_warmup()` is `true`, this operation appends the values
   * to vectors for later use. Otherwise, it does nothing if
   * `save_warmup()` is `false`.
   *
   * @param[in] position The position observed.
   * @param[in] lp The log density of the position observed.
   * @param[in] step_size The step size observed.
   * @param[in] diag_inv_mass The diagonal of the inverse mass matrix observed.
   */
  void on_warmup(const Eigen::VectorXd& position, const double lp,
                 const double step_size, const Eigen::VectorXd& diag_inv_mass) {
    if (!save_warmup_) {
      return;
    }
    warmup_draws_.push_back(position);
    warmup_lps_.push_back(lp);
    warmup_stepsizes_.push_back(step_size);
    warmup_diag_inv_masses_.push_back(diag_inv_mass);
  }

  /**
   * @brief Handle the warmup completion event.
   *
   * @param[in] step_size The step size.
   * @param[in] diag_inv_mass The diagonal inverse mass matrix.
   */
  void on_warmup_complete(double step_size,
                          const Eigen::VectorXd& diag_inv_mass) {
    stepsize_ = step_size;
    diag_inv_mass_ = diag_inv_mass;
  }

  /**
   * @brief Handle a sampling event.
   *
   * @param[in] position The position.
   * @param[in] lp The log density.
   */
  void on_sample(const Eigen::VectorXd& position, double lp) {
    draws_.push_back(position);
    lps_.push_back(lp);
  }

  void on_logp_exception(const Eigen::VectorXd& position,
                         const std::exception& exn) const noexcept {
    std::cout << "Logp failed with exception " << exn.what() << " at "
              << position.transpose() << "\n";
  }

  /**
   * @brief Return `true` if warmup iterations are saved.
   *
   * @return `true` if warmup iterations are saved.
   */
  bool save_warmup() const noexcept { return save_warmup_; }

  /**
   * @brief Return the step size.
   *
   * This method only makes sense once `on_warmup_complete()` has been called.
   *
   * @return The step size.
   */
  double step_size() const noexcept { return stepsize_; }

  /**
   * @brief Return the diagonal inverse mass matrix.
   *
   * @return The inverse mass matrix.
   */
  const Eigen::VectorXd& diag_inv_mass() const noexcept {
    return diag_inv_mass_;
  }

  /**
   * @brief Return the draws from sampling.
   *
   * @return The sampling draws.
   */
  const std::vector<Eigen::VectorXd>& draws() const noexcept { return draws_; }

  /**
   * @brief Return the log densities for the draws from sampling.
   *
   * @return The sampling log densities.
   */
  const std::vector<double>& log_probs() const noexcept { return lps_; }

  /**
   * @brief Return the draws from warmup.
   *
   * @return The warmup draws.
   */
  const std::vector<Eigen::VectorXd>& warmup_draws() const noexcept {
    return warmup_draws_;
  }

  /**
   * @brief Return the log densities for the draws from warmup.
   *
   * @return The warmup log densities.
   */
  const std::vector<double>& warmup_log_probs() const noexcept {
    return warmup_lps_;
  }

  /**
   * @brief Return the step sizes from warmup.
   *
   * @return The warmup step sizes.
   */
  const std::vector<double>& warmup_step_sizes() const noexcept {
    return warmup_stepsizes_;
  }

  /**
   * @brief Return the diagonal inverse mass matrices from warmup.
   *
   * @return The warmup mass matrices.
   */
  const std::vector<Eigen::VectorXd>& warmup_diag_inv_masses() const noexcept {
    return warmup_diag_inv_masses_;
  }

 private:
  bool save_warmup_;

  double stepsize_ = 0;
  Eigen::VectorXd diag_inv_mass_;

  std::vector<double> r_hats_;

  std::vector<Eigen::VectorXd> draws_;
  std::vector<double> lps_;

  std::vector<Eigen::VectorXd> warmup_draws_;
  std::vector<double> warmup_lps_;
  std::vector<double> warmup_stepsizes_;
  std::vector<Eigen::VectorXd> warmup_diag_inv_masses_;
};

/**
 * @brief Write the step sizes in binary format.
 *
 * The binary format is a single `uint64_t` indicating the number of
 * chains and then one `double` per chain for the step sizes in order.
 *
 * @param[out] os The output stream.
 * @param[out] handlers The storage handlers for the chains.
 */
static void write_step_size(std::ostream& os,
                            const std::vector<ChainStore>& handlers) {
  if (handlers.empty()) {
    return;
  }
  detail::write_binary<std::uint64_t>(os, handlers.size());
  for (const auto& handler : handlers) {
    detail::write_binary<double>(os, handler.step_size());
  }
}

/**
 * @brief Write the diagonal mass matrices in binary format.
 *
 * The binary format is a single `uint64_t` indicating the
 * number of chains, then a single `uint64_t` indicating
 * the dimensionality of each chain, then the output of each
 * each chain sequentially. Each chain is output as a sequence
 * of `double` values.
 *
 * @param[out] os The output stream.
 * @param[out] handlers The storage handlers for the chains.
 */
static void write_mass_matrix(std::ostream& os,
                              const std::vector<ChainStore>& handlers) {
  if (handlers.empty()) {
    return;
  }
  Eigen::Index D = handlers[0].diag_inv_mass().size();
  if (D == 0) {
    return;
  }
  std::size_t M = handlers.size();
  detail::write_binary<std::uint64_t>(os, M);
  detail::write_binary<std::uint64_t>(os, D);
  for (const auto& handler : handlers) {
    detail::write_vector(os, handler.diag_inv_mass());
  }
}

/**
 * @brief Write the draws and log densities to the stream, specifying if
 * they are sampling or warmup draws.
 *
 * First, the number of chains is written (`std::uint64_t`). Then the
 * draws are written in order. For each draw, its log density is
 * written first (`double`). Then the values are written in order
 * (`double`), one per dimension.
 *
 * This is not a recoverable format without knowing the dimensionality and
 * number of draws.
 *
 * @param[in] os Output stream to which values are written.
 * @param[in] is_sampling `true` if these draws are sampling draws, `false` if
 * warmup.
 * @param[in] draws The draws to write.
 * @param[in] log_probs The log densities to write with the draws.
 */
static void write_draws(std::ostream& os, bool is_sampling,
                        const std::vector<Eigen::VectorXd>& draws,
                        const std::vector<double>& log_probs) {
  detail::write_binary<std::uint64_t>(os, draws.size());
  for (std::size_t m = 0; m < draws.size(); ++m) {
    detail::write_binary<uint64_t>(os, is_sampling);
    detail::write_binary<double>(os, log_probs[m]);
    detail::write_vector(os, draws[m]);
  }
}

/**
 * @brief Write the draws in binary format, optionally including
 * warmup draws.
 *
 * The output format first writes the number of chains
 * (`std::uint64_t`), then the number of dimensions (`std::uint64_t`).
 * If `include_warmup` is true, it then writes the warmup draws for
 * each chain in order (see `write_draws()`). Then it writes the
 * draws for each chain (see `write_draws()`).
 *
 * @param[out] os The output stream.
 * @param[out] handlers The storage handlers for the chains.
 * @param[in] include_warmup `true` if warmup draws are included in output.
 */
static void write_sample(std::ostream& os,
                         const std::vector<ChainStore>& handlers,
                         bool include_warmup = false) {
  // get dims from first real draw---roundabout for possible zero sizes
  Eigen::Index D = 0;
  for (const auto& h : handlers) {
    if (!h.draws().empty()) {
      D = h.draws()[0].size();
      break;
    }
    if (!h.warmup_draws().empty()) {
      D = h.warmup_draws()[0].size();
      break;
    }
  }
  if (D == 0) {
    return;
  }
  std::size_t M = handlers.size();
  detail::write_binary<std::uint64_t>(os, M);
  detail::write_binary<std::uint64_t>(os, D);
  if (include_warmup) {
    for (const auto& h : handlers) {
      write_draws(os, false, h.warmup_draws(), h.warmup_log_probs());
    }
  }
  for (const auto& h : handlers) {
    write_draws(os, true, h.draws(), h.log_probs());
  }
}

/**
 * @brief Write the step sizes in binary format.
 *
 * The binary format is a single `uint64_t` indicating the number of
 * chains and then one `double` per chain for the step sizes in order.
 *
 * @param[in] file_name The name of the file to which the step sizes are
 * written.
 * @param[out] handlers The storage handlers for the chains.
 * @throw std::invalid_argument If the file cannot be opened for writing.
 */
static void write_step_size(const std::string& file_name,
                            const std::vector<ChainStore>& handlers) {
  if (handlers.empty()) {
    return;
  }
  std::ofstream os(file_name);
  detail::validate_open(os, file_name);
  write_step_size(os, handlers);
}

/**
 * @brief Write the diagonal mass matrices in binary format.
 *
 * See `write_mass_matrix(const std::vector<ChainStore>&, std::ostream&)` for
 * a description of the format.
 *
 * @param[out] file_name The anme of the file to which the step sizes are
 * written.
 * @param[out] handlers The storage handlers for the chains.
 * @throw std::invalid_argument If the file cannot be opened for writing.
 */
static void write_mass_matrix(const std::string& file_name,
                              const std::vector<ChainStore>& handlers) {
  if (handlers.empty()) {
    return;
  }
  std::ofstream os(file_name);
  detail::validate_open(os, file_name);
  write_mass_matrix(os, handlers);
}

/**
 * @brief Write the draws in binary format, optionally including
 * warmup draws.
 *
 * See `write_sample(const std::vector<ChainStore>&, std::ostream&, bool)` for
 * details of the format.
 *
 * @param[out] file_name The name of the file to which output is written.
 * @param[out] handlers The storage handlers for the chains.
 * @param[in] include_warmup `true` if warmup draws are included in output.
 */
static void write_sample(const std::string& file_name,
                         const std::vector<ChainStore>& handlers,
                         bool include_warmup = false) {
  if (handlers.empty()) {
    return;
  }
  std::vector<char> filebuf(8u << 20);  // bigger than default buffer
  std::ofstream os;
  os.rdbuf()->pubsetbuf(filebuf.data(),
                        static_cast<std::streamsize>(filebuf.size()));
  os.open(file_name, std::ios::binary);
  detail::validate_open(os, file_name);
  write_sample(os, handlers, include_warmup);
}
