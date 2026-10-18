#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "walnutpie/concepts.hpp"
#include "walnutpie/util.hpp"
#include "walnutpie/validate.hpp"

namespace walnutpie::detail {

/**
 * @brief A class for holding the minimal information in a Hamiltonian
 * trajectory required for Walnuts.
 *
 * A span has member variables for the initial and final states' (a)
 * position, (b) momentum, (c) log density of the state, and (d)
 * gradient of target log density. It also holds a selected state,
 * the gradient of the selected state, and the log of the sum of all
 * joint densities on the trajectory. The gradients could be recomputed,
 * but storing them serves as a local cache.
 *
 */
class SpanW {
 public:
  /**
   * @brief Initialize this span in place to the span of one state with the
   * specified position, momentum, gradient, and log densities.
   *
   * The argument vectors are swapped into this span, so if this span's
   * vectors are already sized for the state (see `resize()`), no heap
   * allocation is required and the arguments are left holding sized
   * storage.
   *
   * @param[in,out] theta The position.
   * @param[in,out] rho The momentum.
   * @param[in,out] grad_theta The gradient of the log density at `theta`.
   * @param[in] logp_pos The log density of the position.
   * @param[in] logp_joint The joint log density of the position and momentum.
   */
  void init_from(Eigen::VectorXd&& theta, Eigen::VectorXd&& rho,
                 Eigen::VectorXd&& grad_theta, double logp_pos,
                 double logp_joint) {
    theta_bk_ = theta;
    theta_fw_ = theta_bk_;
    theta_select_.swap(theta);
    rho_bk_ = rho;
    rho_fw_.swap(rho);
    grad_theta_bk_ = grad_theta;
    grad_theta_fw_ = grad_theta_bk_;
    grad_select_.swap(grad_theta);
    logp_bk_ = logp_joint;
    logp_fw_ = logp_joint;
    logp_pos_select_ = logp_pos;
    logp_ = logp_joint;
  }

  /**
   * @brief Initialize this span in place by concatenating the two specified
   * spans, with the given state selected and total log density.
   *
   * This span must not be one of the specified spans. As with
   * `init_from()`, the vectors are swapped in place.
   *
   * @param[in,out] span1 The earlier span by temporal ordering.
   * @param[in,out] span2 The later span by temporal ordering.
   * @param[in,out] selected The span holding the selected position, which
   * may be `span1` or `span2`.
   * @param[in] logp_pos_select The log density of the selected position.
   * @param[in] logp_joint_total The log of the sum of the joint densities of
   * positions and momentums on the trajectory.
   */
  void init_from_subspans(SpanW& span1, SpanW& span2, SpanW& selected,
                          double logp_pos_select, double logp_joint_total) {
    theta_bk_.swap(span1.theta_bk_);
    rho_bk_.swap(span1.rho_bk_);
    grad_theta_bk_.swap(span1.grad_theta_bk_);
    logp_bk_ = span1.logp_bk_;
    theta_fw_.swap(span2.theta_fw_);
    rho_fw_.swap(span2.rho_fw_);
    grad_theta_fw_.swap(span2.grad_theta_fw_);
    logp_fw_ = span2.logp_fw_;
    theta_select_.swap(selected.theta_select_);
    grad_select_.swap(selected.grad_select_);
    logp_pos_select_ = logp_pos_select;
    logp_ = logp_joint_total;
  }

  /**
   * @brief Resize all the vectors of this span to hold `dim` values.
   *
   * @param[in] dim The number of values in each vector.
   */
  void resize(Eigen::Index dim) {
    theta_bk_.resize(dim);
    rho_bk_.resize(dim);
    grad_theta_bk_.resize(dim);
    theta_fw_.resize(dim);
    rho_fw_.resize(dim);
    grad_theta_fw_.resize(dim);
    theta_select_.resize(dim);
    grad_select_.resize(dim);
  }

  /** The earliest state. */
  Eigen::VectorXd theta_bk_;

  /** The earliest momentum. */
  Eigen::VectorXd rho_bk_;

  /** The gradient of the target log density at the earliest state . */
  Eigen::VectorXd grad_theta_bk_;

  /** The joint log density of the earliest position and momentum. */
  double logp_bk_;

  /** The latest state in the trajectory. */
  Eigen::VectorXd theta_fw_;

  /** The latest momentum in the trajectory. */
  Eigen::VectorXd rho_fw_;

  /** The gradient of the target log density at the latest position. */
  Eigen::VectorXd grad_theta_fw_;

  /** The joint log density of the latest position and momentum. */
  double logp_fw_;

  /** The selected state. */
  Eigen::VectorXd theta_select_;

  /** The gradient of the log density at the selected state. */
  Eigen::VectorXd grad_select_;

  /** The log density of the selected state. */
  double logp_pos_select_;

  /** The log of the sum of the joint densities in the trajectory. */
  double logp_;
};

/**
 * @brief Preallocated storage for building one Hamiltonian trajectory.
 *
 * Building a trajectory out of freshly allocated spans costs a heap
 * allocation per vector per leaf, which dominates the arithmetic of a
 * leapfrog step. A workspace instead holds two span slots per tree
 * level, two for the accumulated trajectory, and the trial vectors for
 * leaf construction and the checks, which never overlap. Vectors are
 * swapped between the equally sized slots, so a transition does not
 * allocate on the heap. Each chain requires its own workspace.
 */
class SpanWorkspace {
 public:
  /**
   * @brief Construct a workspace for positions of the specified
   * dimensionality and trajectories of up to the specified number of
   * doublings.
   *
   * @param[in] dim The dimensionality of the positions.
   * @param[in] max_depth The maximum number of trajectory doublings.
   */
  SpanWorkspace(Eigen::Index dim, std::size_t max_depth)
      : spans_(2 * (max_depth + 1) + 2) {
    for (auto& span : spans_) {
      span.resize(dim);
    }
    theta_next_.resize(dim);
    rho_next_.resize(dim);
    grad_next_.resize(dim);
    rev_theta_.resize(dim);
    rev_rho_.resize(dim);
    rev_grad_.resize(dim);
    scaled_diff_.resize(dim);
  }

  /**
   * @brief Return one of the two span slots for the halves of the
   * subtree built at the specified tree level.
   *
   * @param[in] level The tree level, from zero.
   * @param[in] slot The index of the slot within the level, zero or one.
   * @return The span slot for the level and slot.
   */
  SpanW& span(std::size_t level, std::size_t slot) {
    return spans_.at(2 * level + slot);
  }

  /**
   * @brief Return one of the two span slots holding the trajectory
   * accumulated so far.
   *
   * @param[in] i The index of the accumulator slot, zero or one.
   * @return The accumulator span slot.
   */
  SpanW& accumulator(std::size_t i) { return spans_.at(spans_.size() - 2 + i); }

  /** The trial position for the macro step building a leaf. */
  Eigen::VectorXd theta_next_;

  /**
   * The trial momentum for the macro step building a leaf; also holds
   * the initial momentum drawn at the start of a transition.
   */
  Eigen::VectorXd rho_next_;

  /** The trial gradient for the macro step building a leaf. */
  Eigen::VectorXd grad_next_;

  /** The trial position for the reversibility check. */
  Eigen::VectorXd rev_theta_;

  /** The trial momentum for the reversibility check. */
  Eigen::VectorXd rev_rho_;

  /** The trial gradient for the reversibility check. */
  Eigen::VectorXd rev_grad_;

  /** The scaled difference between span ends for the U-turn check. */
  Eigen::VectorXd scaled_diff_;

 private:
  /** The span slots: two per tree level, then the two accumulators. */
  std::vector<SpanW> spans_;
};

/**
 * @brief Return a tuple of the arguments ordered by direction.

 * The arguments are forwarded as is the returned tuple and returned
 * by reference, so function arguments must stay in scope. If the
 * template argument `D` is `Direction::Forward`, then the tuple is
 * `(x1, x2)`; if `D` is `Direction::Backward`, the returned tuple is
 * `(x2, x1)`.
 *
 * The template parameter `T` is generic in order to allow reference
 * collapsing in callers. Working through all of the types and forwarding
 * here, the type of the return
 *
 * @tparam D The `Direction` in which to combine the arguments
 * (`Forward` or `Backward`).
 * @tparam T The type of the arguments.
 * @param[in] x1 The first argument.
 * @param[in] x2 The second argument.
 * @return The arguments ordered according to `D`.
 */
template <Direction D, typename T>
static std::tuple<T&, T&> order_forward_backward(T&& x1, T&& x2) {
  if constexpr (D == Direction::Forward) {
    return std::forward_as_tuple(std::forward<T>(x1), std::forward<T>(x2));
  } else {  // Direction::Backward
    return std::forward_as_tuple(std::forward<T>(x2), std::forward<T>(x1));
  }
}

/**
 * @brief Return `true` if the two spans ordered as specified form a
 * U-turn in the metric determined by the inverse mass matrix.
 *
 * For computing U-turns, the squared distance between two vectors `x` and `y`
 * is defined by the Mahalanobis distance,
 * ```
 * d(x, y)**2 = (x - y)' * inv_mass * (x - y).
 * ```
 * Equivalently, distance is measured in the Euclidean metric with metric
 * tensor given by the mass matrix.
 *
 * If the spans ordered according to `D` are `(span_bk, span_fw)`, let
 * `theta_start` be the first position in `span_bk` and let `theta_end` be
 * the last position of `span_fw`. The U-turn condition will be satisfied if
 * ```
 * theta_start * delta < 0  OR   theta_end * delta < 0,
 * ```
 * where
 * ```
 * delta = inv_mass .* (theta_end - theta_start).
 * ```
 *
 * @tparam D The direction in which to order the spans.
 * @tparam U The type of spans, which must define begin and end positions.
 * @param[in] span1 The first argument span.
 * @param[in] span2 The second argument span.
 * @param[in] inv_mass The inverse mass matrix to determine distances.
 * @param[out] scaled_diff Storage for the scaled difference between the
 * span ends, sized for the positions.
 * @return `true` if there is a U-turn between the ends of the ordered spans.
 */
template <Direction D>
static bool uturn(const SpanW& span1, const SpanW& span2,
                  const Eigen::VectorXd& inv_mass,
                  Eigen::VectorXd& scaled_diff) {
  auto [span_bk, span_fw] = order_forward_backward<D>(span1, span2);
  scaled_diff =
      inv_mass.array() * (span_fw.theta_fw_ - span_bk.theta_bk_).array();
  return span_fw.rho_fw_.dot(scaled_diff) < 0 ||
         span_bk.rho_bk_.dot(scaled_diff) < 0;
}

/**
 * @brief Return `true` if running the specified number of leapfrog steps
 * is within the maximum error tolerance.
 *
 * @tparam F The type of the log density/gradient function.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The micro step size.
 * @param[in] num_steps The number of micro steps to take.
 * @param[in] max_error The maximum error in Hamiltonian at macro steps.
 * @param[in] logp_next Initial log density.
 * @param[in,out] theta_next Input initial position, set to final position.
 * @param[in,out] rho_next Input initial momentum, set to final position.
 * @param[in,out] grad_next Input initial gradient, set to final gradient.
 */
template <LogpGrad F>
static bool within_tolerance(const F& logp_grad,
                             const Eigen::VectorXd& inv_mass, double step,
                             std::size_t num_steps, double max_error,
                             double logp_next, Eigen::VectorXd& theta_next,
                             Eigen::VectorXd& rho_next,
                             Eigen::VectorXd& grad_next) {
  double half_step = 0.5 * step;
  double logp = logp_next;
  for (std::size_t n = 0; n < num_steps; ++n) {
    rho_next += half_step * grad_next;
    theta_next.array() += step * inv_mass.array() * rho_next.array();
    logp_grad(theta_next, logp_next, grad_next);
    rho_next += half_step * grad_next;
  }
  logp_next += logp_momentum(rho_next, inv_mass);
  return std::abs(logp_next - logp) <= max_error;  // only tests one way
}

/**
 * @brief Return `true` if the number of micro steps provided is the one chosen
 * from the input position, moment, and gradient.
 *
 * @tparam F Type of log density/gradient function.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The micro step size.
 * @param[in] num_steps The number of micro steps proposed forward.
 * @param[in] min_micro_steps The minimum number of micro steps to take.
 * @param[in] max_error The maximum error tolerance in Hessians.
 * @param[in] logp_next The log density of the starting position.
 * @param[in] theta The final position from which to reverse.
 * @param[in] rho The final momentum from which to reverse.
 * @param[in] grad The final gradient from which to reverse.
 * @param[out] theta_next Storage for the trial position, sized for `theta`.
 * @param[out] rho_next Storage for the trial momentum, sized for `theta`.
 * @param[out] grad_next Storage for the trial gradient, sized for `theta`.
 * @return `true` if the path ending in the specified state is reversible.
 */
template <LogpGrad F>
static bool reversible(const F& logp_grad, const Eigen::VectorXd& inv_mass,
                       double step, std::size_t num_steps,
                       std::size_t min_micro_steps, double max_error,
                       double logp_next, const Eigen::VectorXd& theta,
                       const Eigen::VectorXd& rho, const Eigen::VectorXd& grad,
                       Eigen::VectorXd& theta_next, Eigen::VectorXd& rho_next,
                       Eigen::VectorXd& grad_next) {
  if (num_steps == 1) {
    return true;
  }
  while (num_steps >= 2 * min_micro_steps) {
    theta_next = theta;
    rho_next = -rho;
    grad_next = grad;
    num_steps /= 2;
    step *= 2;
    if (within_tolerance(logp_grad, inv_mass, step, num_steps, max_error,
                         logp_next, theta_next, rho_next, grad_next)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Take a macro step from the specified state given the log
 * density/gradient, tuning parameters and adaptation handler and
 * return whether it conserves the Hamiltonian and is reversible.
 *
 * @tparam D The time direction of Hamiltonian simulation.
 * @tparam F The type of the log density/gradient function.
 * @tparam A The type of the adaptation handler.
 * @param[in] logp_grad The target log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The initial micro step size.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum difference in Hamiltonians allowed in macro
 * steps.
 * @param[in] span The span to extend.
 * @param[out] theta_next The position after the macro step.
 * @param[out] rho_next The momentum after the macro step.
 * @param[out] grad_next The gradient of the position after the macro step.
 * @param[out] logp_pos_next The log density of the position and momentum after
 * the macro step.
 * @param[out] logp_next The log density of the position and momentum after the
 * macro step.
 * @param[in,out] workspace The workspace holding the trial vectors for
 * the reversibility check.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return `true` if the Hamiltonian is conserved reversibly.
 */
template <Direction D, LogpGrad F, StepSizeAdapter A>
static bool macro_step(const F& logp_grad, const Eigen::VectorXd& inv_mass,
                       double step, std::size_t max_step_halvings,
                       std::size_t min_micro_steps, double max_error,
                       const SpanW& span, Eigen::VectorXd& theta_next,
                       Eigen::VectorXd& rho_next, Eigen::VectorXd& grad_next,
                       double& logp_pos_next, double& logp_next,
                       SpanWorkspace& workspace, A& adapt_handler) {
  constexpr bool is_forward = (D == Direction::Forward);
  const Eigen::VectorXd& theta = is_forward ? span.theta_fw_ : span.theta_bk_;
  const Eigen::VectorXd& rho = is_forward ? span.rho_fw_ : span.rho_bk_;
  const Eigen::VectorXd& grad =
      is_forward ? span.grad_theta_fw_ : span.grad_theta_bk_;
  double logp = is_forward ? span.logp_fw_ : span.logp_bk_;
  step = is_forward ? step : -step;
  for (std::size_t num_steps = min_micro_steps, halvings = 0;
       halvings < max_step_halvings; ++halvings, num_steps *= 2, step *= 0.5) {
    theta_next = theta;
    rho_next = rho;
    grad_next = grad;
    double half_step = 0.5 * step;
    for (std::size_t n = 0; n < num_steps; ++n) {
      rho_next += half_step * grad_next;
      theta_next.array() += step * inv_mass.array() * rho_next.array();
      logp_grad(theta_next, logp_pos_next, grad_next);
      rho_next += half_step * grad_next;
    }
    logp_next = logp_pos_next + logp_momentum(rho_next, inv_mass);
    if (num_steps == min_micro_steps) {
      double min_accept = std::exp(-std::fabs(logp - logp_next));
      // Treat invalid energy changes as rejection for step-size adaptation.
      adapt_handler(std::isfinite(min_accept) ? min_accept : 0.0);
    }
    if (std::fabs(logp - logp_next) <= max_error) {
      return reversible(logp_grad, inv_mass, step, num_steps, min_micro_steps,
                        max_error, logp_next, theta_next, rho_next, grad_next,
                        workspace.rev_theta_, workspace.rev_rho_,
                        workspace.rev_grad_);
    }
  }
  return false;
}

/**
 * @brief Combine the specified spans into the output span and update
 * the selected position.
 *
 * If the direction `D` is `Forward`, then `span_new` is ordered after
 * `span_old` in time; if it is `Backward`, then `span_new` is before
 * `span_old`.
 *
 * The new selected state is determined with either a
 * Metropolis update rule or a Barker update rule based on the
 * template parameter, using the specified random number generator.
 *
 * @tparam U The type of update (`Metropolis` or `Barker`).
 * @tparam D The direction of combination in time (`Forward` or `Backward`).
 * @tparam Rand The type for the source of randomness.
 * @param[in,out] rng The random number generator used to select a new position.
 * @param[in,out] span_old The old span.
 * @param[in,out] span_new The span continuing the old span forward or backward
 * in time.
 * @param[out] out The combined span; must not alias the arguments.
 */
template <Update U, Direction D, std::uniform_random_bit_generator RNG>
inline void combine(Random<RNG>& rng, SpanW& span_old, SpanW& span_new,
                    SpanW& out) {
  double logp_total = log_sum_exp(span_old.logp_, span_new.logp_);
  double log_denominator;
  if constexpr (U == Update::Metropolis) {
    log_denominator = span_old.logp_;
  } else {  // Update::Barker
    log_denominator = logp_total;
  }
  double update_logprob = span_new.logp_ - log_denominator;
  bool update = std::log(rng.uniform_real_01()) < update_logprob;
  SpanW& selected = update ? span_new : span_old;
  double logp_pos_select =
      update ? span_new.logp_pos_select_ : span_old.logp_pos_select_;
  auto&& [span_bk, span_fw] = order_forward_backward<D>(span_old, span_new);
  out.init_from_subspans(span_bk, span_fw, selected, logp_pos_select,
                         logp_total);
}

/**
 * @brief Extend the specified span with a span of a single state.
 *
 * Given the specified span and direction `D`, build a new leaf span consisting
 * of a single state. If `D` is `Forward`, the leaf extends the specified span
 * forward in time; if `Backward`, it extends the span backward in time.
 *
 * The step-size adaptation handler is called with the acceptance of each
 * macro step attempt.
 *
 * The step size is reduced so that the Hamiltonian is conserved
 * within the specified error. The mass matrix and macro step size
 * are passed on to the leapfrog algorithm.
 *
 * @tparam D The direction in time to extend.
 * @tparam F The type of the log density/gradient function.
 * @tparam A The type of the adaptation handler.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] span The span to extend.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The macro step size.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum error allowed in the Hamiltonian.
 * @param[in,out] workspace The storage holding the preallocated vectors.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @param[out] out The span resulting from extending the specified span.
 * @return `true` if the span could be extended reversibly within threshold.
 */
template <Direction D, LogpGrad F, StepSizeAdapter A>
static bool build_leaf(const F& logp_grad, const SpanW& span,
                       const Eigen::VectorXd& inv_mass, double step,
                       std::size_t max_step_halvings,
                       std::size_t min_micro_steps, double max_error,
                       SpanWorkspace& workspace, A& adapt_handler, SpanW& out) {
  Eigen::VectorXd& theta_next = workspace.theta_next_;
  Eigen::VectorXd& rho_next = workspace.rho_next_;
  Eigen::VectorXd& grad_theta_next = workspace.grad_next_;
  // values are dummies; will be reset by macro step
  double logp_pos_next = -std::numeric_limits<double>::infinity();
  double logp_next = -std::numeric_limits<double>::infinity();
  if (!macro_step<D>(logp_grad, inv_mass, step, max_step_halvings,
                     min_micro_steps, max_error, span, theta_next, rho_next,
                     grad_theta_next, logp_pos_next, logp_next, workspace,
                     adapt_handler)) {
    return false;
  }
  out.init_from(std::move(theta_next), std::move(rho_next),
                std::move(grad_theta_next), logp_pos_next, logp_next);
  return true;
}

/**
 * @brief Build a span of two to the power of the depth states extending from
 * the specified span into the output span.
 *
 * @tparam D The direction in time to extend.
 * @tparam F The type of the log density/gradient function.
 * @tparam Rand The type for the source of randomness.
 * @tparam A The type of the step-size adaptation callback function.
 * @param[in,out] rng The random number generator.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] step The macro step size.
 * @param[in] depth The maximum Nuts depth.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum error allowed at macro steps.
 * @param[in] last_span The span to extend.
 * @param[in,out] workspace The storage holding the span slots.
 * @param[out] out The new span; must not alias the workspace slots used by
 * the two halves of this level.
 * @param[in] level The workspace level holding the two halves, from one.
 * @param[in,out] adapt_handler The step-size adaptation handler.
 * @return `true` if the span could be constructed.
 */
template <Direction D, LogpGrad F, std::uniform_random_bit_generator RNG,
          StepSizeAdapter A>
static bool build_span(Random<RNG>& rng, const F& logp_grad,
                       const Eigen::VectorXd& inv_mass, double step,
                       std::size_t depth, std::size_t max_step_halvings,
                       std::size_t min_micro_steps, double max_error,
                       const SpanW& last_span, SpanWorkspace& workspace,
                       SpanW& out, std::size_t level, A& adapt_handler) {
  if (depth == 0) {
    return build_leaf<D>(logp_grad, last_span, inv_mass, step,
                         max_step_halvings, min_micro_steps, max_error,
                         workspace, adapt_handler, out);
  }
  SpanW& half1 = workspace.span(level, 0);
  SpanW& half2 = workspace.span(level, 1);
  if (!build_span<D>(rng, logp_grad, inv_mass, step, depth - 1,
                     max_step_halvings, min_micro_steps, max_error, last_span,
                     workspace, half1, level + 1, adapt_handler)) {
    return false;
  }
  if (!build_span<D>(rng, logp_grad, inv_mass, step, depth - 1,
                     max_step_halvings, min_micro_steps, max_error, half1,
                     workspace, half2, level + 1, adapt_handler)) {
    return false;
  }
  if (uturn<D>(half1, half2, inv_mass, workspace.scaled_diff_)) {
    return false;
  }
  combine<Update::Barker, D>(rng, half1, half2, out);
  return true;
}

/**
 * @brief Return the next state in the Markov chain given the previous state.
 *
 * @tparam F The type of the log density/gradient function.
 * @tparam Rand The type for the source of randomness.
 * @tparam A The type of the step-size adaptation callback function.
 * @param[in,out] rand The random number generator.
 * @param[in] logp_grad The log density/gradient function.
 * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
 * @param[in] chol_mass The diagonal of the diagonal Cholesky factor of the mass
 * matrix.
 * @param[in] step The macro step size.
 * @param[in] max_depth The maximum number of trajectory doublings in Nuts.
 * @param[in] max_step_halvings The maximum number of halvings of the step size.
 * @param[in] min_micro_steps The minimum number of micro steps per macro step.
 * @param[in] max_error The maximum difference in Hamiltonians.
 * @param[in] theta The current state.
 * @param[out] depth The number of doublings included in the final
 * trajectory, which holds `2^depth` states.
 * @param[in,out] theta_grad The gradient of the log density at `theta` on
 * input; the gradient at the selected state on output.
 * @param[in,out] logp_pos_select The log density of `theta` on input; the
 * log density of the selected position on output.
 * @param[in,out] workspace The workspace for building the trajectory,
 * constructed for the dimensionality of `theta` and `max_depth` doublings.
 * @param[in,out] step_size_adapter The step-size adaptation handler.
 * @return The next position in the Markov chain.
 */
template <LogpGrad F, class Rand, StepSizeAdapter A>
inline Eigen::VectorXd transition_w(
    Rand& rand, const F& logp_grad, const Eigen::VectorXd& inv_mass,
    const Eigen::VectorXd& chol_mass, double step, std::size_t max_depth,
    std::size_t max_step_halvings, std::size_t min_micro_steps,
    double max_error, Eigen::VectorXd&& theta, std::size_t& depth,
    Eigen::VectorXd& theta_grad, double& logp_pos_select,
    SpanWorkspace& workspace, A& step_size_adapter) {
  Eigen::Index dim = chol_mass.size();
  Eigen::VectorXd& rho = workspace.rho_next_;
  rand.standard_normal(dim, rho);
  rho.array() *= chol_mass.array();
  double logp_joint = logp_pos_select + logp_momentum(rho, inv_mass);
  std::size_t accum = 0;
  workspace.accumulator(accum).init_from(std::move(theta), std::move(rho),
                                         std::move(theta_grad), logp_pos_select,
                                         logp_joint);
  depth = 0;
  while (depth < max_depth) {
    auto expand_in_direction = [&](auto direction) -> bool {
      constexpr Direction D = direction;
      SpanW& span_accum = workspace.accumulator(accum);
      SpanW& next_span = workspace.span(0, 0);
      if (!build_span<D>(rand, logp_grad, inv_mass, step, depth,
                         max_step_halvings, min_micro_steps, max_error,
                         span_accum, workspace, next_span, 1,
                         step_size_adapter)) {
        return true;
      }
      bool combined_uturn =
          uturn<D>(span_accum, next_span, inv_mass, workspace.scaled_diff_);
      SpanW& combined = workspace.accumulator(1 - accum);
      combine<Update::Metropolis, D>(rand, span_accum, next_span, combined);
      accum = 1 - accum;
      ++depth;
      return combined_uturn;
    };

    bool go_forward = rand.uniform_binary();
    bool made_uturn = go_forward ? expand_in_direction(Forward_t{})
                                 : expand_in_direction(Backward_t{});

    if (made_uturn) {
      break;
    }
  }
  SpanW& span_accum = workspace.accumulator(accum);
  theta.swap(span_accum.theta_select_);
  theta_grad.swap(span_accum.grad_select_);
  logp_pos_select = span_accum.logp_pos_select_;
  return std::move(theta);
}

/**
 * @brief A functor of one argument that does nothing.
 *
 * The use is as an adaptation handler when there is no adaptation. Because
 * it has no body, it will be inlined away at optimization level `-O2` or
 * above.
 */
class NoOpStepSizeAdapter {
 public:
  /**
   * Do nothing, ignoring the step size acceptance argument.
   *
   */
  constexpr void operator()(double) const noexcept {}

  /**
   * Return an invalid step size.
   */
  [[noreturn]] double step_size() const {
    throw std::logic_error(
        "should not call step_size() in NoOpStepSizeAdapter");
  }
};

}  // namespace walnutpie::detail

namespace walnutpie {

/**
 * @brief The Walnuts Markov chain Monte Carlo (MCMC) sampler.
 *
 * The sampler is constructed with a base random number generator, a log density
 * and gradient function, an initialization, and several tuning parameters.
 * It provides a no-argument functor for generating the next element of the
 * Markov chain.
 *
 * @tparam F The type of the log density and gradient function.
 * @tparam RNG The type of the base random number generator.
 * @tparam Handler The type of the sampling event handler.
 */
template <LogpGrad F, std::uniform_random_bit_generator RNG, SampleHandler H>
class WalnutsSampler {
 public:
  /**
   * @brief Construct a Walnuts sampler from the specified RNG, target log
   * density/gradient initialization, and tuning parameters.
   *
   * @param[in,out] rng The base random number generator.
   * @param[in,out] sample_handler The sampling and on-stop event handler.
   * @param[in] logp_grad The target log density and gradient function (see the
   * class documentation.
   * @param[in] theta The initial position.
   * @param[in] inv_mass The diagonal of the diagonal inverse mass matrix.
   * @param[in] macro_time The macro time discretization interval.
   * @param[in] max_nuts_depth The maximum number of trajectory doublings for
   * Nuts.
   * @param[in] max_step_halvings The maximum number of times the step size is
   * halved.
   * @param[in] min_micro_steps The minimum number of micro steps per macro
   * step.
   * @param[in] max_error The log of the maximum error in joint densities
   * allowed in Hamiltonian trajectories.
   * @throw std::invalid_argument If `inv_mass_matrix` has non-positive or
   * infinite entries.
   * @throw std::invalid_argument If `macro_time` is not positive or not
   * finite.
   * @throw std::invalid_argument If `max_nuts_depth` is not positive.
   * @throw std::invalid_argument If `max_step_halvings` is not positive.
   * @throw std::invalid_argument If `min_micro_steps` is not positive.
   * @throw std::invalid_argument If `max_error` is not positive or not finite.

   */
  WalnutsSampler(RNG& rng, H& sample_handler, const F& logp_grad,
                 const Eigen::VectorXd& theta, const Eigen::VectorXd& inv_mass,
                 double macro_time, std::size_t max_nuts_depth,
                 std::size_t max_step_halvings, std::size_t min_micro_steps,
                 double max_error)
      : rand_(rng),
        sample_handler_(sample_handler),
        logp_grad_(logp_grad, sample_handler),
        workspace_(theta.size(), max_nuts_depth),
        theta_(theta),
        inv_mass_(inv_mass),
        cholesky_mass_(inv_mass.array().sqrt().inverse().matrix()),
        macro_time_(macro_time),
        max_nuts_depth_(max_nuts_depth),
        max_step_halvings_(max_step_halvings),
        min_micro_steps_(min_micro_steps),
        max_error_(max_error),
        no_op_step_size_adapter_() {
    detail::validate_positive(inv_mass, "inv_mass");
    detail::validate_positive(macro_time, "macro_time");
    detail::validate_positive(max_nuts_depth, "max_nuts_depth");
    detail::validate_positive(max_step_halvings, "max_step_halvings");
    detail::validate_positive(min_micro_steps, "min_micro_steps");
    detail::validate_positive(max_error, "max_error");
    logp_grad_(theta_, logp_, grad_);
  }

  /**
   * @brief Construct a sampler by copying the specified sampler.
   *
   * @param[in] sampler Sampler to copy.
   */
  WalnutsSampler(const WalnutsSampler& sampler) = default;

  /**
   * @brief Construct a sampler by moving the specified sampler.
   *
   * @param[in] sampler Sampler to move.
   */
  WalnutsSampler(WalnutsSampler&& sampler) = default;

  /**
   * @brief Generate the next draw and send it to the handler and return
   * its log density.
   *
   * @return The unnormalized log density of the next draw.
   */
  double operator()() {
    std::size_t depth;
    theta_ =
        transition_w(rand_, logp_grad_, inv_mass_, cholesky_mass_, macro_time_,
                     max_nuts_depth_, max_step_halvings_, min_micro_steps_,
                     max_error_, std::move(theta_), depth, grad_, logp_,
                     workspace_, no_op_step_size_adapter_);
    sample_handler_.get().on_sample(theta_, logp_);
    return logp_;
  }

  /**
   * @brief  Return a constant reference the diagonal of the diagonal inverse
   * mass matrix.
   *
   * The value of the inverse mass matrix will change on subsequent calls to
   * `operator()(S&)`.
   *
   * @return The diagonal of the inverse mass matrix.
   */
  const Eigen::VectorXd& inverse_mass_matrix_diagonal() const noexcept {
    return inv_mass_;
  }

  /**
   * @brief Return the macro time discretization interval for Nuts.
   *
   * @return The time discretization interval for Nuts.
   */
  double macro_time() const noexcept { return macro_time_; }

  /**
   * @brief Return the maximum error allowed among Hamiltonians.
   *
   * @return The maximum error allowed among Hamiltonians.
   */
  double max_error() const noexcept { return max_error_; }

  /**
   * @brief Return the number of dimensions.
   *
   * @return The number of dimensions.
   */
  std::size_t dim() const noexcept {
    return static_cast<std::size_t>(theta_.size());
  }

 private:
  /** The underlying randomizer. */
  detail::Random<RNG> rand_;

  /** Reference to the sampling event handler. */
  std::reference_wrapper<H> sample_handler_;

  /** The target log density/gradient function. */
  const detail::NoExceptLogpGrad<F, H> logp_grad_;

  /** The preallocated storage for trajectory construction. */
  detail::SpanWorkspace workspace_;

  /** The current position. */
  Eigen::VectorXd theta_;

  /** The gradient of the log density at `theta_`. */
  Eigen::VectorXd grad_;

  /** The log density at `theta_`. */
  double logp_;

  /** The diagonal of the diagonal inverse mass matrix. */
  Eigen::VectorXd inv_mass_;

  /** The diagonal of the diagonal Cholesky factor of the mass matrix. */
  Eigen::VectorXd cholesky_mass_;

  /** The macro time discretization interval for Nuts. */
  const double macro_time_;

  /** The maximum number of doublings in Nuts trajectories. */
  const std::size_t max_nuts_depth_;

  /** The maximum number of halvings of the step size. */
  const std::size_t max_step_halvings_;

  /** The minimum number of micro steps per macro step. */
  const std::size_t min_micro_steps_;

  /** The max difference of Hamiltonians along a macro step. */
  const double max_error_;

  /** A handler for adaptation which does nothing. */
  const detail::NoOpStepSizeAdapter no_op_step_size_adapter_;
};

}  // namespace walnutpie
