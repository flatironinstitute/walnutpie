#pragma once

#include <concepts>
#include <cstddef>
#include <ranges>

#include <Eigen/Dense>

namespace walnutpie::detail {

/**
 * @brief Concept for a type with a `.size()` member function.
 *
 * A type `T` satisfies `Sized` if `t.size()` is callable on a const
 * instance and returns a value comparable for equality. This matches
 * standard containers, `std::string`, `std::span`, and Eigen vectors
 * and matrices.
 */
template <typename T>
concept Sized = requires(const T& t) {
  { t.size() } -> std::equality_comparable;
};

/**
 * @brief Concept for an Eigen dense type with floating-point scalar.
 *
 * A type `T` satisfies `EigenDenseType` if it provides:
 *  - a nested `Scalar` type that is a floating-point type,
 *  - a `size()` member returning a value convertible to `Eigen::Index`,
 *  - element access via `t(i)` returning a value convertible to `Scalar`.
 *
 * Models include `Eigen::VectorXd`, `Eigen::MatrixXd`, and other
 * fixed- or dynamic-size Eigen dense types with floating-point scalars.
 * For matrices, `t(i)` accesses elements in storage order (column-major
 * by default).
 */
template <typename T>
concept EigenDenseType = requires(const T& t) {
  typename T::Scalar;
  { t.size() } -> std::convertible_to<Eigen::Index>;
  { t(0) } -> std::convertible_to<typename T::Scalar>;
} && std::floating_point<typename T::Scalar>;

/**
 * @brief Concept for a leaf type in the floating-point validation tree.
 *
 * A type `T` satisfies `FloatingPointLeaf` if it is either a
 * floating-point type or an Eigen dense type with floating-point
 * scalar. These are the types whose elements can be checked directly,
 * without further recursive descent through containers.
 */
template <typename T>
concept FloatingPointLeaf = std::floating_point<T> || EigenDenseType<T>;

/**
 * @brief Concept for a floating-point leaf or a one-level container
 * of such leaves.
 *
 * A type `T` satisfies `NestedFloatingPoint` if any of the following
 * hold:
 *  - `T` satisfies `FloatingPointLeaf` (a floating-point value or an
 *    Eigen dense type with floating-point scalar),
 *  - `T` is a `std::ranges::range` whose element type satisfies
 *    `FloatingPointLeaf`.
 *
 * This covers `double`, `Eigen::VectorXd`, `std::vector<double>`, and
 * `std::vector<Eigen::VectorXd>`, but does not recurse to deeper
 * nesting such as `std::vector<std::vector<double>>`.
 */
template <typename T>
concept NestedFloatingPoint =
    FloatingPointLeaf<T> ||
    (std::ranges::range<T> && FloatingPointLeaf<std::ranges::range_value_t<T>>);

/**
 * @brief Concept for a Markov chain sampler.
 *
 * A type `S` satisfies `Sampler` if it provides:
 *  - `s()` callable on a non-const instance, returning a log density value
 *    convertible to `double`. Each call advances the chain by one
 *    iteration and returns the log density of the new draw. The
 *    sampler is responsible for delivering the draw itself to any
 *    chain-local handler it holds.
 *  - `s.dim()` callable on a const instance, returning a value
 *    convertible to `std::size_t`. This reports the dimensionality
 *    of the parameter space being sampled.
 *
 * The single-argument operator must be invocable through a
 * `std::reference_wrapper<S>`, which means `operator()` must not be
 * `const`-only — it should be a non-const member function (or
 * mutable-callable in some other way), since sampling advances
 * internal state such as the position, RNG, and any adapted
 * parameters.
 */
template <typename S>
concept Sampler = requires(S& s, const S& cs) {
  { s() } -> std::convertible_to<double>;
  { cs.dim() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Concept for a step size adaptation handler.
 *
 * A type `H` satisfies `StepSizeAdapter` if it provides:
 *  - `h(accept_prob)` callable on a non-const instance with a
 *    `double` argument, returning `void`. Each call observes one
 *    acceptance probability and updates the internal estimate.
 *  - `h.step_size()` callable on a const instance, returning a
 *    value convertible to `double`. Reports the current adapted
 *    step size.
 */
template <typename H>
concept StepSizeAdapter = requires(H& h, const H& ch, double accept_prob) {
  { h(accept_prob) } -> std::same_as<void>;
  { ch.step_size() } -> std::convertible_to<double>;
};

/**
 * @brief Concept for an adaptive sampler that tunes parameters during
 * warmup and produces a fixed-parameter sampler for post-warmup
 * draws.
 *
 * A type `A` satisfies `AdaptiveSampler` if it provides the following
 * members:
 *  - `a()` callable on a non-const instance, returning `void`. Each
 *    call advances the adaptation by one iteration, updating tuning
 *    parameters and calling back on any chain-local handler.
 *  - `a.sampler()` callable on a non-const instance, returning a type
 *    that satisfies the `Sampler` concept. This finalizes adaptation
 *    and produces a sampler with fixed tuning parameters.
 *  - `a.step_size()` callable on a const instance, returning a value
 *    convertible to `double`. Reports the current adapted step size.
 *  - `a.inv_mass()` callable on a const instance, returning a type
 *    convertible to `Eigen::VectorXd`. Reports the current diagonal
 *    inverse mass matrix estimate.
 *  - `a.dim()` callable on a const instance, returning a value
 *    convertible to `std::size_t`. Reports the dimensionality of the
 *    parameter space.
 *  - `a.iter()` callable on a const instance, returning a value
 *    convertible to `std::size_t`. Reports the current iteration
 *    count.
 *  - `a.log_step_size()` callable on a const instance, returning a
 *    value convertible to `double`. Reports the log of the current
 *    adapted step size.
 *  - `a.log_mass()` callable on a const instance, returning a type
 *    convertible to `Eigen::VectorXd`. Reports the log of the
 *    diagonal mass matrix.
 */
template <typename A>
concept AdaptiveSampler = requires(A& a, const A& ca) {
  { a() } -> std::same_as<void>;
  { a.sampler() } -> detail::Sampler;
  { ca.step_size() } -> std::convertible_to<double>;
  { ca.inv_mass() } -> std::convertible_to<Eigen::VectorXd>;
  { ca.dim() } -> std::convertible_to<std::size_t>;
  { ca.iter() } -> std::convertible_to<std::size_t>;
  { ca.log_step_size() } -> std::convertible_to<double>;
  { ca.log_mass() } -> std::convertible_to<Eigen::VectorXd>;
};

}  // namespace walnutpie::detail

namespace walnutpie {

/**
 * @brief Concept for a handler of cross-chain events.
 *
 * This only handles R-hat updates now.
 *
 * A type `H` satisfies `Handler` if it provides:
 *  - `on_r_hat(double)` callable on a non-const instance, returning `void`,
 */
template <typename H>
concept GlobalHandler = requires(H& h, const H& ch, double r_hat) {
  { h.on_r_hat(r_hat) } -> std::same_as<void>;
};

/**
 * @brief Concept for an interrupt callback.
 *
 * A type `H` satisfies `Handler` if it provides:
 *  - `received_interrupt()` will return `true` if the process should
 *    be interrupted.
 */
template <typename H>
concept InterruptCallback = requires(H& h, const H& ch, double r_hat) {
  { h.throw_if_interrupted() } -> std::same_as<void>;
};

/**
 * @brief Concept for handler for errors
 * The following member is required
 * - `on_logp_exception(const Eigen::VectorXd&, std::exception&)` called
 *    when the log density function throws an error
 */
template <typename H>
concept ErrorCallback =
    requires(H& h, const Eigen::VectorXd& position, const std::exception& exn) {
      { h.on_logp_exception(position, exn) } -> std::same_as<void>;
      { noexcept(h.on_logp_exception(position, exn)) };
    };

/**
 * @brief Concept for a handler of sampling events.
 *
 * A type `H` satisfies `SampleHandler` if it satisfies `ErrorCallback` and
 * additional provides the following member functions, each callable
 * on a non-const instance and returning `void`:
 *  - `on_sample(const Eigen::VectorXd&, double)` called once per
 *    draw with the position and log density.
 */
template <typename H>
concept SampleHandler =
    ErrorCallback<H> &&
    requires(H& h, const Eigen::VectorXd& position, double lp) {
      { h.on_sample(position, lp) } -> std::same_as<void>;
    };

/**
 * @brief An extension of the `SampleHandler` concept for additionally
 * handling warmup events.
 *
 *
 * A type `C` satisfies `ChainHandler` if it provides the following
 * member functions, each callable on a non-const instance and returning
 * `void`:
 *  - `on_warmup(const Eigen::VectorXd&, double, double, const
 * Eigen::VectorXd&)` called once per warmup draw with the position, log
 * density, step size, and diagonal inverse mass matrix.
 *  - `on_warmup_complete(double, const Eigen::VectorXd&)` called once
 *    when warmup finishes, with the final step size and diagonal inverse
 *    mass matrix.
 *  - `on_sample(const Eigen::VectorXd&, double)` called once per
 *    post-warmup draw with the position and log density.
 */
template <typename C>
concept ChainHandler =
    SampleHandler<C> && requires(C& c, const Eigen::VectorXd& position,
                                 const Eigen::VectorXd& diag_inv_mass,
                                 double lp, double step_size) {
      {
        c.on_warmup(position, lp, step_size, diag_inv_mass)
      } -> std::same_as<void>;
      { c.on_warmup_complete(step_size, diag_inv_mass) } -> std::same_as<void>;
    };

/**
 * @brief Concept for a log density and gradient function.
 *
 * A type `F` satisfies `LogpGrad` if an object of type `const F&` can be
 * called with arguments `(const Eigen::VectorXd&, double&, Eigen::VectorXd&)`
 * and the call returns `void`. The first argument is the position at which
 * to evaluate, and the second and third are output parameters set to the
 * log density and its gradient, respectively.
 *
 * @tparam F The callable type to constrain.
 */
template <typename F>
concept LogpGrad = requires(const F& f, const Eigen::VectorXd& x, double& logp,
                            Eigen::VectorXd& grad) {
  { f(x, logp, grad) } -> std::same_as<void>;
};

/**
 * @brief Concept for a sequence of Markov chains over a shared set
 * of variables.
 *
 * A type `M` satisfies `MarkovChainCollection` if it provides the
 * following member functions, all callable on a const instance:
 *  - `num_chains()` returning a value convertible to `std::size_t`,
 *    the number of chains in the collection.
 *  - `num_draws()` returning a value convertible to `std::size_t`,
 *    the total number of draws across all chains.
 *  - `dims()` returning a value convertible to `std::size_t`, the
 *    dimensionality of each draw.
 *  - `min_chain_size()` returning a value convertible to
 *    `Eigen::Index`, the length of the shortest chain.
 *  - `chain_view(std::size_t)` returning a type convertible to
 *    `Eigen::MatrixXd`, a view of one chain (one draw per row).
 *  - `draws(Eigen::Index)` returning a type convertible to
 *    `Eigen::VectorXd`, the sequence of draws for one variable across
 *    all chains.
 */
template <typename M>
concept MarkovChainSequence =
    requires(const M& m, std::size_t chain_index, Eigen::Index dim_index) {
      { m.num_chains() } -> std::convertible_to<std::size_t>;
      { m.num_draws() } -> std::convertible_to<std::size_t>;
      { m.dims() } -> std::convertible_to<std::size_t>;
      { m.min_chain_size() } -> std::convertible_to<Eigen::Index>;
      { m.chain_view(chain_index) } -> std::convertible_to<Eigen::MatrixXd>;
      { m.draws(dim_index) } -> std::convertible_to<Eigen::VectorXd>;
    };

}  // namespace walnutpie
