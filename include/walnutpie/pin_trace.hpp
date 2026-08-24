#pragma once

// W-43 (blr short-warmup pin diagnosis): per-iteration warmup trace,
// activated by the environment variable WALNUTPIE_PIN_TRACE=1
// (precedent: WALNUTPIE_DEBUG_ALPHA / WALNUTPIE_DEBUG_WARMUP /
// WALNUTPIE_GRAD_ACCOUNTING). NO behavior change: nothing but this
// header's own scratch state is written, no floating-point sampler
// state is touched, no RNG is consumed, and no output is produced
// here; the CLI handler (StanHandler::on_warmup) reads the scratch
// after each warmup iteration and prints one record per iteration.
//
// Intended use is SINGLE-CHAIN runs (the W-38 accounting precedent);
// the scratch below is deliberately not atomic — under the multi-chain
// threaded mode records would interleave (documented limitation).
//
// Recorded per transition (reset by begin_transition()):
//   z_norm      norm of the fresh momentum draw z (M4: stochastic
//               escape via lucky momentum),
//   step_in     the macro step size the transition ran with,
//   macro_steps macro_step invocations,
//   attempts    forward dyadic attempts, evals their logp_grad calls,
//   min_abs_dh  min |dH| over ALL forward attempts of ALL macro steps
//               (tests M5: tolerance-passing-but-ladder-rejected),
//   alpha/dlogp/step_abs  the LAST macro step's min-micro attempt
//               acceptance statistic and dH (exactly what the step
//               adapter saw),
//   tol_pass_any / h_accept / ladder_rejects / exhausted  outcome flags.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace walnutpie::detail::pin_trace {

/**
 * @brief Return whether the pin trace is enabled, reading
 * WALNUTPIE_PIN_TRACE from the environment exactly once.
 */
inline bool on() {
  static const bool flag = [] {
    const char* env = std::getenv("WALNUTPIE_PIN_TRACE");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();
  return flag;
}

// ---- per-transition scratch (valid between begin_transition() and
// the next begin_transition(); read by the CLI handler) ----

inline double z_norm = -1.0;         // ||z|| of this transition's momentum
inline double step_in = -1.0;        // macro step size used by this transition
inline std::uint64_t macro_steps = 0;
inline std::uint64_t attempts = 0;   // forward dyadic attempts
inline std::uint64_t evals = 0;      // forward-attempt logp_grad calls
inline std::uint64_t ladder_calls = 0;
inline std::uint64_t ladder_rejects = 0;  // coarser lattice within tolerance
inline std::uint64_t exhausted = 0;       // macro steps with all attempts failing
inline double min_abs_dh = std::numeric_limits<double>::infinity();
inline bool tol_pass_any = false;    // any attempt passed the |dH| cap
inline long h_accept = -2;           // halving level of a reversible-accepted
                                     // attempt (-1 = tol pass, ladder reject;
                                     // -2 = none)
inline double alpha = -1.0;          // last macro step's min-attempt alpha
inline double dlogp =                // last macro step's min-attempt dH
    std::numeric_limits<double>::quiet_NaN();
inline double step_abs = -1.0;       // |step| of that min attempt

// Run-lifetime totals (never reset by begin_transition) so the CLI can
// report the backward-ladder share of the gradient budget over the whole
// run, sampling phase included. Env-gated scratch only.
inline std::uint64_t total_evals_forward = 0;
inline std::uint64_t total_evals_ladder = 0;

/** @brief Reset the per-transition scratch (call at transition start). */
inline void begin_transition() {
  if (!on()) {
    return;
  }
  z_norm = -1.0;
  step_in = -1.0;
  macro_steps = 0;
  attempts = 0;
  evals = 0;
  ladder_calls = 0;
  ladder_rejects = 0;
  exhausted = 0;
  min_abs_dh = std::numeric_limits<double>::infinity();
  tol_pass_any = false;
  h_accept = -2;
  alpha = -1.0;
  dlogp = std::numeric_limits<double>::quiet_NaN();
  step_abs = -1.0;
}

/** @brief Record the momentum draw's norm (transition start). */
inline void observe_z(double norm) {
  if (!on()) {
    return;
  }
  z_norm = norm;
}

/** @brief Record the macro step size this transition runs with. */
inline void observe_step(double step) {
  if (!on()) {
    return;
  }
  step_in = step;
}

/** @brief Count one macro_step invocation. */
inline void observe_macro_step() {
  if (!on()) {
    return;
  }
  ++macro_steps;
}

/**
 * @brief Record one forward dyadic attempt: its |dH| and eval count.
 */
inline void observe_attempt(double abs_dh, std::size_t num_evals) {
  if (!on()) {
    return;
  }
  ++attempts;
  evals += static_cast<std::uint64_t>(num_evals);
  total_evals_forward += static_cast<std::uint64_t>(num_evals);
  if (abs_dh < min_abs_dh) {
    min_abs_dh = abs_dh;
  }
}

/**
 * @brief Record the min-micro attempt's acceptance statistic (exactly
 * the value the step-size adapter observes) and its dH / step.
 */
inline void observe_min_attempt(double min_accept, double d,
                                double step_magnitude) {
  if (!on()) {
    return;
  }
  alpha = min_accept;
  dlogp = d;
  step_abs = step_magnitude;
}

/**
 * @brief Record a tolerance-passing attempt's reversibility outcome.
 *
 * @param reversible_ok whether the backward ladder confirmed the step.
 * @param halvings the halving level of the passing attempt.
 */
inline void observe_outcome(bool reversible_ok, std::size_t halvings) {
  if (!on()) {
    return;
  }
  tol_pass_any = true;
  h_accept = reversible_ok ? static_cast<long>(halvings) : -1;
}

/** @brief Record one backward-ladder walk and whether it rejected. */
inline void observe_ladder(bool within_tol, std::size_t num_evals) {
  if (!on()) {
    return;
  }
  ++ladder_calls;
  evals += static_cast<std::uint64_t>(num_evals);
  total_evals_ladder += static_cast<std::uint64_t>(num_evals);
  if (within_tol) {
    ++ladder_rejects;
  }
}

/** @brief Count one macro step whose attempts all failed tolerance. */
inline void observe_exhausted() {
  if (!on()) {
    return;
  }
  ++exhausted;
}

}  // namespace walnutpie::detail::pin_trace
