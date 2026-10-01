#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "walnutpie/adapt.hpp"
#include "walnutpie/adaptive_walnuts.hpp"
#include "walnutpie/concepts.hpp"
#include "walnutpie/config.hpp"
#include "walnutpie/sampler.hpp"
#include "walnutpie/walnuts.hpp"

namespace walnutpie {

/**
 * Return the chain records from running Walnuts with the specified
 * seed, sampling event handlers, and configuration.
 *
 * @tparam Handler The type of the event handlers.
 * @param[in] seed The seed for the pseudo-random number generator.
 * @param[in] chain_handlers The collection of chain-specific handlers, which
 * are called back.
 * @param[in] global_handler The handler for global cross-chain events.
 * @param[in] interrupt_callback The callback for stopping.
 * @param[in] log_p_grad The log density and gradient function, called back.
 * @param[in] config The configuration for Walnutpie.
 * @throws std::invalid_argument If the number of handlers doesn't match
 * the initialization configuration's number of chains.
 */
template <std::uniform_random_bit_generator RNG, ChainHandler H,
          GlobalHandler GH, InterruptCallback IC, LogpGrad F>
inline void walnuts(std::size_t seed, std::vector<H>& chain_handlers,
                    GH& global_handler, const IC& interrupt_callback,
                    const F& log_p_grad, const WalnutsConfig& config) {
  using AdaptiveSampler = AdaptiveWalnuts<F, RNG, H>;
  using Sampler = WalnutsSampler<F, RNG, H>;

  if (chain_handlers.size() != config.init().num_chains()) {
    throw std::invalid_argument(
        "chain_handlers.size() must be equal to config.init().num_chains()");
  }

  std::vector<RNG> rngs(0);
  rngs.reserve(config.init().num_chains());
  for (std::size_t m = 0; m < config.init().num_chains(); ++m) {
    std::seed_seq ss{seed, m + 1u};
    rngs.emplace_back(ss);
  }
  std::vector<AdaptiveSampler> adapters;
  adapters.reserve(config.init().num_chains());
  for (std::size_t m = 0; m < config.init().num_chains(); ++m) {
    adapters.emplace_back(rngs[m], chain_handlers[m], log_p_grad,
                          config.init().init_chain_config(m), config.warmup(),
                          config.sampling());
  }
  detail::adapt(config.init(), config.warmup(), adapters, interrupt_callback);

  std::vector<Sampler> samplers;
  samplers.reserve(adapters.size());
  for (std::size_t n = 0; n < adapters.size(); ++n) {
    samplers.emplace_back(std::move(adapters[n].sampler()));
  }

  detail::sample(config.sampling(), samplers, global_handler,
                 interrupt_callback);
}

}  // namespace walnutpie
