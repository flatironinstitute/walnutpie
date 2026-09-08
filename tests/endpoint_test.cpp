#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <functional>
#include <vector>
#include <walnutpie/adaptive_walnuts.hpp>

namespace {
using namespace walnutpie;
using Bits = std::vector<std::uint64_t>;
struct Target {
  int kind = 0;
  mutable std::size_t calls = 0;
  int failure = 0;
  double offset = 0;
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& g) const {
    ++calls;
    if (failure == 1 || (failure == 4 && x.squaredNorm() > 0)) {
      throw std::runtime_error("test failure");
    }
    if (kind == 0) {
      lp = -0.5 * x.squaredNorm();
      g = -x;
    }
    if (kind == 1) {
      lp = -0.5 * (2 * x[0] * x[0] + 2 * x[0] * x[1] + 3 * x[1] * x[1]);
      g.resize(2);
      g << -2 * x[0] - x[1], -x[0] - 3 * x[1];
    }
    if (kind == 2) {
      lp = -0.5 * x.squaredNorm() - 0.25 * x.array().pow(4).sum();
      g = -x.array() - x.array().cube();
    }
    lp += offset;
    if (failure == 2) {
      lp = -std::numeric_limits<double>::infinity();
    }
    if (failure == 3) {
      g.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }
};
struct Handler {
  Bits trace;
  std::size_t errors = 0, warmups = 0, samples = 0, freezes = 0;
  std::function<void()> after_sample, after_warmup, after_freeze;
  void add(double v) { trace.push_back(std::bit_cast<std::uint64_t>(v)); }
  void add(const Eigen::VectorXd& x) {
    for (double v : x) {
      add(v);
    }
  }
  void on_sample(const Eigen::VectorXd& x, double lp) {
    trace.push_back(1);
    add(x);
    add(lp);
    ++samples;
    if (after_sample) {
      after_sample();
    }
  }
  void on_warmup(const Eigen::VectorXd& x, double lp, double step,
                 const Eigen::VectorXd& mass) {
    trace.push_back(2);
    add(x);
    add(lp);
    add(step);
    add(mass);
    ++warmups;
    if (after_warmup) {
      after_warmup();
    }
  }
  void on_warmup_complete(double step, const Eigen::VectorXd& mass) {
    trace.push_back(3);
    add(step);
    add(mass);
    ++freezes;
    if (after_freeze) {
      after_freeze();
    }
  }
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {
    ++errors;
  }
};
Eigen::VectorXd zero() { return Eigen::VectorXd::Zero(2); }
Eigen::VectorXd ones() { return Eigen::VectorXd::Ones(2); }
auto sampling_config() {
  return SamplingConfigBuilder()
      .max_trajectory_doublings(4)
      .max_step_halvings(8)
      .build();
}
struct Run {
  Bits trace;
  std::size_t calls;
};
Run chain(int kind, unsigned seed, EndpointReuse policy, int warmup = 100) {
  Target t{kind};
  Handler h;
  std::mt19937_64 rng(seed);
  auto init = InitChainConfig(0.2, zero(), ones());
  auto wc = WarmupConfigBuilder().build();
  auto sc = sampling_config();
  AdaptiveWalnuts a(rng, h, t, init, wc, sc, policy);
  for (int i = 0; i < warmup; ++i) {
    a();
  }
  auto before = t.calls;
  auto sampler = a.sampler();
  EXPECT_EQ(before, t.calls);
  for (int i = 0; i < 100; ++i) {
    auto lp = sampler();
    EXPECT_EQ(std::bit_cast<std::uint64_t>(lp), h.trace.back());
  }
  EXPECT_EQ(h.warmups, warmup);
  EXPECT_EQ(h.samples, 100);
  EXPECT_EQ(h.freezes, 1);
  EXPECT_EQ(h.errors, 0);
  return {h.trace, t.calls};
}
TEST(EndpointReuse, AnalyticParityAndCounts) {
  for (int kind = 0; kind < 3; ++kind) {
    for (unsigned seed : {17u, 20260819u}) {
      auto off = chain(kind, seed, EndpointReuse::Disabled);
      auto on = chain(kind, seed, EndpointReuse::Deterministic);
      EXPECT_EQ(off.trace, on.trace);
      EXPECT_EQ(off.calls - on.calls, 199);
      std::cout << "endpoint kind=" << kind << " seed=" << seed
                << " baseline=" << off.calls << " cached=" << on.calls << "\n";
    }
  }
}
TEST(EndpointReuse, ZeroWarmupAndColdSampler) {
  auto off = chain(0, 17, EndpointReuse::Disabled, 0);
  auto on = chain(0, 17, EndpointReuse::Deterministic, 0);
  EXPECT_EQ(off.trace, on.trace);
  EXPECT_EQ(off.calls - on.calls, 99);
  std::size_t calls[2];
  Bits traces[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    WalnutsSampler s(
        rng, h, t, zero(), ones(), 0.2, 4, 8, 1, 1.0,
        enabled ? EndpointReuse::Deterministic : EndpointReuse::Disabled);
    for (int i = 0; i < 100; ++i) {
      s();
    }
    calls[enabled] = t.calls;
    traces[enabled] = h.trace;
  }
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(calls[0] - calls[1], 99);
}
TEST(EndpointReuse, InvalidEndpointsRetryAndRecover) {
  for (int failure : {1, 2, 3}) {
    std::size_t calls[2], errors[2];
    Bits traces[2];
    for (int enabled = 0; enabled < 2; ++enabled) {
      Target t;
      t.failure = failure;
      Handler h;
      std::mt19937_64 rng(17);
      WalnutsSampler s(
          rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0,
          enabled ? EndpointReuse::Deterministic : EndpointReuse::Disabled);
      for (int i = 0; i < 3; ++i) {
        s();
      }
      EXPECT_EQ(t.calls, 6);
      EXPECT_EQ(h.errors, failure == 1 ? 6 : 0);
      t.failure = 0;  // invalid endpoints must not prevent retry/recovery
      s();
      EXPECT_EQ(t.calls, 8);
      s();
      calls[enabled] = t.calls;
      errors[enabled] = h.errors;
      traces[enabled] = h.trace;
    }
    EXPECT_EQ(traces[0], traces[1]);
    EXPECT_EQ(errors[0], errors[1]);
    EXPECT_EQ(calls[0] - calls[1], 1);
  }
}
TEST(EndpointReuse, RejectedTrajectoryKeepsFiniteEndpoint) {
  Bits traces[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    t.failure = 4;
    Handler h;
    std::mt19937_64 rng(17);
    WalnutsSampler s(
        rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0,
        enabled ? EndpointReuse::Deterministic : EndpointReuse::Disabled);
    for (int i = 0; i < 3; ++i) {
      s();
    }
    EXPECT_EQ(t.calls, enabled ? 4 : 6);
    EXPECT_EQ(h.errors, 3);
    EXPECT_EQ(h.samples, 3);
    traces[enabled] = h.trace;
  }
  EXPECT_EQ(traces[0], traces[1]);
}
TEST(EndpointReuse, InvalidationAndDefaultMutableTarget) {
  Bits traces[2];
  std::size_t counts[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    // The default constructor API remains valid.
    WalnutsSampler plain(rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0);
    WalnutsSampler cached(rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0,
                          EndpointReuse::Deterministic);
    h.after_sample = [&] {
      t.offset += 100;
      cached.invalidate_endpoint_cache();
    };
    for (int i = 0; i < 3; ++i) {
      if (enabled) {
        cached();
      } else {
        plain();
      }
    }
    counts[enabled] = t.calls;
    traces[enabled] = h.trace;
  }
  EXPECT_EQ(counts[0], 6);
  EXPECT_EQ(counts[0], counts[1]);
  EXPECT_EQ(traces[0], traces[1]);
}
TEST(EndpointReuse, WarmupCallbackInvalidation) {
  Bits traces[2];
  std::size_t counts[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    auto init = InitChainConfig(0.2, zero(), ones());
    auto wc = WarmupConfigBuilder().build();
    auto sc = sampling_config();
    AdaptiveWalnuts a(
        rng, h, t, init, wc, sc,
        enabled ? EndpointReuse::Deterministic : EndpointReuse::Disabled);
    h.after_warmup = [&] {
      t.offset += 100;
      a.invalidate_endpoint_cache();
    };
    for (int i = 0; i < 3; ++i) {
      a();
    }
    auto sampler = a.sampler();
    sampler();
    traces[enabled] = h.trace;
    counts[enabled] = t.calls;
  }
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(counts[0], counts[1]);
}

TEST(EndpointReuse, FreezeCallbackInvalidationAndOrdering) {
  Bits traces[2];
  std::size_t counts[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    auto init = InitChainConfig(0.2, zero(), ones());
    auto wc = WarmupConfigBuilder().build();
    auto sc = sampling_config();
    AdaptiveWalnuts a(
        rng, h, t, init, wc, sc,
        enabled ? EndpointReuse::Deterministic : EndpointReuse::Disabled);
    a();
    auto before = t.calls;
    h.after_freeze = [&] {
      EXPECT_EQ(t.calls, before);
      t.offset += 100;
      a.invalidate_endpoint_cache();
    };
    auto s = a.sampler();
    EXPECT_EQ(t.calls, before);
    s();
    // Each call delivers a completion event, just as upstream does.
    before = t.calls;
    auto unused = a.sampler();
    EXPECT_EQ(h.freezes, 2);
    EXPECT_EQ(t.calls, before);
    counts[enabled] = t.calls;
    traces[enabled] = h.trace;
  }
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(counts[0], counts[1]);
}
TEST(EndpointReuse, CopyMoveValueOwnership) {
  for (bool populated : {false, true}) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    WalnutsSampler original(rng, h, t, zero(), ones(), 0.2, 4, 8, 1, 1.0,
                            EndpointReuse::Deterministic);
    if (populated) {
      original();
    }
    auto copy = original;
    auto moved = std::move(copy);
    auto state = rng;
    h.trace.clear();
    auto before = t.calls;
    original();
    auto trace = h.trace;
    auto calls = t.calls - before;
    rng = state;
    h.trace.clear();
    before = t.calls;
    moved();
    EXPECT_EQ(trace, h.trace);
    EXPECT_EQ(calls, t.calls - before);
    // Invalidating one copy must not clear the other's populated cache.
    auto twin = moved;
    moved.invalidate_endpoint_cache();
    state = rng;
    h.trace.clear();
    before = t.calls;
    moved();
    trace = h.trace;
    calls = t.calls - before;
    rng = state;
    h.trace.clear();
    before = t.calls;
    twin();
    EXPECT_EQ(trace, h.trace);
    EXPECT_EQ(calls, t.calls - before + 1);
  }
}
TEST(EndpointReuse, AdaptiveCopyMoveAndIndependentFreeze) {
  for (bool populated : {false, true}) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    auto init = InitChainConfig(0.2, zero(), ones());
    auto wc = WarmupConfigBuilder().build();
    auto sc = sampling_config();
    AdaptiveWalnuts a(rng, h, t, init, wc, sc, EndpointReuse::Deterministic);
    if (populated) {
      a();
    }
    auto copy = a;
    auto moved = std::move(copy);
    auto state = rng;
    h.trace.clear();
    auto before = t.calls;
    a();
    auto trace = h.trace;
    auto calls = t.calls - before;
    rng = state;
    h.trace.clear();
    before = t.calls;
    moved();
    EXPECT_EQ(trace, h.trace);
    EXPECT_EQ(calls, t.calls - before);
    auto s = a.sampler();
    auto twin = moved.sampler();
    a.invalidate_endpoint_cache();
    state = rng;
    h.trace.clear();
    before = t.calls;
    s();
    trace = h.trace;
    calls = t.calls - before;
    rng = state;
    h.trace.clear();
    before = t.calls;
    twin();
    EXPECT_EQ(trace, h.trace);
    EXPECT_EQ(calls, t.calls - before);
  }
}
TEST(EndpointReuse, LegacyTransitionEntryPoint) {
  Target t;
  std::mt19937_64 rng(17);
  detail::Random random(rng);
  detail::NoOpStepSizeAdapter adapter;
  std::size_t depth;
  Eigen::VectorXd grad;
  double lp;
  auto theta = detail::transition_w(random, t, ones(), ones(), 0.1, 1, 1, 1,
                                    1.0, zero(), depth, grad, lp, adapter);
  EXPECT_EQ(t.calls, 2);
  EXPECT_EQ(grad.size(), theta.size());
  EXPECT_TRUE(std::isfinite(lp));
}

TEST(EndpointReuse, WrongSizedCacheFallsBack) {
  Target t;
  std::mt19937_64 rng(17);
  detail::Random random(rng);
  detail::NoOpStepSizeAdapter adapter;
  detail::EndpointCache cache;
  cache.store(Eigen::VectorXd::Ones(3), 999.0);
  std::size_t depth;
  Eigen::VectorXd grad;
  double lp;
  auto theta =
      detail::transition_w_impl(random, t, ones(), ones(), 0.1, 1, 1, 1, 1.0,
                                zero(), depth, grad, lp, adapter, &cache);
  EXPECT_EQ(t.calls, 2);
  EXPECT_EQ(grad.size(), theta.size());
  EXPECT_LE(lp, 0);
}

TEST(EndpointReuse, ExplicitValidityAndFiniteCache) {
  detail::EndpointCache c;
  EXPECT_FALSE(c.valid);
  c.store(Eigen::VectorXd(), 0.0);
  EXPECT_TRUE(c.valid);
  c.store(ones(), std::numeric_limits<double>::infinity());
  EXPECT_FALSE(c.valid);
  c.store(
      Eigen::VectorXd::Constant(2, std::numeric_limits<double>::quiet_NaN()),
      0.0);
  EXPECT_FALSE(c.valid);
}
}  // namespace
