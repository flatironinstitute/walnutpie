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
template <typename S>
auto draw_with_reuse(S& sampler, bool reuse = true) {
  if (!reuse) {
    sampler.invalidate_endpoint_cache();
  }
  return sampler();
}
struct Run {
  Bits trace;
  std::size_t calls;
  std::mt19937_64 rng;
};
Run chain(int kind, unsigned seed, bool reuse, int warmup = 100) {
  Target t{kind};
  Handler h;
  std::mt19937_64 rng(seed);
  auto init = InitChainConfig(0.2, zero(), ones());
  auto wc = WarmupConfigBuilder().build();
  auto sc = sampling_config();
  AdaptiveWalnuts a(rng, h, t, init, wc, sc);
  for (int i = 0; i < warmup; ++i) {
    draw_with_reuse(a, reuse);
  }
  auto before = t.calls;
  auto sampler = a.sampler();
  EXPECT_EQ(before, t.calls);
  for (int i = 0; i < 100; ++i) {
    auto lp = draw_with_reuse(sampler, reuse);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(lp), h.trace.back());
  }
  EXPECT_EQ(h.warmups, warmup);
  EXPECT_EQ(h.samples, 100);
  EXPECT_EQ(h.freezes, 1);
  EXPECT_EQ(h.errors, 0);
  return {h.trace, t.calls, rng};
}
TEST(EndpointReuse, AnalyticParityAndCounts) {
  for (int kind = 0; kind < 3; ++kind) {
    for (unsigned seed : {17u, 20260819u}) {
      auto off = chain(kind, seed, false);
      auto on = chain(kind, seed, true);
      EXPECT_EQ(off.trace, on.trace);
      EXPECT_EQ(off.rng, on.rng);
      EXPECT_EQ(off.calls - on.calls, 199);
      std::cout << "endpoint kind=" << kind << " seed=" << seed
                << " baseline=" << off.calls << " cached=" << on.calls << "\n";
    }
  }
}
TEST(EndpointReuse, ZeroWarmupAndColdSampler) {
  auto off = chain(0, 17, false, 0);
  auto on = chain(0, 17, true, 0);
  EXPECT_EQ(off.trace, on.trace);
  EXPECT_EQ(off.rng, on.rng);
  EXPECT_EQ(off.calls - on.calls, 99);
  std::size_t calls[2];
  Bits traces[2];
  for (int enabled = 0; enabled < 2; ++enabled) {
    Target t;
    Handler h;
    std::mt19937_64 rng(17);
    WalnutsSampler s(rng, h, t, zero(), ones(), 0.2, 4, 8, 1, 1.0);
    for (int i = 0; i < 100; ++i) {
      draw_with_reuse(s, enabled);
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
      WalnutsSampler s(rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0);
      for (int i = 0; i < 3; ++i) {
        draw_with_reuse(s, enabled);
      }
      EXPECT_EQ(t.calls, 6);
      EXPECT_EQ(h.errors, failure == 1 ? 6 : 0);
      t.failure = 0;  // invalid endpoints must not prevent retry/recovery
      draw_with_reuse(s, enabled);
      EXPECT_EQ(t.calls, 8);
      draw_with_reuse(s, enabled);
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
    WalnutsSampler s(rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0);
    for (int i = 0; i < 3; ++i) {
      draw_with_reuse(s, enabled);
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
    WalnutsSampler cached(rng, h, t, zero(), ones(), 0.1, 1, 1, 1, 1.0);
    h.after_sample = [&] {
      t.offset += 100;
      cached.invalidate_endpoint_cache();
    };
    for (int i = 0; i < 3; ++i) {
      if (enabled) {
        cached();
      } else {
        draw_with_reuse(plain, false);
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
    AdaptiveWalnuts a(rng, h, t, init, wc, sc);
    h.after_warmup = [&] {
      t.offset += 100;
      a.invalidate_endpoint_cache();
    };
    for (int i = 0; i < 3; ++i) {
      draw_with_reuse(a, enabled);
    }
    auto sampler = a.sampler();
    draw_with_reuse(sampler, enabled);
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
    AdaptiveWalnuts a(rng, h, t, init, wc, sc);
    draw_with_reuse(a, enabled);
    auto before = t.calls;
    h.after_freeze = [&] {
      EXPECT_EQ(t.calls, before);
      t.offset += 100;
      a.invalidate_endpoint_cache();
    };
    auto s = a.sampler();
    EXPECT_EQ(t.calls, before);
    draw_with_reuse(s, enabled);
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
    WalnutsSampler original(rng, h, t, zero(), ones(), 0.2, 4, 8, 1, 1.0);
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
    AdaptiveWalnuts a(rng, h, t, init, wc, sc);
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
  Eigen::VectorXd wrong_gradient = Eigen::VectorXd::Ones(3);
  std::size_t depth;
  Eigen::VectorXd grad;
  double lp;
  auto theta = detail::transition_w_impl(random, t, ones(), ones(), 0.1, 1, 1,
                                         1, 1.0, zero(), depth, grad, lp,
                                         adapter, &wrong_gradient, 999.0);
  EXPECT_EQ(t.calls, 2);
  EXPECT_EQ(grad.size(), theta.size());
  EXPECT_LE(lp, 0);
}

struct MutatingDensityHandler : Handler {
  void on_sample(const Eigen::VectorXd& x, double& lp) {
    Handler::on_sample(x, lp);
    lp = 12345;
  }
  void on_warmup(const Eigen::VectorXd& x, double& lp, double step,
                 const Eigen::VectorXd& mass) {
    Handler::on_warmup(x, lp, step, mass);
    lp = 12345;
  }
};
TEST(EndpointReuse, CallbackDensityReferenceDoesNotChangeStoredEndpoint) {
  Bits traces[2];
  std::mt19937_64 states[2];
  std::size_t counts[2];
  for (bool reuse : {false, true}) {
    Target target;
    MutatingDensityHandler handler;
    std::mt19937_64 rng(17);
    auto init = InitChainConfig(.2, zero(), ones());
    auto warmup = WarmupConfigBuilder().build();
    auto config = sampling_config();
    AdaptiveWalnuts adaptive(rng, handler, target, init, warmup, config);
    for (int i = 0; i < 20; ++i) {
      draw_with_reuse(adaptive, reuse);
    }
    auto sampler = adaptive.sampler();
    for (int i = 0; i < 20; ++i) {
      EXPECT_EQ(draw_with_reuse(sampler, reuse), 12345);
    }
    traces[reuse] = handler.trace;
    states[reuse] = rng;
    counts[reuse] = target.calls;
  }
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(states[0], states[1]);
  EXPECT_EQ(counts[0] - counts[1], 39);
}
}  // namespace
