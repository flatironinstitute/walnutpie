#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <walnutpie/adaptive_walnuts.hpp>

namespace {

using namespace walnutpie;

// Standard normal target that counts evaluations and can fail away from the
// origin.
struct Target {
  mutable std::size_t calls = 0;
  mutable Eigen::VectorXd last_evaluated;
  mutable bool fail_away_from_origin = false;
  void operator()(const Eigen::VectorXd& x, double& lp,
                  Eigen::VectorXd& g) const {
    ++calls;
    last_evaluated = x;
    if (fail_away_from_origin && x.squaredNorm() > 0) {
      throw std::runtime_error("test failure");
    }
    lp = -0.5 * x.squaredNorm();
    g = -x;
  }
};

// Records sampled draws, the last warmup position, and target failures.
struct Handler {
  std::vector<std::pair<Eigen::VectorXd, double>> samples;
  Eigen::VectorXd last_warmup;
  std::size_t errors = 0;
  void on_sample(const Eigen::VectorXd& x, double lp) {
    samples.emplace_back(x, lp);
  }
  void on_warmup(const Eigen::VectorXd& x, double, double,
                 const Eigen::VectorXd&) {
    last_warmup = x;
  }
  void on_warmup_complete(double, const Eigen::VectorXd&) {}
  void on_logp_exception(const Eigen::VectorXd&,
                         const std::exception&) noexcept {
    ++errors;
  }
};

using Sampler = WalnutsSampler<Target, std::mt19937_64, Handler>;

Eigen::VectorXd zero() { return Eigen::VectorXd::Zero(2); }
Eigen::VectorXd ones() { return Eigen::VectorXd::Ones(2); }

// A fresh sampler built at the previous draw re-evaluates the endpoint in
// its constructor: an uncached baseline. The persistent sampler must give
// the same chain, with one evaluation saved per draw.
TEST(EndpointReuse, MatchesUncachedBaselineAndSavesOneEvaluationPerDraw) {
  for (unsigned seed : {17u, 20260910u}) {
    Target cached_target, baseline_target;
    Handler cached_handler, baseline_handler;
    std::mt19937_64 cached_rng(seed), baseline_rng(seed);
    Sampler cached(cached_rng, cached_handler, cached_target, zero(), ones(),
                   0.2, 4, 8, 1, 1.0);
    EXPECT_EQ(cached_target.calls, 1);  // evaluated once at construction

    Eigen::VectorXd position = zero();
    for (int i = 0; i < 50; ++i) {
      Sampler baseline(baseline_rng, baseline_handler, baseline_target,
                       position, ones(), 0.2, 4, 8, 1, 1.0);
      baseline();
      position = baseline_handler.samples.back().first;
    }

    for (int i = 0; i < 50; ++i) {
      cached();
    }
    EXPECT_EQ(cached_handler.samples, baseline_handler.samples);
    EXPECT_EQ(cached_rng, baseline_rng);
    EXPECT_EQ(baseline_target.calls - cached_target.calls, 49);
  }
}

// While the target fails away from the origin, each draw keeps the stored
// endpoint and costs only the failed trajectory evaluations (1 + 2 + 4 with
// one doubling and three halvings). After the target recovers, the chain
// moves on.
TEST(EndpointReuse, FailuresKeepStoredEndpointUntilRecovery) {
  Target target;
  target.fail_away_from_origin = true;
  Handler handler;
  std::mt19937_64 rng(17);
  Sampler sampler(rng, handler, target, zero(), ones(), 0.2, 1, 3, 1, 1.0);
  EXPECT_EQ(target.calls, 1);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(sampler(), 0);
    EXPECT_EQ(handler.samples.back().first.norm(), 0);
  }
  EXPECT_EQ(target.calls, 1 + 3 * 7);
  EXPECT_EQ(handler.errors, 3 * 7);

  target.fail_away_from_origin = false;
  for (int i = 0; i < 5; ++i) {
    sampler();
  }
  EXPECT_EQ(handler.errors, 3 * 7);
  EXPECT_GT(handler.samples.back().first.norm(), 0);
}

// The returned sampler evaluates its endpoint at the final warmup position,
// so freezing costs one evaluation.
TEST(EndpointReuse, FreezeEvaluatesOnceAtFinalWarmupPosition) {
  Target target;
  Handler handler;
  std::mt19937_64 rng(17);
  auto warmup_cfg = WarmupConfigBuilder().build();
  auto sampling_cfg = SamplingConfigBuilder().build();
  AdaptiveWalnuts<Target, std::mt19937_64, Handler> adaptive(
      rng, handler, target, InitChainConfig(0.2, zero(), ones()), warmup_cfg,
      sampling_cfg);
  EXPECT_EQ(target.calls, 1);
  for (int i = 0; i < 20; ++i) {
    adaptive();
  }
  std::size_t calls = target.calls;
  auto sampler = adaptive.sampler();
  EXPECT_EQ(target.calls, calls + 1);
  EXPECT_EQ(target.last_evaluated, handler.last_warmup);
}

}  // namespace
