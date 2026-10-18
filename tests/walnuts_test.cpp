#include <gtest/gtest.h>
#include <walnutpie/adam.hpp>
#include <walnutpie/adapt.hpp>
#include <walnutpie/config.hpp>
#include <walnutpie/walnuts.hpp>

#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <random>
#include <vector>

namespace {
using walnutpie::detail::Adam;
using walnutpie::detail::Direction;
using walnutpie::detail::SpanW;

void gaussian(const Eigen::VectorXd& x, double& lp, Eigen::VectorXd& g) {
  lp = -0.5 * x.squaredNorm();
  g = -x;
}

struct RecordingAdapter {
  std::vector<double> observed;
  void operator()(double alpha) { observed.push_back(alpha); }
  double step_size() const { return 1.0; }
};

template <Direction D, typename F, typename A>
bool take_macro_step(const F& logp_grad, A& adapter,
                     std::size_t max_halvings = 3) {
  // The initial state has finite log density and gradient. Invalid values
  // below arise only when evaluating a proposed point, not at initialization.
  walnutpie::detail::SpanWorkspace workspace(1, 0);
  SpanW& span = workspace.span(0, 0);
  span.init_from(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1),
                 Eigen::VectorXd::Zero(1), 0.0, -0.5);
  Eigen::VectorXd theta(1), rho(1), grad(1);
  double logp_pos, logp;
  return walnutpie::detail::macro_step<D>(
      logp_grad, Eigen::VectorXd::Ones(1), 1.0, max_halvings, 1, 1.0, span,
      theta, rho, grad, logp_pos, logp, workspace, adapter);
}

template <typename F>
std::size_t transition_depth(const F& target, std::size_t max_depth,
                             double step = 0.05) {
  std::mt19937 rng(7);
  walnutpie::detail::Random<std::mt19937> rand(rng);
  walnutpie::detail::NoOpStepSizeAdapter adapter;
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd grad(2);
  double logp;
  target(theta, logp, grad);
  std::size_t depth;
  walnutpie::detail::SpanWorkspace workspace(2, max_depth);
  walnutpie::detail::transition_w(rand, target, Eigen::VectorXd::Ones(2),
                                  Eigen::VectorXd::Ones(2), step, max_depth, 5,
                                  1, 1.0, std::move(theta), depth, grad, logp,
                                  workspace, adapter);
  return depth;
}

TEST(MacroStepAdaptation, NonfiniteLogDensityIsZeroAcceptanceInBothDirections) {
  for (double logp : {std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()}) {
    auto target = [logp](const Eigen::VectorXd& theta, double& lp,
                         Eigen::VectorXd& grad) {
      lp = theta.isZero(0.0) ? 0.0 : logp;
      grad.setZero();
    };
    RecordingAdapter forward, backward;
    EXPECT_FALSE(take_macro_step<Direction::Forward>(target, forward));
    EXPECT_FALSE(take_macro_step<Direction::Backward>(target, backward));
    EXPECT_EQ(forward.observed, std::vector<double>{0.0});
    EXPECT_EQ(backward.observed, std::vector<double>{0.0});
  }
}

TEST(MacroStepAdaptation, NonfiniteGradientIsZeroAcceptance) {
  auto target = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& grad) {
    lp = 0.0;
    grad.setConstant(std::numeric_limits<double>::quiet_NaN());
  };
  RecordingAdapter adapter;
  EXPECT_FALSE(take_macro_step<Direction::Forward>(target, adapter));
  EXPECT_EQ(adapter.observed, std::vector<double>{0.0});
}

TEST(MacroStepAdaptation, AdamMatchesRejectionAndRemainsUsable) {
  auto target = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& grad) {
    lp = std::numeric_limits<double>::quiet_NaN();
    grad.setZero();
  };
  Adam actual(1.0, 0.8, 0.05, 0.9, 0.999, 1e-8, 0.5);
  Adam expected = actual;
  EXPECT_FALSE(take_macro_step<Direction::Forward>(target, actual));
  expected(0.0);
  EXPECT_DOUBLE_EQ(actual.step_size(), expected.step_size());
  EXPECT_LT(actual.step_size(), 1.0);
  actual(0.9);
  expected(0.9);
  EXPECT_DOUBLE_EQ(actual.step_size(), expected.step_size());
  EXPECT_TRUE(std::isfinite(actual.step_size()));
}

TEST(MacroStepAdaptation, FiniteEnergyErrorKeepsAcceptanceStatistic) {
  // Zero gradient and unit momentum keep kinetic energy constant.
  auto target = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& grad) {
    lp = -2.0;
    grad.setZero();
  };
  RecordingAdapter adapter;
  EXPECT_FALSE(take_macro_step<Direction::Forward>(target, adapter));
  ASSERT_EQ(adapter.observed.size(), 1);
  EXPECT_DOUBLE_EQ(adapter.observed[0], std::exp(-2.0));
}

TEST(MacroStepAdaptation, ConservedEnergyKeepsUnitAcceptance) {
  auto target = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& grad) {
    lp = 0.0;
    grad.setZero();
  };
  RecordingAdapter adapter;
  EXPECT_TRUE(take_macro_step<Direction::Forward>(target, adapter));
  EXPECT_EQ(adapter.observed, std::vector<double>{1.0});
}

struct StopPolling {
  void throw_if_interrupted() const {
    throw std::runtime_error("not converged");
  }
};
  
TEST(ControllerLoop, RejectsLowStepWhenHighStepsAreWithinTolerance) {
  StopPolling interrupt;
  auto init = walnutpie::InitConfigBuilder(3, 1).build();
  auto warmup_cfg = walnutpie::WarmupConfigBuilder()
                        .step_size_converge_tol(0.1)
                        .build();
  std::deque<walnutpie::detail::SpscBuffer<walnutpie::detail::AdaptSnapshot>>
      buffers;
  walnutpie::detail::AdaptSnapshot snap(1);
  snap.iter = warmup_cfg.min_iter();
  snap.log_mass.setZero();
  snap.mass.setOnes();
  for (double step : {0.8, 1.05, 1.05}) {
    snap.log_step = std::log(step);
    buffers.emplace_back(snap);
  }
  EXPECT_THROW(
      walnutpie::detail::controller_loop(buffers, interrupt, init, warmup_cfg),
      std::runtime_error);
}

TEST(ControllerLoop, AcceptWhenStepsConverge) {
  StopPolling interrupt;
  auto init = walnutpie::InitConfigBuilder(3, 1).build();
  auto warmup_cfg = walnutpie::WarmupConfigBuilder()
                        .step_size_converge_tol(0.1)
                        .build();
  std::deque<walnutpie::detail::SpscBuffer<walnutpie::detail::AdaptSnapshot>>
      buffers;
  walnutpie::detail::AdaptSnapshot snap(1);
  snap.iter = warmup_cfg.min_iter();
  snap.log_mass.setZero();
  snap.mass.setOnes();
  for (double step : {1.0, 1.0, 1.0}) {
    snap.log_step = std::log(step);
    buffers.emplace_back(snap);
  }
  EXPECT_NO_THROW(walnutpie::detail::controller_loop(buffers, interrupt, init, warmup_cfg));
}
  
TEST(TransitionDepth, ReportsIncludedDoublings) {
  for (std::size_t max_depth : {std::size_t{1}, std::size_t{3}}) {
    EXPECT_EQ(transition_depth(gaussian, max_depth), max_depth);
  }
  auto nonfinite = [](const Eigen::VectorXd&, double& lp, Eigen::VectorXd& g) {
    lp = -std::numeric_limits<double>::infinity();
    g.setZero();
  };
  EXPECT_EQ(transition_depth(nonfinite, 4), 0u);
}

TEST(SpanWorkspace, ReusedWorkspaceMatchesFreshWorkspace) {
  auto run_chain = [](bool reuse_workspace) {
    std::mt19937 rng(7);
    walnutpie::detail::Random<std::mt19937> rand(rng);
    walnutpie::detail::NoOpStepSizeAdapter adapter;
    walnutpie::detail::SpanWorkspace workspace(2, 4);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd grad(2);
    double logp;
    gaussian(theta, logp, grad);
    std::vector<Eigen::VectorXd> thetas;
    std::vector<std::size_t> depths;
    for (int i = 0; i < 25; ++i) {
      walnutpie::detail::SpanWorkspace fresh_workspace(2, 4);
      std::size_t depth;
      theta = walnutpie::detail::transition_w(
          rand, gaussian, Eigen::VectorXd::Ones(2), Eigen::VectorXd::Ones(2),
          0.3, 4, 5, 2, 1.0, std::move(theta), depth, grad, logp,
          reuse_workspace ? workspace : fresh_workspace, adapter);
      thetas.push_back(theta);
      depths.push_back(depth);
    }
    return std::pair(thetas, depths);
  };
  EXPECT_EQ(run_chain(true), run_chain(false));
}

TEST(SpanWorkspace, CombiningSpansLeavesEveryVectorSized) {
  walnutpie::detail::SpanWorkspace workspace(3, 1);
  SpanW& span1 = workspace.span(0, 0);
  SpanW& span2 = workspace.span(0, 1);
  SpanW& combined = workspace.accumulator(0);
  span1.init_from(Eigen::VectorXd::Constant(3, 1.0),
                  Eigen::VectorXd::Constant(3, 2.0),
                  Eigen::VectorXd::Constant(3, 3.0), -1.0, -1.0);
  span2.init_from(Eigen::VectorXd::Constant(3, 4.0),
                  Eigen::VectorXd::Constant(3, 5.0),
                  Eigen::VectorXd::Constant(3, 6.0), -2.0, -2.0);
  combined.init_from_subspans(span1, span2, span2, -2.0, -0.5);
  EXPECT_TRUE(combined.theta_bk_.isConstant(1.0));
  EXPECT_TRUE(combined.theta_fw_.isConstant(4.0));
  EXPECT_TRUE(combined.theta_select_.isConstant(4.0));
  for (const SpanW* span : {&span1, &span2, &combined}) {
    for (const Eigen::VectorXd* v :
         {&span->theta_bk_, &span->rho_bk_, &span->grad_theta_bk_,
          &span->theta_fw_, &span->rho_fw_, &span->grad_theta_fw_,
          &span->theta_select_, &span->grad_select_}) {
      EXPECT_EQ(v->size(), 3);
    }
  }
}

}  // namespace
