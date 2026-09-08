#include <gtest/gtest.h>
#include <walnutpie/adam.hpp>
#include <walnutpie/walnuts.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace {
using walnutpie::detail::Adam;
using walnutpie::detail::Direction;
using walnutpie::detail::SpanW;

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
  auto span = SpanW::from_initial_point(Eigen::VectorXd::Zero(1),
                                        Eigen::VectorXd::Ones(1),
                                        Eigen::VectorXd::Zero(1), 0.0, -0.5);
  Eigen::VectorXd theta(1), rho(1), grad(1);
  double logp_pos, logp;
  return walnutpie::detail::macro_step<D>(
      logp_grad, Eigen::VectorXd::Ones(1), 1.0, max_halvings, 1, 1.0, span,
      theta, rho, grad, logp_pos, logp, adapter);
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
}  // namespace
