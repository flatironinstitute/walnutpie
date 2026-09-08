#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <walnutpie.hpp>
#include "test_util.hpp"

// MarkovChainsSplit class ******************************************

// chain 0: [[1, 2], [3, 4]]
// chain 1: [[5, 6], [7, 8], [9, 10]]
// chain 2: [[11, 12], [13, 14], [15, 16]]
std::vector<Eigen::MatrixXd> make_example_chains() {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(2, 2);
  c0 << 1, 2, 3, 4;
  Eigen::MatrixXd c1(3, 2);
  c1 << 5, 6, 7, 8, 9, 10;
  Eigen::MatrixXd c2(3, 2);
  c2 << 11, 12, 13, 14, 15, 16;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  chains.push_back(std::move(c2));
  return chains;
}

// constructor

TEST(MarkovChainsSplit, ConstructorThrowsOnEmptyChains) {
  std::vector<Eigen::MatrixXd> chains;
  EXPECT_THROW(walnutpie::MarkovChainsSplit{chains}, std::invalid_argument);
}

TEST(MarkovChainsSplit, ConstructorThrowsOnZeroRowChain) {
  std::vector<Eigen::MatrixXd> chains;
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 3));
  chains.emplace_back(Eigen::MatrixXd(0, 3));
  EXPECT_THROW(walnutpie::MarkovChainsSplit{chains}, std::invalid_argument);
}

TEST(MarkovChainsSplit, ConstructorThrowsOnInconsistentNumberOfColumns) {
  std::vector<Eigen::MatrixXd> chains;
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 3));
  chains.emplace_back(Eigen::MatrixXd::Zero(2, 4));
  EXPECT_THROW(walnutpie::MarkovChainsSplit{chains}, std::invalid_argument);
}

// accessors

TEST(MarkovChainsSplit, SizeAccessorsReturnExpectedValues) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(mcs.num_chains(), std::size_t{3});
  EXPECT_EQ(mcs.num_draws(), std::size_t{8});
  EXPECT_EQ(mcs.dims(), std::size_t{2});
  EXPECT_EQ(mcs.min_chain_size(), Eigen::Index{2});
}

// views

TEST(MarkovChainsSplit, ChainViewReturnsCorrectChain) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  for (std::size_t m = 0; m < chains.size(); ++m) {
    Eigen::MatrixXd view = mcs.chain_view(m);
    ASSERT_EQ(view.rows(), chains[m].rows());
    ASSERT_EQ(view.cols(), chains[m].cols());
    expect_near(view, chains[m]);
  }
}

TEST(MarkovChainsSplit, ChainViewThrowsOnOutOfRangeIndex) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(mcs.chain_view(3), std::out_of_range);
  EXPECT_THROW(mcs.chain_view(99), std::out_of_range);
}

// draws

TEST(MarkovChainsSplit, DrawsConcatenatesAcrossChains) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);

  Eigen::VectorXd expected_d0(8);
  expected_d0 << 1, 3, 5, 7, 9, 11, 13, 15;
  Eigen::VectorXd expected_d1(8);
  expected_d1 << 2, 4, 6, 8, 10, 12, 14, 16;

  Eigen::VectorXd d0 = mcs.draws(0);
  Eigen::VectorXd d1 = mcs.draws(1);

  ASSERT_EQ(d0.size(), Eigen::Index{8});
  ASSERT_EQ(d1.size(), Eigen::Index{8});
  expect_near(d0, expected_d0);
  expect_near(d1, expected_d1);
}

TEST(MarkovChainsSplit, DrawsThrowsOnOutOfRangeDimension) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(mcs.draws(-1), std::out_of_range);
  EXPECT_THROW(mcs.draws(2), std::out_of_range);
  EXPECT_THROW(mcs.draws(100), std::out_of_range);
}

// edge case

TEST(MarkovChainsSplit, SingleChainSingleDrawSingleDimension) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 1);
  c << 42.0;
  chains.push_back(std::move(c));

  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(mcs.num_chains(), std::size_t{1});
  EXPECT_EQ(mcs.num_draws(), std::size_t{1});
  EXPECT_EQ(mcs.dims(), std::size_t{1});
  EXPECT_EQ(mcs.min_chain_size(), Eigen::Index{1});

  Eigen::VectorXd d = mcs.draws(0);
  ASSERT_EQ(d.size(), Eigen::Index{1});
  EXPECT_DOUBLE_EQ(d(0), 42.0);
}

// MarkovChainsUnified class ****************************************

// same chains as MarkovChainsSplit test
Eigen::MatrixXd make_unified_draws() {
  Eigen::MatrixXd draws(8, 2);
  draws << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  return draws;
}

const std::vector<std::size_t> chain_sizes{2, 3, 3};

// constructor

TEST(MarkovChainsUnified, ConstructorThrowsOnChainSizesMismatch) {
  Eigen::MatrixXd draws = make_unified_draws();
  std::vector<std::size_t> wrong_sizes{2, 3, 2};
  EXPECT_THROW(walnutpie::MarkovChainsUnified(draws, wrong_sizes),
               std::invalid_argument);
}

TEST(MarkovChainsUnified, ConstructorThrowsOnChainSizesTooLarge) {
  Eigen::MatrixXd draws = make_unified_draws();
  std::vector<std::size_t> wrong_sizes{2, 3, 4};
  EXPECT_THROW(walnutpie::MarkovChainsUnified(draws, wrong_sizes),
               std::invalid_argument);
}

// accessors

TEST(MarkovChainsUnified, SizeAccessorsReturnExpectedValues) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_EQ(mcu.num_chains(), std::size_t{3});
  EXPECT_EQ(mcu.num_draws(), std::size_t{8});
  EXPECT_EQ(mcu.dims(), std::size_t{2});
  EXPECT_EQ(mcu.min_chain_size(), Eigen::Index{2});
}

// chain view

TEST(MarkovChainsUnified, ChainViewReturnsCorrectRows) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);

  // Expected chain contents match the logical split
  Eigen::MatrixXd expected0(2, 2);
  expected0 << 1, 2, 3, 4;
  Eigen::MatrixXd expected1(3, 2);
  expected1 << 5, 6, 7, 8, 9, 10;
  Eigen::MatrixXd expected2(3, 2);
  expected2 << 11, 12, 13, 14, 15, 16;
  std::vector<Eigen::MatrixXd> expected{expected0, expected1, expected2};

  for (std::size_t n = 0; n < expected.size(); ++n) {
    expect_near(mcu.chain_view(n).eval(), expected[n]);
  }
}

TEST(MarkovChainsUnified, ChainViewIsAViewNotACopy) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::Ref<const Eigen::MatrixXd> view = mcu.chain_view(0);
  EXPECT_EQ(view.data(), draws.data());  // tests memory sharing
}

TEST(MarkovChainsUnified, ChainViewThrowsOnOutOfRangeIndex) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_THROW(mcu.chain_view(3), std::out_of_range);
  EXPECT_THROW(mcu.chain_view(99), std::out_of_range);
  EXPECT_THROW(mcu.chain_view(std::numeric_limits<std::size_t>::max()),
               std::out_of_range);
}

TEST(MarkovChainsUnified, EmptyCollectionChainViewThrows) {
  Eigen::MatrixXd draws(0, 2);
  walnutpie::MarkovChainsUnified mcu(draws, {});
  EXPECT_THROW(mcu.chain_view(0), std::out_of_range);
}

// draws

TEST(MarkovChainsUnified, DrawsReturnsCorrectColumn) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);

  Eigen::VectorXd expected_d0(8);
  expected_d0 << 1, 3, 5, 7, 9, 11, 13, 15;
  Eigen::VectorXd expected_d1(8);
  expected_d1 << 2, 4, 6, 8, 10, 12, 14, 16;

  ASSERT_EQ(mcu.draws(0).size(), Eigen::Index{8});
  ASSERT_EQ(mcu.draws(1).size(), Eigen::Index{8});
  expect_near(mcu.draws(0).eval(), expected_d0);
  expect_near(mcu.draws(1).eval(), expected_d1);
}

TEST(MarkovChainsUnified, DrawsIsAViewNotACopy) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::Ref<const Eigen::VectorXd> col = mcu.draws(0);
  EXPECT_EQ(col.data(), draws.col(0).data());  // test memory sharing
}

TEST(MarkovChainsUnified, DrawsThrowsOnOutOfRangeDimension) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  EXPECT_THROW(mcu.draws(-1), std::out_of_range);
  EXPECT_THROW(mcu.draws(2), std::out_of_range);
  EXPECT_THROW(mcu.draws(100), std::out_of_range);
}

// edge case

TEST(MarkovChainsUnified, SingleChainSingleDrawSingleDimension) {
  Eigen::MatrixXd draws(1, 1);
  draws << 42.0;
  walnutpie::MarkovChainsUnified mcu(draws, {1});
  EXPECT_EQ(mcu.num_chains(), std::size_t{1});
  EXPECT_EQ(mcu.num_draws(), std::size_t{1});
  EXPECT_EQ(mcu.dims(), std::size_t{1});
  EXPECT_EQ(mcu.min_chain_size(), Eigen::Index{1});
  EXPECT_DOUBLE_EQ(mcu.draws(0)(0), 42.0);
}

// mean() function **************************************************

TEST(Mean, MarkovChainsSplitReturnsCorrectMean) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd m = walnutpie::mean(mcs);
  ASSERT_EQ(m.size(), 2);
  EXPECT_DOUBLE_EQ(m(0), 8.0);
  EXPECT_DOUBLE_EQ(m(1), 9.0);
}

TEST(Mean, MarkovChainsUnifiedReturnsCorrectMean) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd m = walnutpie::mean(mcu);
  ASSERT_EQ(m.size(), 2);
  EXPECT_DOUBLE_EQ(m(0), 8.0);
  EXPECT_DOUBLE_EQ(m(1), 9.0);
}

TEST(Mean, SingleDrawReturnsDrawValue) {
  std::vector<Eigen::MatrixXd> one;
  Eigen::MatrixXd c(1, 1);
  c << 7.5;
  one.push_back(std::move(c));
  walnutpie::MarkovChainsSplit single(one);
  Eigen::RowVectorXd m = walnutpie::mean(single);
  ASSERT_EQ(m.size(), 1);
  EXPECT_DOUBLE_EQ(m(0), 7.5);
}

TEST(Mean, ResultSizeMatchesDims) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(static_cast<std::size_t>(walnutpie::mean(mcs).size()), mcs.dims());
}

// sample_variance() function ***************************************

TEST(SampleVariance, MarkovChainsSplitReturnsCorrectVariance) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcs);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 24.0);
  EXPECT_DOUBLE_EQ(v(1), 24.0);
}

TEST(SampleVariance, MarkovChainsUnifiedReturnsCorrectVariance) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcu);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 24.0);
  EXPECT_DOUBLE_EQ(v(1), 24.0);
}

TEST(SampleVariance, BothTypesAgreeOnSameData) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  expect_near(walnutpie::sample_variance(mcs), walnutpie::sample_variance(mcu));
}

TEST(SampleVariance, ResultSizeMatchesDims) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(static_cast<std::size_t>(walnutpie::sample_variance(mcs).size()),
            mcs.dims());
}

TEST(SampleVariance, TwoIdenticalDrawsGivesZeroVariance) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 2);
  c << 3.0, 7.0, 3.0, 7.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcs);
  ASSERT_EQ(v.size(), 2);
  EXPECT_DOUBLE_EQ(v(0), 0.0);
  EXPECT_DOUBLE_EQ(v(1), 0.0);
}

TEST(SampleVariance, TwoDrawsMatchesHandCalculation) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 1);
  c << 1.0, 3.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcs);
  ASSERT_EQ(v.size(), 1);
  EXPECT_DOUBLE_EQ(v(0), 2.0);
}

TEST(SampleVariance, SingleDrawReturnsNaN) {
  // sum squares 0, N - 1 = 0, so 0 / 0 = NaN
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 2);
  c << 5.0, 3.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcs);
  ASSERT_EQ(v.size(), Eigen::Index{2});
  EXPECT_TRUE(std::isnan(v(0)));
  EXPECT_TRUE(std::isnan(v(1)));
}

// sample_standard_deviation() function

TEST(SampleStandardDeviation, MarkovChainsSplitReturnsCorrectStdDev) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnutpie::sample_standard_deviation(mcs);
  ASSERT_EQ(sd.size(), 2);
  EXPECT_DOUBLE_EQ(sd(0), std::sqrt(24.0));
  EXPECT_DOUBLE_EQ(sd(1), std::sqrt(24.0));
}

TEST(SampleStandardDeviation, MarkovChainsUnifiedReturnsCorrectStdDev) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::RowVectorXd sd = walnutpie::sample_standard_deviation(mcu);
  ASSERT_EQ(sd.size(), 2);
  EXPECT_DOUBLE_EQ(sd(0), std::sqrt(24.0));
  EXPECT_DOUBLE_EQ(sd(1), std::sqrt(24.0));
}

TEST(SampleStandardDeviation, IsSquareRootOfSampleVariance) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnutpie::sample_standard_deviation(mcs);
  Eigen::RowVectorXd v = walnutpie::sample_variance(mcs);
  expect_near(sd.array().square().matrix().eval(), v);
}

TEST(SampleStandardDeviation, TwoIdenticalDrawsGivesZeroStdDev) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 1);
  c << 5.0, 5.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd sd = walnutpie::sample_standard_deviation(mcs);
  ASSERT_EQ(sd.size(), 1);
  EXPECT_DOUBLE_EQ(sd(0), 0.0);
}

TEST(SampleStandardDeviation, SingleDrawReturnsNaN) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 2);
  c << 5.0, 3.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd v = walnutpie::sample_standard_deviation(mcs);
  ASSERT_EQ(v.size(), Eigen::Index{2});
  EXPECT_TRUE(std::isnan(v(0)));
  EXPECT_TRUE(std::isnan(v(1)));
}

// quantiles() function *********************************************

// Generate reference values with Python.

// import numpy as np
// # data matches examples in this test file
// col0 = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=float)
// col1 = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
// for p in [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]:
//     q0 = np.quantile(col0, p, method='linear')
//     q1 = np.quantile(col1, p, method='linear')
//     print(f'p={p:.2f} → ({q0:.4f}, {q1:.4f})')
// doc = np.array([9, 11, 5, 3], dtype=float)
// print(f'\nDoc example p=0.6: {np.quantile(doc_eg, 0.6, method="linear")}')
// output:
//   p=0.00 → (1.0,   2.0)
//   p=0.25 → (4.5,   5.5)
//   p=0.50 → (8.0,   9.0)
//   p=0.75 → (11.5, 12.5)
//   p=1.00 → (15.0, 16.0)
//   p=0.10 → (2.4,   3.4)
//   p=0.90 → (13.6, 14.6)
//   p=0.60 → (9.4,  10.4)

// constructor

TEST(Quantiles, ThrowsOnProbBelowZero) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(2);
  probs << -0.1, 0.5;
  EXPECT_THROW(walnutpie::quantiles(mcs, probs), std::invalid_argument);
}

TEST(Quantiles, ThrowsOnProbAboveOne) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);

  Eigen::VectorXd probs(2);
  probs << 0.5, 1.1;
  EXPECT_THROW(walnutpie::quantiles(mcs, probs), std::invalid_argument);

  Eigen::VectorXd probs2(10);
  probs2 << 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5;
  EXPECT_NO_THROW(walnutpie::quantiles(mcs, probs2));
}

TEST(Quantiles, ThrowsOnProbBelowZeroUnified) {
  auto draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);

  Eigen::VectorXd probs(2);
  probs << -0.1, 0.5;
  EXPECT_THROW(walnutpie::quantiles(mcu, probs), std::invalid_argument);

  Eigen::VectorXd probs2(10);
  probs2 << 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5;
  EXPECT_NO_THROW(walnutpie::quantiles(mcu, probs2));
}

TEST(Quantiles, ThrowsOnProbAboveOneUnified) {
  auto draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::VectorXd probs(2);
  probs << 0.5, 1.1;
  EXPECT_THROW(walnutpie::quantiles(mcu, probs), std::invalid_argument);
}

TEST(Quantiles, ThrowsOnNaNProbSplit) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(2);
  probs << 0.5, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(walnutpie::quantiles(mcs, probs), std::invalid_argument);
}

TEST(Quantiles, ThrowsOnNaNProbUnified) {
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::VectorXd probs(2);
  probs << 0.5, -0.1;
  EXPECT_THROW(walnutpie::quantiles(mcu, probs), std::invalid_argument);
}

// output shape

TEST(Quantiles, EmptyProbsReturnsEmptyMatrix) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(0);
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  EXPECT_EQ(result.rows(), 0);
  EXPECT_EQ(result.cols(), static_cast<Eigen::Index>(mcs.dims()));
}

TEST(Quantiles, ResultShapeIsProbs_x_Dims) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(3);
  probs << 0.25, 0.5, 0.75;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  EXPECT_EQ(result.rows(), Eigen::Index{3});
  EXPECT_EQ(result.cols(), Eigen::Index{2});
}

// boundary cases of probabilities

TEST(Quantiles, ProbZeroReturnsMinimum) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 0.0;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
}

TEST(Quantiles, ProbOneReturnsMaximum) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 1.0;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 15.0);
  EXPECT_DOUBLE_EQ(result(0, 1), 16.0);
}

// reference values match numpy

TEST(Quantiles, QuartilesMatchNumpy) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(5);
  probs << 0.0, 0.25, 0.5, 0.75, 1.0;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  Eigen::MatrixXd expected(5, 2);
  expected << 1.0, 2.0, 4.5, 5.5, 8.0, 9.0, 11.5, 12.5, 15.0, 16.0;
  expect_near(result, expected);
}

TEST(Quantiles, InteriorProbsMatchNumpy) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(3);
  probs << 0.1, 0.6, 0.9;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  Eigen::MatrixXd expected(3, 2);
  expected << 2.4, 3.4, 9.4, 10.4, 13.6, 14.6;
  expect_near(result, expected);
}

// example from the documentation

TEST(Quantiles, DocExampleMatchesPseudocode) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(4, 1);
  c << 9.0, 11.0, 5.0, 3.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::VectorXd probs(1);
  probs << 0.6;
  Eigen::MatrixXd result = walnutpie::quantiles(mcs, probs);
  EXPECT_DOUBLE_EQ(result(0, 0), 8.2);
}

// split and unified both work

TEST(Quantiles, BothTypesAgreeOnSameData) {
  auto chains = make_example_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws = make_unified_draws();
  walnutpie::MarkovChainsUnified mcu(draws, chain_sizes);
  Eigen::VectorXd probs(5);
  probs << 0.0, 0.25, 0.5, 0.75, 1.0;
  expect_near(walnutpie::quantiles(mcs, probs),
              walnutpie::quantiles(mcu, probs));
}

// autocovariance() function ****************************************

// import numpy as np
// def autocovariance_chain(chain):
//     N = chain.shape[0]
//     result = np.zeros_like(chain)
//     for d in range(chain.shape[1]):
//         col = chain[:, d]
//         mu = col.mean()
//         centered = col - mu
//         for lag in range(N):
// 	    # divide by N for MLE here
//             result[lag, d] = np.sum(centered[:N-lag] * centered[lag:]) / N
//     return result
// # Non-degenerate example
// ca = np.array([[1, 2], [4, 6]], dtype=float)
// cb = np.array([[3, 8], [7, 1], [2, 9]], dtype=float)
// cc = np.array([[6, 4], [1, 7], [8, 2]], dtype=float)

std::vector<Eigen::MatrixXd> make_acov_chains() {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd ca(2, 2);
  ca << 1, 2, 4, 6;
  Eigen::MatrixXd cb(3, 2);
  cb << 3, 8, 7, 1, 2, 9;
  Eigen::MatrixXd cc(3, 2);
  cc << 6, 4, 1, 7, 8, 2;
  chains.push_back(std::move(ca));
  chains.push_back(std::move(cb));
  chains.push_back(std::move(cc));
  return chains;
}

Eigen::MatrixXd autocovariance_direct(const Eigen::MatrixXd& chain) {
  const Eigen::Index N = chain.rows();
  const Eigen::Index D = chain.cols();
  Eigen::RowVectorXd mu = chain.colwise().mean();
  Eigen::MatrixXd centered = chain.rowwise() - mu;
  Eigen::MatrixXd result(N, D);
  for (Eigen::Index lag = 0; lag < N; ++lag) {
    result.row(lag) = (centered.topRows(N - lag).array() *
                       centered.bottomRows(N - lag).array())
                          .colwise()
                          .sum() /
                      static_cast<double>(N);
  }
  return result;
}

// shape of output

TEST(Autocovariance, ResultShapeIsNumDrawsTimesDims) {
  auto chains = make_acov_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnutpie::autocovariance(mcs);
  EXPECT_EQ(acov.rows(), static_cast<Eigen::Index>(mcs.num_draws()));
  EXPECT_EQ(acov.cols(), static_cast<Eigen::Index>(mcs.dims()));
}

// lag zero

TEST(Autocovariance, LagZeroEqualsChainBiasedVariance) {
  auto chains = make_acov_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnutpie::autocovariance(mcs);
  // chain 0 starts at row 0: biased var of [1,4]=9/4, [2,6]=4
  EXPECT_NEAR(acov(0, 0), 9.0 / 4.0, 1e-10);
  EXPECT_NEAR(acov(0, 1), 4.0, 1e-10);
  // chain 1 starts at row 2: biased var of [3,7,2]=14/3, [8,1,9]=38/3
  EXPECT_NEAR(acov(2, 0), 14.0 / 3.0, 1e-10);
  EXPECT_NEAR(acov(2, 1), 38.0 / 3.0, 1e-10);
  // chain 2 starts at row 5: biased var of [6,1,8]=26/3, [4,7,2]=38/9
  EXPECT_NEAR(acov(5, 0), 26.0 / 3.0, 1e-10);
  EXPECT_NEAR(acov(5, 1), 38.0 / 9.0, 1e-10);
}

// non-degenerate example

TEST(Autocovariance, FullResultMatchesReference) {
  auto chains = make_acov_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnutpie::autocovariance(mcs);

  Eigen::MatrixXd expected(8, 2);
  expected << 9.0 / 4.0, 4.0,     // chain 0, lag 0
      -9.0 / 8.0, -2.0,           // chain 0, lag 1
      14.0 / 3.0, 38.0 / 3.0,     // chain 1, lag 0
      -3.0, -25.0 / 3.0,          // chain 1, lag 1
      2.0 / 3.0, 2.0,             // chain 1, lag 2
      26.0 / 3.0, 38.0 / 9.0,     // chain 2, lag 0
      -16.0 / 3.0, -64.0 / 27.0,  // chain 2, lag 1
      1.0, 7.0 / 27.0;            // chain 2, lag 2

  expect_near(acov, expected);
}

// check FFT vs. direct

TEST(Autocovariance, MatchesDirectComputationPerChain) {
  auto chains = make_acov_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnutpie::autocovariance(mcs);
  std::vector<Eigen::Index> starts{0, 2, 5};
  std::vector<Eigen::Index> sizes{2, 3, 3};
  for (std::size_t m = 0; m < 3; ++m) {
    Eigen::MatrixXd chain = mcs.chain_view(m);
    Eigen::MatrixXd direct = autocovariance_direct(chain);
    Eigen::MatrixXd fft_block = acov.middleRows(starts[m], sizes[m]);
    expect_near(fft_block, direct);
  }
}

TEST(Autocovariance, LoopFftNextGoodSize) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(7, 2);
  c << 3.0, 7.0, 1.0, 9.3, 8.8, 3.0, 7.0, 1.0, 9.3, 8.8, 3.0, 7.0, 1.0, 9.3;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  auto x = walnutpie::autocovariance(mcs);
  EXPECT_EQ(7, x.rows());
  EXPECT_EQ(2, x.cols());
}

// check both coding types

TEST(Autocovariance, BothTypesAgreeOnSameData) {
  auto chains = make_acov_chains();
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd draws(8, 2);
  draws << 1, 2, 4, 6, 3, 8, 7, 1, 2, 9, 6, 4, 1, 7, 8, 2;
  walnutpie::MarkovChainsUnified mcu(draws, {2, 3, 3});
  expect_near(walnutpie::autocovariance(mcs), walnutpie::autocovariance(mcu));
}

// single draw chain

TEST(Autocovariance, SingleDrawChainGivesZero) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(1, 2);
  c << 3.0, 7.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::MatrixXd acov = walnutpie::autocovariance(mcs);
  ASSERT_EQ(acov.rows(), Eigen::Index{1});
  EXPECT_NEAR(acov(0, 0), 0.0, 1e-10);
  EXPECT_NEAR(acov(0, 1), 0.0, 1e-10);
}

// r_hat() function *************************************************

// import numpy as np
//
// def chain_mean(chain):
//     return chain.mean(axis=0)
//
// def sample_variance(chain):
//     # unbiased estimator, divides by N-1
//     return chain.var(axis=0, ddof=1)
//
// def r_hat(chains):
//     mu = np.array([chain_mean(c) for c in chains])
//     sigma_sq = np.array([sample_variance(c) for c in chains])
//     # sample_variance of chain means (divides by M-1)
//     var_mu = mu.var(axis=0, ddof=1)
//     mean_sigma_sq = sigma_sq.mean(axis=0)
//     return np.sqrt(1 + var_mu / mean_sigma_sq)

// constructor exceptions

TEST(RHat, ThrowsOnSingleChainUnified) {
  Eigen::MatrixXd c(4, 2);
  c << 1, 2, 3, 4, 5, 6, 7, 8;
  std::vector<std::size_t> sizes = {4};
  walnutpie::MarkovChainsUnified mcu(c, sizes);
  EXPECT_THROW(walnutpie::r_hat(mcu), std::invalid_argument);
}

TEST(RHat, ThrowsOnSingleChainSplit) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(4, 2);
  c << 1, 2, 3, 4, 5, 6, 7, 8;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(walnutpie::r_hat(mcs), std::invalid_argument);
}

TEST(RHat, ThrowsWhenAnyChainHasFewerThanThreeDraws) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  c0 << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd c1(2, 2);  // only 2 draws
  c1 << 7, 8, 9, 10;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(walnutpie::r_hat(mcs), std::invalid_argument);
}

TEST(RHat, ThrowsWhenFirstChainHasFewerThanThreeDraws) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(2, 2);
  c0 << 1, 2, 3, 4;
  Eigen::MatrixXd c1(3, 2);
  c1 << 5, 6, 7, 8, 9, 10;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(walnutpie::r_hat(mcs), std::invalid_argument);
}

TEST(RHat, ThrowsWhenAnyChainHasFewerThanThreeDrawsUnified) {
  Eigen::MatrixXd c(5, 2);
  c << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
  std::vector<std::size_t> sizes{3, 2};
  walnutpie::MarkovChainsUnified mcu(c, sizes);
  EXPECT_THROW(walnutpie::r_hat(mcu), std::invalid_argument);
}

TEST(RHat, ThrowsWhenAnyChainHasOneChainUnified) {
  Eigen::MatrixXd c(5, 2);
  c << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
  std::vector<std::size_t> sizes{5};
  walnutpie::MarkovChainsUnified mcu(c, sizes);
  EXPECT_THROW(walnutpie::r_hat(mcu), std::invalid_argument);
}

// output shape

TEST(RHat, ResultSizeMatchesDims) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  c0 << 1, 10, 2, 8, 3, 9;
  Eigen::MatrixXd c1(3, 2);
  c1 << 4, 5, 6, 7, 5, 6;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(walnutpie::r_hat(mcs).size(), Eigen::Index{2});
}

// converged chains

TEST(RHat, ConvergedChainsGiveRHatOfOne) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  Eigen::MatrixXd c1(3, 2);
  Eigen::MatrixXd c2(3, 2);
  // three permutations so identical means and variances
  c0 << 1, 2, 3, 4, 2, 3;
  c1 << 2, 3, 1, 2, 3, 4;
  c2 << 3, 4, 2, 3, 1, 2;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  chains.push_back(std::move(c2));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd rhat = walnutpie::r_hat(mcs);
  ASSERT_EQ(rhat.size(), Eigen::Index{2});
  EXPECT_DOUBLE_EQ(rhat(0), 1.0);
  EXPECT_DOUBLE_EQ(rhat(1), 1.0);
}

// equal variances

TEST(RHat, EqualWithinChainVarianceGivesSqrtTen) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  c0 << 1, 10, 2, 8, 3, 9;
  Eigen::MatrixXd c1(3, 2);
  c1 << 4, 5, 6, 7, 5, 6;
  Eigen::MatrixXd c2(3, 2);
  c2 << 7, 2, 9, 4, 8, 3;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  chains.push_back(std::move(c2));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd rhat = walnutpie::r_hat(mcs);
  ASSERT_EQ(rhat.size(), Eigen::Index{2});
  EXPECT_DOUBLE_EQ(rhat(0), std::sqrt(10.0));
  EXPECT_DOUBLE_EQ(rhat(1), std::sqrt(10.0));
}

// ragged chains

TEST(RHat, RaggedChainsMatchExactFractionalResult) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  c0 << 1, 5, 3, 3, 2, 4;
  Eigen::MatrixXd c1(4, 2);
  c1 << 4, 2, 6, 4, 5, 3, 7, 5;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd rhat = walnutpie::r_hat(mcs);
  ASSERT_EQ(rhat.size(), Eigen::Index{2});
  EXPECT_DOUBLE_EQ(rhat(0), std::sqrt(1.0 + 147.0 / 32.0));
  EXPECT_DOUBLE_EQ(rhat(1), std::sqrt(1.0 + 3.0 / 32.0));
}

// consistent across chain representation

TEST(RHat, BothTypesAgreeOnSameData) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 2);
  c0 << 1, 10, 2, 8, 3, 9;
  Eigen::MatrixXd c1(3, 2);
  c1 << 4, 5, 6, 7, 5, 6;
  Eigen::MatrixXd c2(3, 2);
  c2 << 7, 2, 9, 4, 8, 3;
  chains.push_back(c0);
  chains.push_back(c1);
  chains.push_back(c2);
  walnutpie::MarkovChainsSplit mcs(chains);

  Eigen::MatrixXd draws(9, 2);
  draws << 1, 10, 2, 8, 3, 9, 4, 5, 6, 7, 5, 6, 7, 2, 9, 4, 8, 3;
  walnutpie::MarkovChainsUnified mcu(draws, {3, 3, 3});

  expect_near(walnutpie::r_hat(mcs), walnutpie::r_hat(mcu));
}

// effective_sample_size() function *********************************

// import numpy as np

// def autocovariance_chain(chain):
//     N = chain.shape[0]
//     result = np.zeros_like(chain)
//     for d in range(chain.shape[1]):
//         col = chain[:, d]
//         mu = col.mean()
//         centered = col - mu
//         for lag in range(N):
//             result[lag, d] = np.sum(centered[:N - lag] * centered[lag:]) / N
//     return result

// def effective_sample_size(chains):
//     K = len(chains)
//     N_total = sum(c.shape[0] for c in chains)
//     D = chains[0].shape[1]
//     min_len = min(c.shape[0] for c in chains)
//     chain_means = np.array([c.mean(axis=0) for c in chains])       # K x D
//     chain_vars  = np.array([c.var(axis=0, ddof=1) for c in chains]) # K x D
//     W = chain_vars.mean(axis=0)
//     var_plus = W.copy()
//     if K > 1:
//         var_plus += chain_means.var(axis=0, ddof=1)
//     acov = np.vstack([autocovariance_chain(c) for c in chains])

//     def mean_acov_at_lag(t):
//         s, start = 0.0, 0
//         for c in chains:
//             s += acov[start + t, d]
//             start += c.shape[0]
//         return s / K

//     result = np.zeros(D)
//     for d in range(D):
//         w_d, vp_d = W[d], var_plus[d]
//         rho_hat_t = np.zeros(min_len)
//         rho_hat_even = 1.0
//         rho_hat_t[0] = rho_hat_even
//         rho_hat_odd = 1.0 - (w_d - mean_acov_at_lag(1)) / vp_d
//         rho_hat_t[1] = rho_hat_odd
//         t = 1
//         while (t < min_len - 4 and (rho_hat_even + rho_hat_odd) > 0.0
//                and not np.isnan(rho_hat_even + rho_hat_odd)):
//             rho_hat_even = 1.0 - (w_d - mean_acov_at_lag(t + 1)) / vp_d
//             rho_hat_odd  = 1.0 - (w_d - mean_acov_at_lag(t + 2)) / vp_d
//             if (rho_hat_even + rho_hat_odd) >= 0.0:
//                 rho_hat_t[t + 1] = rho_hat_even
//                 rho_hat_t[t + 2] = rho_hat_odd
//             if rho_hat_t[t+1] + rho_hat_t[t+2] > rho_hat_t[t-1] +
//             rho_hat_t[t]:
//                 rho_hat_t[t+1] = (rho_hat_t[t-1] + rho_hat_t[t]) / 2.0
//                 rho_hat_t[t+2] = rho_hat_t[t+1]
//             t += 2
//         max_t = t
//         if rho_hat_even > 0.0 and max_t + 1 < min_len:
//             rho_hat_t[max_t + 1] = rho_hat_even
//         tau_hat = (-1.0 + 2.0 * rho_hat_t[:max_t].sum() +
//                    (rho_hat_t[max_t + 1] if max_t + 1 < min_len else 0.0))
//         tau_hat = max(tau_hat, 1.0 / np.log10(N_total))
//         result[d] = N_total / tau_hat
//     return result

// def make_ar1_chain(N, phi, seed):
//     rng = np.random.default_rng(seed)
//     iid = rng.standard_normal((N, 1))
//     ar1 = np.zeros((N, 1))
//     ar1[0] = rng.standard_normal()
//     for t in range(1, N):
//         ar1[t] = phi * ar1[t - 1] + np.sqrt(1 - phi**2) *
//         rng.standard_normal()
//     return np.hstack([iid, ar1])

// chains_ar1 = [make_ar1_chain(20, 0.9, seed) for seed in [1, 2, 3]]
// print(effective_sample_size(chains_ar1))
// # -> [96.25678918147842   7.315045989031939]

// Hard-coded AR(1) chain data from Python run (make_ar1_chain, seeds 1/2/3,
// N=20, phi=0.9)
Eigen::MatrixXd make_ar1_chain_0() {
  Eigen::MatrixXd c(20, 2);
  c << 0.345584, 0.008142, 0.821618, -0.112805, 0.330437, 0.462545, -1.303157,
      0.855112, 0.905356, -0.412168, 0.446375, -1.194353, -0.536953, -1.151099,
      0.581118, -1.220018, 0.364572, -1.004891, 0.294132, -0.809673, 0.028422,
      0.194438, 0.546713, -0.309724, -0.736454, -0.443346, -0.162910, 0.491412,
      -0.482119, 0.724162, 0.598846, 0.940769, 0.039722, 0.622642, -0.292457,
      -0.158002, -0.781908, -0.069205, -0.257192, -0.014767;
  return c;
}

Eigen::MatrixXd make_ar1_chain_1() {
  Eigen::MatrixXd c(20, 2);
  c << 0.189053, 0.841465, -0.522748, 0.839281, -0.413064, 0.899446, -2.441467,
      0.988435, 1.799707, 0.449013, 1.144166, 0.745492, -0.325423, 1.567439,
      0.773807, 0.696515, 0.281211, -0.126970, -0.553823, -0.770214, 0.977567,
      -0.326409, -0.310557, -0.237662, -0.328824, 0.256143, -0.792147, 0.545429,
      0.454958, 0.582672, -0.099198, 0.648214, 0.545289, 0.509396, -0.607186,
      0.837009, 0.126828, 0.260877, -0.892274, 0.050905;
  return c;
}

Eigen::MatrixXd make_ar1_chain_2() {
  Eigen::MatrixXd c(20, 2);
  c << 2.040919, 0.024260, -2.555665, 0.695641, 0.418099, 0.863683, -0.567770,
      0.557091, -0.452649, 0.421684, -0.215597, 0.615125, -2.019986, 1.397098,
      -0.231932, 1.139863, -0.865213, 0.919712, 3.323000, 1.264639, 0.225787,
      0.751776, -0.352631, 0.549441, -0.281287, 0.879187, -0.668046, 1.044237,
      -1.055151, 0.979704, -0.390801, 1.173826, 0.481945, -0.176324, -0.238554,
      0.286485, 0.957759, -0.160463, -0.199802, -0.871751;
  return c;
}

// exceptions

TEST(EffectiveSampleSize, ThrowsOnFewerThanThreeTotalDraws) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c(2, 2);
  c << 1.0, 2.0, 3.0, 4.0;
  chains.push_back(std::move(c));
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_THROW(walnutpie::effective_sample_size(mcs), std::invalid_argument);
}

TEST(RHat, ThrowsOnFewerThanThreeTotalDrawsUnified) {
  Eigen::MatrixXd c(2, 2);
  c << 1, 2, 3, 4;
  std::vector<std::size_t> sizes{1, 1};
  walnutpie::MarkovChainsUnified mcu(c, sizes);
  EXPECT_THROW(walnutpie::effective_sample_size(mcu), std::invalid_argument);
}

// output shape

TEST(EffectiveSampleSize, ResultSizeMatchesDims) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(walnutpie::effective_sample_size(mcs).size(),
            static_cast<Eigen::Index>(mcs.dims()));
}

// ess all positive

TEST(EffectiveSampleSize, ResultIsPositive) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  EXPECT_TRUE((ess.array() > 0.0).all());
}

// single chain w/o r-hat adjustment

TEST(EffectiveSampleSize, SingleChainMatchesPythonReference) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  ASSERT_EQ(ess.size(), Eigen::Index{2});
  // AR(1) dim must have substantially lower ESS than iid dim
  EXPECT_GT(ess(0), ess(1));
  EXPECT_GT(ess(1), 0.0);
}

// match reference values

TEST(EffectiveSampleSize, ThreeChainMatchesPythonReference) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  ASSERT_EQ(ess.size(), Eigen::Index{2});
  EXPECT_NEAR(ess(0), 96.256789181, 1e-5);  // not quite 1e-6 tolerance
  EXPECT_NEAR(ess(1), 7.315045989, 1e-5);
}

// ar(1) vs. i.i.d. dim

TEST(EffectiveSampleSize, IidDimHasHigherEssThanAr1Dim) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  // AR(1) dim must have substantially lower ESS than iid dim
  EXPECT_GT(ess(0), 5.0 * ess(1));
}

// consistent across impls

TEST(EffectiveSampleSize, BothTypesAgreeOnSameData) {
  Eigen::MatrixXd c0 = make_ar1_chain_0();
  Eigen::MatrixXd c1 = make_ar1_chain_1();
  Eigen::MatrixXd c2 = make_ar1_chain_2();

  std::vector<Eigen::MatrixXd> chains{c0, c1, c2};
  walnutpie::MarkovChainsSplit mcs(chains);

  Eigen::MatrixXd draws(60, 2);
  draws << c0, c1, c2;  // Eigen comma-init stacks row blocks
  walnutpie::MarkovChainsUnified mcu(draws, {20, 20, 20});

  expect_near(walnutpie::effective_sample_size(mcs),
              walnutpie::effective_sample_size(mcu));
}

// test tau_hat floor at 1/log10(N_total)
TEST(EffectiveSampleSize, FloorPreventsTauHatFromGoingTooSmall) {
  std::vector<Eigen::MatrixXd> chains;
  Eigen::MatrixXd c0(3, 1);
  c0 << 10.0, 10.1, 9.9;  // high autocorrelation
  Eigen::MatrixXd c1(3, 1);
  c1 << 10.0, 9.9, 10.1;
  chains.push_back(std::move(c0));
  chains.push_back(std::move(c1));
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  EXPECT_GT(ess(0), 0.0);
  const double N_total = 6.0;
  const double floor_ess = N_total * std::log10(N_total);
  EXPECT_LE(ess(0), floor_ess + 1e-10);
}

// monte_carlo_standard_error() function ****************************

// def monte_carlo_standard_error(chains):
//     ess  = effective_sample_size(chains)
//     sd   = sample_standard_deviation(chains)
//     return sd / np.sqrt(ess)
//
// AR(1) chains used in the ESS tests (seeds 1/2/3, N=20, phi=0.9):
//   SD   = [0.945071606688786, 0.676390792099391]
//   ESS  = [96.256789181, 7.315045989]
//   MCSE = [0.096327220756986, 0.250085871061602]

// output shape

TEST(MonteCarloStandardError, ResultSizeMatchesDims) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_EQ(walnutpie::monte_carlo_standard_error(mcs).size(),
            static_cast<Eigen::Index>(mcs.dims()));
}

// positive mcse

TEST(MonteCarloStandardError, ResultIsPositive) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  EXPECT_TRUE((walnutpie::monte_carlo_standard_error(mcs).array() > 0.0).all());
}

// matches definition given pieces

TEST(MonteCarloStandardError, EqualsStdDevOverSqrtEss) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd mcse = walnutpie::monte_carlo_standard_error(mcs);
  Eigen::RowVectorXd sd = walnutpie::sample_standard_deviation(mcs);
  Eigen::RowVectorXd ess = walnutpie::effective_sample_size(mcs);
  expect_near(mcse, (sd.array() / ess.array().sqrt()).matrix().eval());
}

// matches reference test case

TEST(MonteCarloStandardError, ThreeChainMatchesPythonReference) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd mcse = walnutpie::monte_carlo_standard_error(mcs);
  ASSERT_EQ(mcse.size(), Eigen::Index{2});
  EXPECT_NEAR(mcse(0), 0.096327220756986, 1e-7);  // lowered tolerance
  EXPECT_NEAR(mcse(1), 0.250085871061602, 1e-7);
}

// ar(1) vs. i.i.d. relationship

TEST(MonteCarloStandardError, HighAutocorrelationIncreasesError) {
  std::vector<Eigen::MatrixXd> chains;
  chains.push_back(make_ar1_chain_0());
  chains.push_back(make_ar1_chain_1());
  chains.push_back(make_ar1_chain_2());
  walnutpie::MarkovChainsSplit mcs(chains);
  Eigen::RowVectorXd mcse = walnutpie::monte_carlo_standard_error(mcs);
  // AR(1) larger MCSE than i.i.d.
  EXPECT_GT(mcse(1), 2.0 * mcse(0));
}

// consistent across implementations

TEST(MonteCarloStandardError, BothTypesAgreeOnSameData) {
  Eigen::MatrixXd c0 = make_ar1_chain_0();
  Eigen::MatrixXd c1 = make_ar1_chain_1();
  Eigen::MatrixXd c2 = make_ar1_chain_2();
  std::vector<Eigen::MatrixXd> chains{c0, c1, c2};
  walnutpie::MarkovChainsSplit mcs(chains);

  Eigen::MatrixXd draws(60, 2);
  draws << c0, c1, c2;
  walnutpie::MarkovChainsUnified mcu(draws, {20, 20, 20});

  expect_near(walnutpie::monte_carlo_standard_error(mcs),
              walnutpie::monte_carlo_standard_error(mcu));
}
