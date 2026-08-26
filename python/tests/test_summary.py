import numpy as np
import walnutpie as wp


def test_chain_ordering():
    # test that there is no difference between calculating stats per-parameter versus all at once
    chains = [
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        np.array([[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]),
    ]

    rhat_together = wp.r_hat(chains)
    rhat_separately = np.array(
        [wp.r_hat([c[:, [j]] for c in chains])[0] for j in range(2)]
    )
    np.testing.assert_array_equal(rhat_separately, rhat_together)

    ess_together = wp.ess(chains)
    ess_separately = np.array(
        [wp.ess([c[:, [j]] for c in chains])[0] for j in range(2)]
    )
    np.testing.assert_array_equal(ess_separately, ess_together)


# remaining tests from from summary_test.cpp


def test_rhat_equal_chain_variance():
    chains = [
        np.array([[1.0, 10], [2, 8], [3, 9]]),
        np.array([[4.0, 5], [6, 7], [5, 6]]),
        np.array([[7.0, 2], [9, 4], [8, 3]]),
    ]
    np.testing.assert_array_almost_equal(
        wp.r_hat(chains), [np.sqrt(10.0), np.sqrt(10.0)]
    )


def test_rhat_ragged_chains():
    chains = [
        np.array([[1.0, 5], [3, 3], [2, 4]]),
        np.array([[4.0, 2], [6, 4], [5, 3], [7, 5]]),
    ]
    np.testing.assert_array_almost_equal(
        wp.r_hat(chains), [np.sqrt(1.0 + 147.0 / 32.0), np.sqrt(1.0 + 3.0 / 32.0)]
    )


def make_ar1_chain(N, phi, seed):
    rng = np.random.default_rng(seed)
    iid = rng.standard_normal((N, 1))
    ar1 = np.zeros((N, 1))
    ar1[0] = rng.standard_normal()
    for t in range(1, N):
        ar1[t] = phi * ar1[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    return np.hstack([iid, ar1])


chains_ar1 = [make_ar1_chain(20, 0.9, seed) for seed in [1, 2, 3]]


def test_ess_reference():
    print(chains_ar1)
    np.testing.assert_array_almost_equal(
        wp.ess(chains_ar1), [96.256789181, 7.315045989]
    )


def test_mcse_reference():
    print(chains_ar1)
    np.testing.assert_array_almost_equal(
        wp.mcse(chains_ar1), [0.096327220756986, 0.250085871061602]
    )
