import numpy as np
import walnutpie as wp


def test_chain_ordering():

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
