from typing import TYPE_CHECKING, List, Union

import numpy as np

from ._ffi import _ffi_ess, _ffi_mcse, _ffi_r_hat

if TYPE_CHECKING:
    from .stan import StanOutputBase


class Summarizer:
    """
    A class to hold multivariate Markov chain Monte Carlo draws and provide
    summary statistics for their variables.
    """

    def __init__(self, draws: Union[List[np.ndarray], List["StanOutputBase"]]):
        """
        Construct an instance from a list of multivariate Markov chains.

        Parameters
        ----------
        draws
            A list of Markov chains represented as matrices with one row per draw or
            a list of Stan outputs.
        """

        if hasattr(draws[0], "parameters"):  # StanOutputBase
            draws = [c.data for c in draws]
        self._stacked = np.ascontiguousarray(np.concat(draws).transpose())
        self._num_params, self._num_draws = self._stacked.shape

        self._lengths = np.array([c.shape[0] for c in draws], dtype=np.int32)
        self._num_chains = len(draws)

    # Implement the simple functions directly in Python rather than with ffi       
    def mean(self):
        """
        Compute the arithmetic mean of sampled variables across all draws.

        Returns
        -------
        np.ndarray
            The posterior means.
        """
        return np.mean(self._stacked, axis=0)

    def variance(self):
        """
        Compute the sample variance (ddof = 1) of the sampled variables across
        all draws.

        Returns
        -------
        np.ndarray
            The posterior sample variances.
        """
        return np.var(self._stacked, axis=0, ddof=1)

    def standard_deviation(self):
        """
        Compute the sample standard deviation (ddof = 1) of the sampled
        variables across all draws.

        Returns
        -------
        np.ndarray
            The posterior sample standard deviations.
        """
        return np.std(self._stacked, axis=0, ddof=1)

    def ess(self) -> np.ndarray:
        """
        Return the estimated effective sample size of the sampled variables.

        The implementation uses initial monotonic sequence estimators for
        integrated autocorrelation. It also discounts ESS for non-convergence
        across chains.

        Returns
        -------
        np.ndarray
            The estimated effective sample sizes.
        """
        out = np.zeros((self._num_params,))
        _ffi_ess(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )

        return out

    def r_hat(self) -> np.ndarray:
        """
        Return the potential scale reduction statistic R-hat for the sampled
        variables.

        The definition of R-hat for ragged chains is conservative in
        that it (a) weighs each chain identically, not by chain length,
        and  (b) replaces the factor of (N - 1) / N when all chains are
        of length N with 1.

        Returns
        -------
        np.ndarray
            The R-hat statistics.
        """
        out = np.zeros((self._num_params,))
        _ffi_r_hat(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )
        return out

    def mcse(self) -> np.ndarray:
        """
        Return an estimate of the Monte Carlo standard error for the sampled
        variables.

        The MCSE is computed in the standard way as the estimated standard
        deviation divided by the square root of the estimated sample size.

        Returns
        -------
        np.ndarray
            The Monte Carlo standard error estimates.
        """
        out = np.zeros((self._num_params,))
        _ffi_mcse(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )
        return out


def ess(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    """
    Return the estimated effective sample size of the sampled variables.

    The implementation uses initial monotonic sequence estimators for
    integrated autocorrelation. It also discounts ESS for non-convergence
    across chains.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
        The estimated effective sample sizes.
    """
    return Summarizer(draws).ess()


def r_hat(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    """
    Return the potential scale reduction statistic R-hat for the sampled
    variables.

    The definition of R-hat for ragged chains is conservative in
    that it (a) weighs each chain identically, not by chain length,
    and  (b) replaces the factor of (N - 1) / N when all chains are
    of length N with 1.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
        The R-hat statistics.
    """
    return Summarizer(draws).r_hat()


def mcse(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    """
    Return an estimate of the Monte Carlo standard error for the sampled
    variables.

    The MCSE is computed in the standard way as the estimated standard
    deviation divided by the square root of the estimated sample size.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
        The Monte Carlo standard error estimates.
    """
    return Summarizer(draws).mcse()


def mean(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    """
    Compute the arithmetic mean of sampled variables across all draws.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
        The posterior means.
    """
    return Summarizer(draws).mean()


def variance(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    """
    Compute the sample variance (ddof = 1) of the sampled variables across
    all draws.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
       The posterior sample variances.
    """
    return Summarizer(draws).variance()


def standard_deviation(
    draws: Union[List[np.ndarray], List["StanOutputBase"]],
) -> np.ndarray:
    """
    Compute the sample standard deviation (ddof = 1) of the sampled
    variables across all draws.

    Parameters
    ----------
    draws : Union[List[np.ndarray], List[&quot;StanOutputBase&quot;]]
        A list of Markov chains represented as matrices with one row per draw or
        a list of Stan outputs.

    Returns
    -------
    np.ndarray
        The posterior sample standard deviations.
    """
    return Summarizer(draws).standard_deviation()
