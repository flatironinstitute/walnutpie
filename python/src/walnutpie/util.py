import warnings
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np


def will_stop_adaptively(
    *,
    num_chains: int,
    min_warmup_iter: int,
    max_warmup_iter: int,
    min_sampling_iter: int,
    max_sampling_iter: int,
) -> bool:
    return num_chains > 1 and (
        min_warmup_iter != max_warmup_iter or min_sampling_iter != max_sampling_iter
    )


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)


def prepare_seed(seed: Optional[int], will_stop_adaptively: bool) -> int:
    if seed is not None:
        if will_stop_adaptively:
            warnings.warn(
                "Setting 'seed' without also disabling adaptive stopping "
                "(by setting min and max number of iterations for warmup and sampling) "
                "will not lead to reproducible sampling due to thread scheduling!",
                UserWarning,
                stacklevel=3,
            )
        return seed
    return rand_u32()


def prepare_output_buffer(
    *,
    num_chains: int,
    num_params: int,
    max_sampling_iter: int,
    max_warmup_iter: int,
    save_warmup: bool,
) -> np.ndarray:
    if num_chains < 1:
        raise ValueError("num_chains must be at least 1")
    if max_warmup_iter < 0:
        raise ValueError("max_warmup_iter must be non-negative")
    if max_sampling_iter < 1:
        raise ValueError("max_sampling_iter must be at least 1")

    num_draws = max_sampling_iter + max_warmup_iter * save_warmup
    return np.zeros((num_chains, num_draws, num_params), dtype=np.float64)


def prepare_inv_metric(
    init_inv_metric: Optional[np.ndarray], metric_size: tuple[int, ...], num_chains: int
) -> Optional[np.ndarray]:
    if init_inv_metric is not None:
        if init_inv_metric.shape == metric_size:
            return np.repeat(init_inv_metric[np.newaxis], num_chains, axis=0)
        elif init_inv_metric.shape == (num_chains, *metric_size):
            return init_inv_metric
        else:
            raise ValueError(
                f"Invalid initial metric size. Expected a {metric_size} "
                f"or {(num_chains, *metric_size)} matrix."
            )


T = TypeVar("T")


@dataclass
class WarmupInfo(Generic[T]):
    """
    Warmup output from a single chain, parameterized by the array type.

    Attributes
    ----------
    stepsize : float
        The adapted step size from warmup.
    inv_metric : Optional[np.ndarray]
        The diagonal inverse mass matrix estimated during warmup, or ```None`` if not saved.
    warmup_draws : Optional[T]
        The warmup draws, or ``None`` if not saved.
    """

    stepsize: float
    inv_metric: Optional[np.ndarray]
    warmup_draws: Optional[T]
