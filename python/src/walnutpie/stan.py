import ctypes
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Union

import bridgestan
import numpy as np
import stanio

from ._ffi import (
    WALNUTPY_SEP,
    _ffi_sample_bridgestan,
    bs_print_callback_type,
    print_callback,
)
from .util import (
    WarmupInfo,
    prepare_inv_metric,
    prepare_output_buffer,
    prepare_seed,
    will_stop_adaptively,
)

StanData = Union[str, os.PathLike, Mapping[str, Any]]


class StanOutputBase:
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutputBase.get` method, or by using the object as a dictionary.
    """

    def __init__(self, parameters: List[str], data: np.ndarray):
        self.raw_parameters = parameters
        self._params = stanio.parse_header(",".join(parameters))
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """The underlying draws from the Stan model."""
        return self._data

    @property
    def parameters(self) -> List[str]:
        """The names of the parameters in the Stan model."""
        return list(self._params.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """Extract a parameter from the Stan output."""
        return self.get(key)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> np.ndarray:
        """
        Extract a parameter from the Stan output.
        Synonym for ``obj[key]``.

        Parameters
        ----------
        key : str
            The name of the parameter to extract.

        Returns
        -------
        np.ndarray
            The parameter values. Shape depends
            on the Stan type and algorithm used.
        """
        return self._params[key].extract_reshape(self._data)

    def __repr__(self) -> str:
        return f"StanOutput(parameters={repr(self.raw_parameters)}, data={repr(self.data)})"

    def __str__(self) -> str:
        p = "\n\t".join(self.parameters)
        return f"StanOutput with parameters:\n\t{p}"


class StanOutput(StanOutputBase):
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutput.get` method, or by using the object as a dictionary.


    Parameters
    ----------
    parameters : List[str]
        The names of the (constrained) model parameters, in output order.
    data : np.ndarray
        The raw sampler output.
    warmup : WarmupInfo[StanOutputBase]
        The warmup diagnostics and warmup draws for this chain.
    """

    def __init__(
        self,
        parameters: List[str],
        data: np.ndarray,
        warmup: WarmupInfo[StanOutputBase],
    ):
        super().__init__(parameters, data)

        #: The saved adaptation and warmup draws, if requested.
        self.warmup: WarmupInfo[StanOutputBase] = warmup

    def create_inits(
        self, *, chains: int = 4, seed: Optional[int] = None
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Create a dictionary of parameters suitable for initializing a new Stan run.

        Parameters
        ----------
        chains : int, optional
            The number of chains needed, by default 4.
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.

        Returns
        -------
        Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
            A dictionary of parameters, or a list of dictionaries if
            chains > 1.
        """
        if self._data.ndim == 1:
            return {
                name: var.extract_reshape(self._data)
                for name, var in self._params.items()
            }

        data = self._data.reshape((-1, self._data.shape[-1]))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(data.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = data[idxs[0]]
            return {
                name: var.extract_reshape(draw) for name, var in self._params.items()
            }
        return [
            {name: var.extract_reshape(data[idx]) for name, var in self._params.items()}
            for idx in idxs
        ]


def encode_stan_json(data: Union[str, os.PathLike, Mapping[str, Any]]) -> bytes:
    """Turn the provided data into something we can send to C++."""
    if isinstance(data, os.PathLike):
        return os.fspath(data).encode()
    if isinstance(data, str):
        return data.encode()
    return stanio.dump_stan_json(data).encode()


def _encode_stan_inits(inits, chains, seed) -> Union[bytes, None]:
    inits_encoded = None
    if inits is not None:
        if isinstance(inits, StanOutput):
            inits = inits.create_inits(chains=chains, seed=seed)

        if isinstance(inits, list):
            inits_encoded = WALNUTPY_SEP.join(encode_stan_json(init) for init in inits)
        else:
            inits_encoded = encode_stan_json(inits)
    return inits_encoded


@bs_print_callback_type
def bs_print_callback(msg, size, is_error):
    print(
        ctypes.string_at(msg, size).decode("utf-8"),
        file=sys.stderr if is_error else sys.stdout,
    )


def walnuts_stan(
    model: bridgestan.StanModel,
    *,
    num_chains: int = 4,
    inits: Union[StanData, List[StanData], None] = None,
    seed: Optional[int] = None,
    id: int = 1,
    init_radius: float = 2.0,
    init_inv_metric: Optional[np.ndarray] = None,
    save_inv_metric: bool = False,
    min_warmup_iter: int = 50,
    max_warmup_iter: int = 1000,
    min_sampling_iter: int = 50,
    max_sampling_iter: int = 1000,
    max_trajectory_doublings: int = 5,
    max_step_halvings: int = 5,
    min_micro_steps: int = 1,
    max_hamiltonian_error: float = 0.5,
    step_size_converge_tol: float = 0.1,
    mass_converge_tol: float = 1.0,
    rhat_converge_tol: float = 1.01,
    mass_init_count: float = 4.0,
    mass_additive_smoothing: float = 1e-5,
    max_macro_steps_target: float = 15.0,
    step_size_init: float = 1.0,
    step_accept_rate_target: float = 0.8,
    step_learning_rate: float = 0.05,
    step_gradient_decay: float = 0.8,
    step_sq_gradient_decay: float = 0.9,
    step_stabilization: float = 1e-4,
    step_learn_rate_decay: float = 0.5,
    save_warmup: bool = False,
    refresh: int = 0,
) -> list[StanOutput]:
    """
    Sample from the specified `Stan <https://mc-stan.org/>`__ model
    using the specified configuration.

    Parameters
    ----------
    model : bridgestan.StanModel
        The BridgeStan model to fit.
    num_chains : int, optional
        The number of Markov chains to run, positive, by default 4.
    inits : Union[StanData, List[StanData], None], optional
        The constrained initialization to use for all chains, or a list of constrained
        initializations, one for each chain, or ``None`` to indicate fully random initialization,
        by default None.
    seed : Optional[int], optional
        The pseudo-random number generator seed, non-negative, or ``None`` to use a random seed, by default ``None``.
    id : int, optional
        Numeric id for the first chain, by default 1. The remaining chains are given consecutive ids following this one.
        This controls the random number generation, along with the ``seed``.
    init_radius : float, optional
        The bounds of uniform random initialization (``-init_radius``, ``init_radius``), positive, by default ``2.0``.
    init_inv_metric : Optional[np.ndarray], optional
        The diagonal of the initial diagonal inverse metric, positive entries and size equal to transformed
        (unconstrained) dimension, or ``None``, in which case the mass matrix is initialized with a smoothed.
        negative outer product of gradients at the initial position, by default ``None``.
    save_inv_metric : bool, optional
        Set to ``True`` to save the inverse metric after adaptation, by default ``False``.
    min_warmup_iter : int, optional
        The minimum number of warmup iterations, greater than or equal to 0, by default ``50``.
    max_warmup_iter : int, optional
        The maximum number of warmup iterations, greater than or equal to ``min_warmup_iter``, by default ``1000``.
    min_sampling_iter : int, optional
        The minimum number of sampling iterations, greater than or equal to 0, by default ``50``.
    max_sampling_iter : int, optional
        The maximum number of sampling iterations, greater than or equal to ``min_sampling_iter``, by default ``1000``.
    max_trajectory_doublings : int, optional
        The maximum number of trajectory doublings for the no-U-turn sampler, positive, by default ``5``.
    max_step_halvings : int, optional
        The maximum number of step size halvings in Walnuts, non-negative, by default ``5``.
    min_micro_steps : int, optional
        The minimum number of micro steps per macro step, positive, by default ``1``.
    max_hamiltonian_error : float, optional
        The maximum error allowed in the Hamiltonian, positive, by default ``0.5``.
    step_size_converge_tol : float, optional
        The relative converge tolerance for difference in step sizes from the geometric mean across chains,
        positive, by default ``0.1``.
    mass_converge_tol : float, optional
        The relative mass matrix norm convergence tolerance from the geometric mean across chains, by default ``1.0``.
    rhat_converge_tol : float, optional
        The convergence tolernace for R-hat, greater than 1, by default ``1.01``.
    mass_init_count : float, optional
        The pseudo-observation count for the initial mass matrix, positive, by default ``4.0``.
    mass_additive_smoothing : float, optional
        The amount to add to the mass matrix estimators for smoothing, non-negative, by default ``1e-5``.
    max_macro_steps_target : float, optional
        The target maximum number of macro steps for adaptation, positive, by default ``15.0``.
    step_size_init : float, optional
        The initial step size, positive, by default ``1.0``.
    step_accept_rate_target : float, optional
        The acceptance rate target for step size adaptation, in (0, 1), by default ``0.8``.
    step_learning_rate : float, optional
        The learning rate for step size in Adam, positive, by default ``0.05``.
    step_gradient_decay : float, optional
        The step size gradient decay in Adam, positive, by default ``0.8``.
    step_sq_gradient_decay : float, optional
        The step size square gradient decay in Adam, positive, by default ``0.9``.
    step_stabilization : float, optional
        The additive step stabilization factor for Adam, non-negative, by default ``1e-4``.
    step_learn_rate_decay : float, optional
        The learning rate decay for Adam, non-negative, by default ``0.5``.
    save_warmup : bool, optional
        Set to ``True`` to save warmup iterations, by default ``False``.
    refresh : int, optional
        Period between iteration console feedback, with ``0`` indicating no feedback, non-netative, by default ``0``.

    Returns
    -------
    list[StanOutput]
        A list of Stan fits of length ``num_chains``, which may not all have the same number of draws.

    Raises
    ------
    ValueError
        If any argument is out of its valid range (documented above) or has inconsistent dimensionality.
    """

    if model.model_version() < (2, 9, 0):
        raise ValueError(
            "BridgeStan version must be at least 2.9.0 for use with walnuts"
        )
    if "STAN_THREADS=true" not in model.model_info():
        raise ValueError(
            "BridgeStan model must be compiled with STAN_THREADS for use with walnuts"
        )

    seed = prepare_seed(
        seed,
        will_stop_adaptively=will_stop_adaptively(
            num_chains=num_chains,
            min_warmup_iter=min_warmup_iter,
            max_warmup_iter=max_warmup_iter,
            min_sampling_iter=min_sampling_iter,
            max_sampling_iter=max_sampling_iter,
        ),
    )

    model_params = model.param_unc_num()
    param_names = model.param_names(include_tp=True, include_gq=True)

    num_params = len(param_names)

    out = prepare_output_buffer(
        num_chains=num_chains,
        num_params=num_params,
        max_sampling_iter=max_sampling_iter,
        max_warmup_iter=max_warmup_iter,
        save_warmup=save_warmup,
    )

    metric_size = (model_params,)
    init_inv_metric = prepare_inv_metric(init_inv_metric, metric_size, num_chains)

    lengths_out = np.zeros((num_chains * 2,), dtype=np.int32)
    stepsize_out = np.zeros(num_chains, dtype=np.float64)

    inv_metric_out = None
    if save_inv_metric:
        inv_metric_out = np.zeros((num_chains, *metric_size), dtype=np.float64)

    _ffi_sample_bridgestan(
        model.lib_path.encode(),
        model.data.encode(),
        bs_print_callback,
        model.seed,
        _encode_stan_inits(inits, num_chains, seed),
        num_chains,
        seed,
        id,
        init_radius,
        init_inv_metric,
        min_warmup_iter,
        max_warmup_iter,
        min_sampling_iter,
        max_sampling_iter,
        max_trajectory_doublings,
        max_step_halvings,
        min_micro_steps,
        max_hamiltonian_error,
        step_size_converge_tol,
        mass_converge_tol,
        rhat_converge_tol,
        mass_init_count,
        mass_additive_smoothing,
        max_macro_steps_target,
        step_size_init,
        step_accept_rate_target,
        step_learning_rate,
        step_gradient_decay,
        step_sq_gradient_decay,
        step_stabilization,
        step_learn_rate_decay,
        save_warmup,
        out,
        out.size,
        lengths_out,
        stepsize_out,
        inv_metric_out,
        refresh,
        print_callback,
    )

    outputs = []
    for i in range(num_chains):
        warmup_written = lengths_out[i]
        samples_written = lengths_out[i + num_chains]

        warmup_output = None
        if save_warmup:
            warmup_output = StanOutputBase(param_names, out[i, 0:warmup_written, :])

        warmup_info = WarmupInfo(
            stepsize_out[i],
            inv_metric_out[i] if inv_metric_out is not None else None,
            warmup_output,
        )

        out_chain = out[i, warmup_written : warmup_written + samples_written, :]
        output_chain = StanOutput(param_names, out_chain, warmup_info)

        outputs.append(output_chain)

    return outputs
