import ctypes
from typing import Any, Callable, Optional, Union

import numpy as np

from ._ffi import _ffi_sample_cfunc, logp_cfunc_type, print_callback
from .util import (
    WarmupInfo,
    prepare_inv_metric,
    prepare_output_buffer,
    prepare_seed,
    will_stop_adaptively,
)


class WalnutsOutputArray(np.ndarray):
    """
    An adapter for ``ndarray`` to set extra atrributes.

    See the numpy.org documentation: `Adding extra attributes to ndarray <https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray>`__.
    """

    #: The saved adaptation and warmup draws, if requested.
    warmup: WarmupInfo[np.ndarray]

    def __new__(cls, input_array, warmup: WarmupInfo[np.ndarray]):
        obj = np.asarray(input_array).view(cls)

        obj.warmup = warmup
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.warmup = getattr(obj, "warmup", None)


@logp_cfunc_type
def logp_c_trampoline(size, buf, grad, lp, logp_ptr):
    x = np.ctypeslib.as_array(buf, (size,))
    g = np.ctypeslib.as_array(grad, (size,))
    logp = ctypes.cast(logp_ptr, ctypes.POINTER(ctypes.py_object))
    try:
        lp[0], g[:] = logp.contents.value(x)
    except Exception as e:
        print(e)
        return 1
    return 0


def walnuts_pyfunc(
    logp: Union[
        Callable[[np.ndarray], tuple[float, np.ndarray]],
        "numba.core.ccallback.CFunc",
        tuple[ctypes.CFUNCTYPE, Any],
    ],
    *,
    num_params: Optional[int] = None,
    inits: Optional[np.ndarray] = None,
    num_chains: int = 4,
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
) -> list[WalnutsOutputArray]:
    """
    Sample from the specified model coded in Python using
    the specified configuration.

    Parameters
    ----------
    logp : Union[ Callable[[np.ndarray], tuple[float, np.ndarray]], numba.core.ccallback.CFunc, tuple[ctypes.CFUNCTYPE, Any], ]
        The target log density and gradient function. Acceptable formats include:
         - A function that accepts an :py:class:`numpy.ndarray` and returns a tuple of the log density and gradient.
         - A :py:func:`numba.cfunc` of type ``(size_t, doubleptr, doubleptr, doubleptr, voidptr)``
           which accepts the position as the second argument and sets the gradient in argument 3 and log density in argument 4.
         - A tuple of a ``ctypes`` callback of the same signature and an arbitrary data argument passed through the final pointer.
    num_params : Optional[int], optional
        The dimensionality of the problem, by default ``None``. At least one of ``num_params`` or ``inits`` must be specified.
    inits : Optional[np.ndarray], optional
        The constrained initialization to use for all chains, or a list of constrained initializations, one for each chain,
        or ``None`` to indicate fully random initialization, by default ``None``.
        At least one of ``num_params`` or ``inits`` must be specified.
    num_chains : int, optional
        The number of Markov chains to run, positive, by default ``4``.
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
    list[WalnutsOutputArray]
        A list of Markov chain of length ``num_chains``, which may not all have the same number of draws.

    Raises
    ------
    ValueError
        If any argument is out of its valid range (documented above) or has inconsistent dimensionality.
    """
    if num_params is None:
        if inits is None:
            raise ValueError("must specify at least one of num_params or inits")
        init_shape = inits.shape
        if len(init_shape) == 2:
            num_params = init_shape[1]
        else:
            num_params = init_shape[0]

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

    out = prepare_output_buffer(
        num_chains=num_chains,
        num_params=num_params,
        max_sampling_iter=max_sampling_iter,
        max_warmup_iter=max_warmup_iter,
        save_warmup=save_warmup,
    )

    if inits is not None:
        if inits.shape == (num_params,):
            inits = np.repeat(inits[np.newaxis], num_chains, axis=0)
        elif inits.shape == (num_chains, num_params):
            pass
        else:
            raise ValueError(
                f"Invalid inits size. Expected a {(num_params,)} "
                f"or {(num_chains, num_params)} matrix."
            )

    metric_size = (num_params,)
    init_inv_metric = prepare_inv_metric(init_inv_metric, metric_size, num_chains)

    lengths_out = np.zeros((num_chains * 2,), dtype=np.int32)
    stepsize_out = np.zeros(num_chains, dtype=np.float64)

    inv_metric_out = None
    if save_inv_metric:
        inv_metric_out = np.zeros((num_chains, num_params), dtype=np.float64)

    if hasattr(logp, "ctypes"):
        # numba's @cfunc decorator, which should generate very fast code
        logp_c = logp.ctypes
        logp_c_data = None
    elif isinstance(logp, tuple):
        logp_c = logp[0]
        logp_c_data = ctypes.byref(logp[1]) if logp[1] is not None else None
    else:
        # if we just have a generic python function, best we can do is wrap it
        # TODO: a faster path exist for JAX?
        #  - PJRT loading would rely on https://github.com/jax-ml/jax/issues/37033
        logp_c = logp_c_trampoline
        logp_c_data = ctypes.byref(ctypes.py_object(logp))

    _ffi_sample_cfunc(
        logp_c,
        logp_c_data,
        num_params,
        inits,
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

        warmup_info = WarmupInfo(
            stepsize=stepsize_out[i],
            inv_metric=inv_metric_out[i] if inv_metric_out is not None else None,
            warmup_draws=(out[i, 0:warmup_written, :] if save_warmup else None),
        )

        output_chain = WalnutsOutputArray(
            out[i, warmup_written : warmup_written + samples_written, :], warmup_info
        )
        outputs.append(output_chain)

    return outputs
