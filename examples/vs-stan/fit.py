from types import SimpleNamespace
import time
import logging
import warnings
import pathlib
import numpy as np
import bridgestan
import walnutpie
import cmdstanpy
from cmdstanpy import CmdStanModel

warnings.simplefilter(action="ignore", category=FutureWarning)
cmdstanpy.utils.get_logger().setLevel(logging.ERROR)

SEED=598333
MIN_ITER=1000
ITER=1000
NUM_CHAINS=4

STAN_JSON_PAIRS = [
    # ("funnel/funnel.stan", "funnel/funnel.json"),
    ("multilevel_regression/multilevel_regression.stan", "multilevel_regression/multilevel_regression.json"),
    ("multilevel_regression/multilevel_regression_logit.stan", "multilevel_regression/multilevel_regression_logit.json"),
    ("measurement_error/measurement_error.stan", "measurement_error/measurement_error_1.json"),
    ### ("measurement_error/measurement_error.stan", "measurement_error/measurement_error_5.json"),
    ("hierarchical_matrix/hierarchical_matrix_1.stan", "hierarchical_matrix/hierarchical_matrix.json"),
    ("hierarchical_matrix/hierarchical_matrix_2.stan", "hierarchical_matrix/hierarchical_matrix.json")
    ]
   

class StanChain:
    def __init__(self, data, names, stepsize, cmdstan=None):
        self.data = data
        self.raw_parameters = names
        self.parameters = names
        self.warmup = SimpleNamespace(stepsize=stepsize, inv_metric=None,
                                      warmup_draws=None)
        self.cmdstan = cmdstan

    def __len__(self):
        return self.data.shape[0]

def flat_names(fit):
    out = []
    for n in fit[0].raw_parameters:
        head, _, idx = n.partition(".")
        out.append(head if not idx else f"{head}[{idx.replace('.', ',')}]")
    return out


def summarize(fit, names=None):
    s = walnutpie.Summarizer(fit)
    cols = dict(mean=s.mean(), sd=s.standard_deviation(), mcse=s.mcse(),
                ess=s.ess(), rhat=s.r_hat())
    k = len(cols["mean"])
    if names is None or len(names) != k:
        names = [f"v[{i}]" for i in range(k)]
    return names, cols, s


def print_summary(name, fit, names, cols, s, top=None):
    print(f"\n=== {name} ===")
    print(f"chains={len(fit)}  draws/chain={[np.asarray(c.data).shape[0] for c in fit]}"
          f"  total # draws={sum(len(c.data) for c in fit)}"
          f"  # parameters={s._stacked.shape[0]}")
    print(f"{'param':<24}{'mean':>12}{'sd':>12}{'mcse':>10}{'ess':>10}{'rhat':>8}")
    idx = range(len(names)) if top is None else range(min(top, len(names)))
    for i in idx:
        print(f"{names[i]:<24}{cols['mean'][i]:>12.2f}{cols['sd'][i]:>12.2f}"
              f"{cols['mcse'][i]:>10.3f}{cols['ess'][i]:>10.0f}{cols['rhat'][i]:>8.2f}")
    ok = np.isfinite(cols["rhat"]) & np.isfinite(cols["ess"]) & (cols["sd"] > 0)
    live = np.flatnonzero(ok)
    worst = int(live[np.argmax(cols["rhat"][live])])
    least = int(live[np.argmin(cols["ess"][live])])
    print(f"max rhat: {cols['rhat'][worst]:.3f} ({names[worst]});  "
          f"min ess: {cols['ess'][least]:.0f} ({names[least]})"
          f"{'' if ok.all() else f'  [{(~ok).sum()} constant/undefined columns excluded]'}")


def print_fit(fit, stan_file, pkg_name, wall=None):
    print(f"\n\n{pkg_name}:" + (f"  wall {wall:.2f}s" if wall else ""))    
    names, cols, s = summarize(fit, flat_names(fit))
    print_summary(pathlib.Path(stan_file).name, fit, names, cols, s, top=5)
    print("adapted step sizes:", [f"{c.warmup.stepsize:6.4f}" for c in fit])
    if getattr(fit[0], "cmdstan", None) is not None:
        print("diagnostics:", stan_diagnostics(fit))


def stan_diagnostics(fit):
    mv = fit[0].cmdstan.method_variables()
    td = mv["treedepth__"]
    return dict(divergences=int(mv["divergent__"].sum()),
                treedepth_median=float(np.median(td)),
                treedepth_max=float(td.max()),
                treedepth_at_max=float((td >= td.max()).mean()))


def fit_walnutpie_one(stan_file, data_file, seed=SEED):
    model = bridgestan.StanModel(str(stan_file),
                                 str(data_file),
                                 make_args=["STAN_THREADS=1"],
                                 capture_stan_prints=False,
                                 seed=seed)
    return walnutpie.walnuts_stan(
        model,
        num_chains=NUM_CHAINS,
        seed=seed,
        min_warmup_iter=MIN_ITER,
        max_warmup_iter=ITER,
        min_sampling_iter=MIN_ITER,
        max_sampling_iter=ITER,
        max_trajectory_doublings=10, # 5 Walnutpie
        max_step_halvings=5, # 5 Walnutpie
        max_macro_steps_target=16,  # 16.0 Walnutpie
          init_radius=0.1,  # 2.0 Walnutpie, 0.1 Good
          step_size_init=0.1,  # 1.0 Walnutpie, 0.1 Good
        max_hamiltonian_error=1,  # 0.5 Walnutpie, 0.5 Good, infty Nuts
        mass_init_count=4,  # 4.0 Walnutpie, 4.0 Good
        rhat_converge_tol=1.01,  # 1.01 Walnutpie
        step_accept_rate_target=0.95,  # 0.8 Walnutpie, 0.8 Good
        step_learning_rate=0.05,  # 0.001 Adam default, 0.05 Walnutpie, 0.05 Good
        step_gradient_decay=0.8,  # 0.9 Adam, 0.8 Walnutpie, 0.9 Good
        step_sq_gradient_decay=0.9,  # 0.999 Adam, 0.9 Walnutpie, 0.999 Good
        step_stabilization=0.0001,  # 1e-7 Adam, 0.0001 Walnutpie
        step_learn_rate_decay=0.5,  # 0.5 Walnutpie
        # init_inv_metric=np.ones(model.param_unc_num()),
        save_inv_metric=False,
        refresh=0)


def fit_stan_one(stan_file, data_file, seed=SEED, num_chains=NUM_CHAINS,
                 iter_warmup=MIN_ITER, iter_sampling=ITER, **sample_kwargs):
    model = CmdStanModel(stan_file=str(stan_file))
    csp_fit = model.sample(data=str(data_file), chains=num_chains,
                           parallel_chains=num_chains, iter_warmup=iter_warmup,
                           iter_sampling=iter_sampling, seed=seed,
                           inits=0.1,
                           show_progress=False, show_console=False, **sample_kwargs)
    keep = [i for i, n in enumerate(csp_fit.column_names) if not n.endswith("__")]
    names = [csp_fit.column_names[i] for i in keep]
    draws = csp_fit.draws(concat_chains=False)
    step = np.atleast_1d(np.asarray(getattr(csp_fit, "step_size", np.nan), dtype=float))
    if step.size < draws.shape[1]:
        step = np.full(draws.shape[1], np.nan)
    return [StanChain(np.ascontiguousarray(draws[:, c, keep]), names,
                      float(step[c]), csp_fit)
            for c in range(draws.shape[1])]


if __name__ == "__main__":
    for stan_file, json_file in STAN_JSON_PAIRS:
        print(f"\n\n***** {stan_file=} {json_file=} *****")
        stan_path = pathlib.Path(stan_file)
        json_path = pathlib.Path(json_file)

        t0 = time.perf_counter()
        fit_wnp = fit_walnutpie_one(stan_path, json_path)
        print_fit(fit_wnp, stan_path, "Walnutpie", time.perf_counter() - t0)

        t0 = time.perf_counter()
        fit_csp = fit_stan_one(stan_path, json_path)
        print_fit(fit_csp, stan_path, "CmdStanPy", time.perf_counter() - t0)
    print("")
