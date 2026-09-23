from types import SimpleNamespace
import json
import time
import logging
import warnings
import pathlib
import numpy as np
import bridgestan
import walnutpie
import cmdstanpy
import nutpie
from cmdstanpy import CmdStanModel

warnings.simplefilter(action="ignore", category=FutureWarning)
cmdstanpy.utils.get_logger().setLevel(logging.ERROR)

SEED = 111111
ITER_WARMUP = 1000
MIN_ITER_WARMUP = ITER_WARMUP
ITER = 1000
MIN_ITER = ITER
NUM_CHAINS = 4
METRIC_PER_LINE = 100
NUTPIE_TARGET_ACCEPT = 0.8
NUTPIE_ADAPTATION = "diag"

STAN_JSON_PAIRS = [
   ("wells/wells_daae_c_model.stan", "wells/wells_data.json"),
   ("wells/wells_interaction_c_model.stan", "wells/wells_data.json"),
   ("wells/wells_dist.stan", "wells/wells_data.json"),

   ("seeds/seeds_model.stan", "seeds/seeds_data.json"),
   ("seeds/seeds_centered_model.stan", "seeds/seeds_data.json"),
   ("seeds/seeds_stanified_model.stan", "seeds/seeds_data.json"),

   ("rats/rats_model.stan", "rats/rats_data.json"),   # not fitting well in Walnutpie with small warmup, warmup min/max off

   ("prophet/prophet.stan", "prophet/rstan_downloads.json"),

   ("occupancy/multi_occupancy.stan", "occupancy/butterfly.json"),

   ("capture/M0_model.stan", "capture/M0_data.json"),
   ("capture/Mtbh_model.stan", "capture/Mtbh_data.json"),

   ("lsat/lsat_model.stan", "lsat/lsat_data.json"),

   ("losscurve/losscurve_sislob.stan", "losscurve/loss_curves.json"),

   ("election/election88_full.stan", "election/election88.json"),

   ("spatial/bym2_offset_only.stan", "spatial/traffic_accident_nyc.json"),  # nutpie much better

    ("mnist/nn_rbm1bJ10.stan", "mnist/mnist_100.json"),  # can do if crank down iterations or wait

    ("diamonds/diamonds.stan", "diamonds/diamonds.json"),   # 2.4 MB data, slow

    ("irt/irt_2pl.stan", "irt/irt_2pl.json"),

    ("earn/earn_height.stan", "earn/earnings.json"),     # *** doesn't fit in Walnutpie ***
    ("earn/logearn_height.stan", "earn/earnings.json"),

    ("kilpisjarvi/kilpisjarvi.stan", "kilpisjarvi/kilpisjarvi_mod.json"),  # *** doesn't fit in Walnutpie ***

    ("nes/nes.stan", "nes/nes1980.json"),
    ("nes/nes.stan", "nes/nes2000.json"),

    ("mesquite/logmesquite_logvolume.stan", "mesquite/mesquite.json"),

    ("mixture/low_dim_gauss_mix.stan", "mixture/low_dim_gauss_mix.json"),

    ("kid/kidscore_interaction.stan", "kid/kidiq.json"),

    ("blr/blr.stan", "blr/sblrc.json"),  # requires > 400 warmup iterations for Walnutpie, but not Nutpie

    ("radon/radon_pooled.stan", "radon/radon_mn.json"),
    ("radon/radon_pooled.stan", "radon/radon_all.json"),

    ("ode/lotka_volterra.stan", "ode/hudson_lynx_hare.json"),

    ("hmm/hmm_example.stan", "hmm/hmm_example.json"),
    ("hmm/hmm_gaussian.stan", "hmm/hmm_gaussian_simulated.json"),

    ("gp/accel_gp.stan", "gp/mcycle_gp.json"),  # hard to fit
    ("gp/gp_pois_regr.stan", "gp/gp_pois_regr.json"),

    ("eight-schools/eight_schools_centered.stan", "eight-schools/eight_schools.json"),
    ("eight-schools/eight_schools_noncentered.stan", "eight-schools/eight_schools.json"),

    ("time-series/arK.stan", "time-series/arK.json"),
    ("time-series/arma11.stan", "time-series/arma.json"),
    ("time-series/garch11.stan", "time-series/garch.json"),

    ("normal/std-normal.stan", "normal/std-normal.json"),
    ("normal/ill-normal.stan", "normal/ill-normal.json"),
    (
        "multilevel_regression/multilevel_regression.stan",
        "multilevel_regression/multilevel_regression.json",
    ),
    (
        "multilevel_regression/multilevel_regression_logit.stan",
        "multilevel_regression/multilevel_regression_logit.json",
    ),
    (
        "measurement_error/measurement_error.stan",
        "measurement_error/measurement_error_1.json",
    ),
    (
        "hierarchical_matrix/hierarchical_matrix_1.stan",
        "hierarchical_matrix/hierarchical_matrix.json",
    ),
    (
        "hierarchical_matrix/hierarchical_matrix_2.stan",
        "hierarchical_matrix/hierarchical_matrix.json",
    ),

#   ("time-series/state_space_stochastic_level_stochastic_seasonal.stan", "time-series/uk_drivers.json"),  # low min ESS in both

#   ("ode/soil_incubation.stan", "ode/soil_carbon.json"),    # low min ESS in both

#   ("ode/sir.stan", "ode/sir.json"),    # initialization fails

#    (
#        "measurement_error/measurement_error.stan",
#        "measurement_error/measurement_error_5.json",
#    ),
#    ("funnel/funnel.stan", "funnel/funnel.json"),
]


class StanChain:
    def __init__(self, data, names, stepsize, inv_metric=None, cmdstan=None):
        self.data = data
        self.raw_parameters = names
        self.parameters = names
        self.warmup = SimpleNamespace(
            stepsize=stepsize, inv_metric=inv_metric, warmup_draws=None
        )
        self.cmdstan = cmdstan

    def __len__(self):
        return self.data.shape[0]


class NutpieChain:
    def __init__(self, data, names, stepsize, inv_metric=None, nutpie_result=None):
        self.data = data
        self.raw_parameters = names
        self.parameters = names
        self.warmup = SimpleNamespace(
            stepsize=stepsize, inv_metric=inv_metric, warmup_draws=None
        )
        self.nutpie = nutpie_result

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
    cols = dict(
        mean=s.mean(),
        sd=s.standard_deviation(),
        mcse=s.mcse(),
        ess=s.ess(),
        rhat=s.r_hat(),
    )
    k = len(cols["mean"])
    if names is None or len(names) != k:
        names = [f"v[{i}]" for i in range(k)]
    return names, cols, s


def print_summary(name, fit, names, cols, s, top=None):
    print(f"\n=== {name} ===")
    print(
        f"chains={len(fit)}  draws/chain={[np.asarray(c.data).shape[0] for c in fit]}"
        f"  total # draws={sum(len(c.data) for c in fit)}"
        f"  # parameters={s._stacked.shape[0]}"
    )
    print(f"{'param':<24}{'mean':>12}{'sd':>12}{'mcse':>10}{'ess':>10}{'rhat':>8}")
    idx = range(len(names)) if top is None else range(min(top, len(names)))
    for i in idx:
        print(
            f"{names[i]:<24}{cols['mean'][i]:>12.2f}{cols['sd'][i]:>12.2f}"
            f"{cols['mcse'][i]:>10.3f}{cols['ess'][i]:>10.0f}{cols['rhat'][i]:>8.2f}"
        )
    ok = np.isfinite(cols["rhat"]) & np.isfinite(cols["ess"]) & (cols["sd"] > 0)
    live = np.flatnonzero(ok)
    worst = int(live[np.argmax(cols["rhat"][live])])
    least = int(live[np.argmin(cols["ess"][live])])
    most = int(live[np.argmax(cols["ess"][live])])
    q = np.quantile(cols["ess"][live], [0.0, 0.1, 0.5, 0.9, 1.0])
    print(
        f"max rhat: {cols['rhat'][worst]:.3f} ({names[worst]})"
        f"{'' if ok.all() else f'  [{(~ok).sum()} constant/undefined columns excluded]'}"
    )
    print(
        f"ess quantiles over {live.size} parameters: "
        f"min {q[0]:.0f} ({names[least]})  10% {q[1]:.0f}  50% {q[2]:.0f}  "
        f"90% {q[3]:.0f}  max {q[4]:.0f} ({names[most]})"
    )


def chain_inv_metric(chain):
    m = getattr(chain.warmup, "inv_metric", None)
    if m is None:
        return None
    m = np.asarray(m, dtype=float)
    return np.diag(m) if m.ndim == 2 else m.ravel()


def print_inv_metrics(fit, pkg_name, per_line=METRIC_PER_LINE):
    ms = [chain_inv_metric(c) for c in fit]
    if all(m is None for m in ms):
        print("inverse metric: not available")
        return
    print("inverse metric diagonal (unconstrained scale):")
    for i, m in enumerate(ms):
        if pkg_name == "Nutpie":
            m = np.array(m)**2
        if m is None:
            print(f"  chain {i}: unavailable")
            continue
        print(
            f"  chain {i}  n={m.size}  min {m.min():.4f}"
            f"  median {np.median(m):.4f}  max {m.max():.4f}"
        )
        # Uncomment to print sqrt(mass matrices)
        # for j in range(0, m.size, per_line):
        #     print(f"    {j:>5}: " + " ".join(f"{v:8.3f}" for v in m[j : j + per_line]))


def print_fit(fit, stan_file, pkg_name, wall=None):
    print(f"\n\n{pkg_name}:" + (f"  wall {wall:.2f}s" if wall else ""))
    names, cols, s = summarize(fit, flat_names(fit))
    print_summary(pathlib.Path(stan_file).name, fit, names, cols, s, top=5)
    print("adapted step sizes:", [f"{c.warmup.stepsize:6.4f}" for c in fit])
    print_inv_metrics(fit, pkg_name)
    if getattr(fit[0], "cmdstan", None) is not None:
        print("diagnostics:", stan_diagnostics(fit))
    if getattr(fit[0], "nutpie", None) is not None:
        print("diagnostics:", nutpie_diagnostics(fit))


def stan_diagnostics(fit):
    mv = fit[0].cmdstan.method_variables()
    td = mv["treedepth__"]
    return dict(
        divergences=int(mv["divergent__"].sum()),
        treedepth_median=float(np.median(td)),
        treedepth_max=float(td.max()),
        treedepth_at_max=float((td >= td.max()).mean()),
    )


def nutpie_group(result, name):
    node = None
    try:
        node = result[name]
    except (KeyError, TypeError, IndexError):
        node = getattr(result, name, None)
    if node is None:
        return None
    to_dataset = getattr(node, "to_dataset", None)
    return to_dataset() if callable(to_dataset) else node


def nutpie_stat(stats, name):
    if stats is None or name not in stats:
        return None
    return np.asarray(stats[name].values, dtype=float)


def nutpie_diagnostics(fit):
    stats = nutpie_group(fit[0].nutpie, "sample_stats")
    div = nutpie_stat(stats, "diverging")
    depth = nutpie_stat(stats, "depth")
    if depth is None:
        depth = nutpie_stat(stats, "tree_depth")
    out = {}
    if div is not None:
        out["divergences"] = int(np.nansum(div))
    if depth is not None:
        out["treedepth_median"] = float(np.median(depth))
        out["treedepth_max"] = float(np.max(depth))
        out["treedepth_at_max"] = float((depth >= np.max(depth)).mean())
    return out


def nutpie_draws(ds):
    names, blocks = [], []
    for v in ds.data_vars:
        a = np.asarray(ds[v].values, dtype=float)
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        dims = a.shape[2:]
        if not dims:
            names.append(str(v))
        else:
            for k in range(flat.shape[2]):
                idx = np.unravel_index(k, dims)
                names.append(f"{v}[{','.join(str(i + 1) for i in idx)}]")
        blocks.append(flat)
    data = np.concatenate(blocks, axis=2)
    return data, names

def nutpie_last_finite(result, name):
    arrs = []
    for grp in ("warmup_sample_stats", "sample_stats"):
        a = nutpie_stat(nutpie_group(result, grp), name)
        if a is not None:
            arrs.append(a)
    if not arrs:
        return None
    a = np.concatenate(arrs, axis=1)
    out = []
    for c in range(a.shape[0]):
        rows = a[c]
        finite = np.flatnonzero(np.all(np.isfinite(rows.reshape(rows.shape[0], -1)), axis=1))
        out.append(rows[finite[-1]] if finite.size else None)
    return out


def nutpie_metrics(result):
    m = nutpie_last_finite(result, "mass_matrix_inv")
    if m is not None and any(x is not None for x in m):
        return m
    s = nutpie_last_finite(result, "mass_matrix_stds")
    return None if s is None else [None if x is None else np.asarray(x) ** 2 for x in s]


def fit_nutpie_one(stan_file, data_file, seed=SEED, num_chains=NUM_CHAINS,
                   tune=ITER_WARMUP, draws=ITER):
    compiled = nutpie.compile_stan_model(filename=str(stan_file), cache=True)
    with open(data_file) as f:
        compiled = compiled.with_data(**json.load(f))
    result = nutpie.sample(
        compiled,
        draws=draws,
        tune=tune,
        chains=num_chains,
        cores=num_chains,
        seed=seed,
        save_warmup=True,
        progress_bar=False,
        adaptation=NUTPIE_ADAPTATION,
        initial_step=0.1,
        target_accept=NUTPIE_TARGET_ACCEPT,
        maxdepth=10,
        store_mass_matrix=True,
    )
    posterior = nutpie_group(result, "posterior")
    data, names = nutpie_draws(posterior)
    step = nutpie_last_finite(result, "step_size")
    metrics = nutpie_metrics(result)
    return [
        NutpieChain(
            np.ascontiguousarray(data[c]),
            names,
            float(step[c]) if step is not None and step[c] is not None else float("nan"),
            metrics[c] if metrics is not None else None,
            result,
        )
        for c in range(data.shape[0])
    ]

def fit_walnutpie_one(stan_file, data_file, seed=SEED):
    model = bridgestan.StanModel(
        str(stan_file),
        str(data_file),
        make_args=["STAN_THREADS=1"],
        capture_stan_prints=False,
        seed=seed,
    )
    return walnutpie.walnuts_stan(
        model,
        num_chains=NUM_CHAINS,
        seed=seed,
        min_warmup_iter=MIN_ITER_WARMUP,
        max_warmup_iter=ITER_WARMUP,
        min_sampling_iter=MIN_ITER,
        max_sampling_iter=ITER,
        max_trajectory_doublings=10,  # 5 Walnutpie, 10 Good
        max_step_halvings=5,  # 5 Walnutpie, 5 good,
        # min_micro_steps=1,
        max_macro_steps_target=1024,  # XXXX 16.0 Walnutpie, 16 Good
        init_radius=0.1,  # 2.0 Walnutpie, 0.1 Good
        step_size_init=0.5,  # 1.0 Walnutpie, 0.1 Good
        max_hamiltonian_error=1e6,  # XXXX # 0.5 Walnutpie, 0.5--1 Good, infty Nuts
        mass_init_count=4.0,  # XXXX 4.0 Walnutpie, 1.01 Good
        rhat_converge_tol=1.01,  # 1.01 Walnutpie
        step_accept_rate_target=0.8,  # 0.8 Walnutpie, 0.9 Good
        step_learning_rate=0.05,  # 0.001 Adam default, 0.05 Walnutpie, 0.05 Good
        step_gradient_decay=0.8,  # 0.9 Adam, 0.8 Walnutpie, 0.8 Good
        step_sq_gradient_decay=0.9,  # 0.999 Adam, 0.9 Walnutpie, 0.9 Good
        step_stabilization=0.0001,  # 1e-7 Adam, 0.0001 Walnutpie, 0.0001 Good
        step_learn_rate_decay=0.5,  # 0.5 Walnutpie, 0.5 Good
        # init_inv_metric=np.ones(model.param_unc_num()),
        save_inv_metric=True,
        refresh=0,
    )


def stan_inv_metrics(csp_fit, num_chains):
    m = getattr(csp_fit, "metric", None)
    if m is None:
        return [None] * num_chains
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m[np.newaxis, :]
    return [m[c] if c < m.shape[0] else None for c in range(num_chains)]


def fit_stan_one(
    stan_file,
    data_file,
    seed=SEED,
    num_chains=NUM_CHAINS,
    iter_warmup=ITER_WARMUP,
    iter_sampling=ITER,
    **sample_kwargs,
):
    model = CmdStanModel(stan_file=str(stan_file))
    csp_fit = model.sample(
        data=str(data_file),
        chains=num_chains,
        parallel_chains=num_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        inits=0.1,
        show_progress=False,
        show_console=False,
        **sample_kwargs,
    )
    keep = [i for i, n in enumerate(csp_fit.column_names) if not n.endswith("__")]
    names = [csp_fit.column_names[i] for i in keep]
    draws = csp_fit.draws(concat_chains=False)
    step = np.atleast_1d(np.asarray(getattr(csp_fit, "step_size", np.nan), dtype=float))
    if step.size < draws.shape[1]:
        step = np.full(draws.shape[1], np.nan)
    metrics = stan_inv_metrics(csp_fit, draws.shape[1])
    return [
        StanChain(
            np.ascontiguousarray(draws[:, c, keep]),
            names,
            float(step[c]),
            metrics[c],
            csp_fit,
        )
        for c in range(draws.shape[1])
    ]


if __name__ == "__main__":
    for stan_file, json_file in STAN_JSON_PAIRS:
        print(f"\n\n***** {stan_file=} {json_file=} *****")
        stan_path = pathlib.Path(stan_file)
        json_path = pathlib.Path(json_file)

        t0 = time.perf_counter()
        fit_wnp = fit_walnutpie_one(stan_path, json_path)
        print_fit(fit_wnp, stan_path, "Walnutpie", time.perf_counter() - t0)

        # t0 = time.perf_counter()
        # fit_csp = fit_stan_one(stan_path, json_path)
        # print_fit(fit_csp, stan_path, "CmdStanPy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        fit_ntp = fit_nutpie_one(stan_path, json_path)
        print_fit(fit_ntp, stan_path, "Nutpie", time.perf_counter() - t0)
    print("")
