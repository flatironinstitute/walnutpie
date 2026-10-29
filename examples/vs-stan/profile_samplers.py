from dataclasses import dataclass
from types import SimpleNamespace
import json
import logging
import pathlib
import sys
import time
import traceback
import warnings

import numpy as np
import bridgestan
import cmdstanpy
import nutpie
import walnutpie
from cmdstanpy import CmdStanModel

warnings.simplefilter(action="ignore", category=FutureWarning)
cmdstanpy.utils.get_logger().setLevel(logging.ERROR)


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True)
class RunConfig:
    seed: int = 456654
    num_chains: int = 16
    iter_warmup: int = 400
    iter_sampling: int = 400
    max_tree_depth: int = 10  # Walnutpie doublings: 5 Walnutpie, 10 Good
    init_radius: float = 0.1
    top_params: int = 5  # rows printed in per-fit summary
    metric_per_line: int = 100


RUN = RunConfig()

ACTIVE_SAMPLERS = ["Walnutpie", "Nutpie"]  # also: "CmdStanPy"

WALNUTPIE_ARGS = dict(
    max_step_halvings=1,  # 5 Walnutpie, 5 good, 1 is Nuts
    min_micro_steps=1,
    max_macro_steps_target=1024,  # XXXX 16.0 Walnutpie, 16 Good
    step_size_init=0.1,  # 1.0 Walnutpie, 0.1 Good
    max_hamiltonian_error=1e10,  # XXXX # 0.5 Walnutpie, 0.5--1 Good, infty Nuts
    mass_init_count=4.0,  # XXXX 4.0 Walnutpie, 1.01 Good
    rhat_converge_tol=1.01,  # 1.01 Walnutpie
    step_accept_rate_target=0.8,  # 0.8 Walnutpie, 0.9 Good
    step_learning_rate=0.025,  # 0.001 Adam default, 0.05 Walnutpie, 0.05 Good
    step_gradient_decay=0.8,  # 0.9 Adam, 0.8 Walnutpie, 0.8 Good
    step_sq_gradient_decay=0.9,  # 0.999 Adam, 0.9 Walnutpie, 0.9 Good
    step_stabilization=1e-7,  # 1e-7 Adam, 0.0001 Walnutpie, 0.0001 Good
    step_learn_rate_decay=0.5,  # 0.5 Walnutpie, 0.5 Good
    # init_inv_metric=np.ones(model.param_unc_num()),
)

NUTPIE_ARGS = dict(
    adaptation="diag",
    initial_step=0.1,
    target_accept=0.8,
)

STAN_ARGS = dict()

MODELS = [
    ("spectroscopy/spectroscopy.stan", "spectroscopy/spectroscopy.json"), 

    ("normal/std-normal.stan", "normal/std-normal.json"),
    ("normal/ill-normal.stan", "normal/ill-normal.json"),

    ("time-series/arK.stan", "time-series/arK.json"),
    ("time-series/arma11.stan", "time-series/arma.json"),
    ("time-series/garch11.stan", "time-series/garch.json"),

    ("eight-schools/eight_schools_centered.stan", "eight-schools/eight_schools.json"),
    ("eight-schools/eight_schools_noncentered.stan", "eight-schools/eight_schools.json"),

    ("multilevel_regression/multilevel_regression.stan",
     "multilevel_regression/multilevel_regression.json"),
    ("multilevel_regression/multilevel_regression_logit.stan",
     "multilevel_regression/multilevel_regression_logit.json"),

    ("measurement_error/measurement_error.stan",
     "measurement_error/measurement_error_1.json"),

    ("hierarchical_matrix/hierarchical_matrix_1.stan",  # hard for Nutpie
     "hierarchical_matrix/hierarchical_matrix.json"),
    ("hierarchical_matrix/hierarchical_matrix_2.stan",
     "hierarchical_matrix/hierarchical_matrix.json"),

    ("wells/wells_daae_c_model.stan", "wells/wells_data.json"),
    ("wells/wells_interaction_c_model.stan", "wells/wells_data.json"),
    ("wells/wells_dist.stan", "wells/wells_data.json"),

    ("seeds/seeds_model.stan", "seeds/seeds_data.json"),
    ("seeds/seeds_centered_model.stan", "seeds/seeds_data.json"),
    ("seeds/seeds_stanified_model.stan", "seeds/seeds_data.json"),

    ("rats/rats_model.stan", "rats/rats_data.json"),  # hard for Walnutpie

    ("prophet/prophet.stan", "prophet/rstan_downloads.json"),

    ("occupancy/multi_occupancy.stan", "occupancy/butterfly.json"),

    ("capture/M0_model.stan", "capture/M0_data.json"),
    ("capture/Mtbh_model.stan", "capture/Mtbh_data.json"),

    ("lsat/lsat_model.stan", "lsat/lsat_data.json"),

    ("losscurve/losscurve_sislob.stan", "losscurve/loss_curves.json"),

    ("election/election88_full.stan", "election/election88.json"),

    ("spatial/bym2_offset_only.stan", "spatial/traffic_accident_nyc.json"),  # nutpie much slower

    ("diamonds/diamonds.stan", "diamonds/diamonds.json"),  # 2.4 MB data, slow

    ("irt/irt_2pl.stan", "irt/irt_2pl.json"),

    ("earn/earn_height.stan", "earn/earnings.json"),  # *** doesn't fit in Walnutpie -- metric to 500K ***
    ("earn/logearn_height.stan", "earn/earnings.json"),

    ("kilpisjarvi/kilpisjarvi.stan", "kilpisjarvi/kilpisjarvi_mod.json"),  # *** doesn't fit in Walnutpie ***

    ("nes/nes.stan", "nes/nes1980.json"),
    ("nes/nes.stan", "nes/nes2000.json"),

    ("mesquite/logmesquite_logvolume.stan", "mesquite/mesquite.json"),

    ("mixture/low_dim_gauss_mix.stan", "mixture/low_dim_gauss_mix.json"),

    ("kid/kidscore_interaction.stan", "kid/kidiq.json"),

    ("blr/blr.stan", "blr/sblrc.json"),  # bad for Walnutpie, but works in Nutpie

    ("radon/radon_pooled.stan", "radon/radon_mn.json"),
    ("radon/radon_pooled.stan", "radon/radon_all.json"),

    ("hmm/hmm_example.stan", "hmm/hmm_example.json"),

    ("gp/accel_gp.stan", "gp/mcycle_gp.json"),  # hard to fit
]

DISABLED_MODELS = {
    ("measurement_error/measurement_error.stan", "measurement_error/measurement_error_5.json"):
        "very big funnel doesn't fit well",
    ("gp/gp_pois_regr.stan", "gp/gp_pois_regr.json"):
        "seg faults Nutpie (maybe bad config?)",
    ("mnist/nn_rbm1bJ10.stan", "mnist/mnist_100.json"):
        "very slow with 8K params; OK with fewer iterations",
    ("hmm/hmm_gaussian.stan", "hmm/hmm_gaussian_simulated.json"):
        "doesn't fit in either",
    ("time-series/state_space_stochastic_level_stochastic_seasonal.stan", "time-series/uk_drivers.json"):
        "low min ESS in both",
    ("ode/soil_incubation.stan", "ode/soil_carbon.json"):
        "low min ESS in both",
    ("ode/sir.stan", "ode/sir.json"):
        "initialization fails",
    ("ode/lotka_volterra.stan", "ode/hudson_lynx_hare.json"):
        "initialization fails",
    ("funnel/funnel.stan", "funnel/funnel.json"):
        "",
}

SUMMARY_FORMATS = dict(
    time_s="{:.1f}".format,
    max_rhat="{:.3f}".format,
    min_ess="{:.0f}".format,
    med_ess="{:.0f}".format,
    mean_step="{:.4f}".format,
)


# ======================================================================
# Chain containers
# ======================================================================

class Chain:
    def __init__(self, data, names, stepsize, inv_metric=None, **backend):
        self.data = data
        self.raw_parameters = names
        self.parameters = names
        self.warmup = SimpleNamespace(
            stepsize=stepsize, inv_metric=inv_metric, warmup_draws=None
        )
        self.cmdstan = backend.get("cmdstan")
        self.nutpie = backend.get("nutpie")

    def __len__(self):
        return self.data.shape[0]


# ======================================================================
# Summaries
# ======================================================================

def model_label(stan_file, data_file):
    return f"{pathlib.Path(stan_file).stem}/{pathlib.Path(data_file).stem}"


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


def ess_rhat_stats(cols):
    ok = np.isfinite(cols["rhat"]) & np.isfinite(cols["ess"]) & (cols["sd"] > 0)
    live = np.flatnonzero(ok)
    return SimpleNamespace(
        ok=ok,
        live=live,
        worst=int(live[np.argmax(cols["rhat"][live])]),
        least=int(live[np.argmin(cols["ess"][live])]),
        most=int(live[np.argmax(cols["ess"][live])]),
        q=np.quantile(cols["ess"][live], [0.0, 0.1, 0.5, 0.9, 1.0]),
    )


def print_summary(label, fit, names, cols, s, st, top=None):
    print(f"\n=== {label} ===")
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
    excluded = "" if st.ok.all() else f"  [{(~st.ok).sum()} constant/undefined columns excluded]"
    print(f"max rhat: {cols['rhat'][st.worst]:.3f} ({names[st.worst]}){excluded}")
    q = st.q
    print(
        f"ess quantiles over {st.live.size} parameters: "
        f"min {q[0]:.0f} ({names[st.least]})  10% {q[1]:.0f}  50% {q[2]:.0f}  "
        f"90% {q[3]:.0f}  max {q[4]:.0f} ({names[st.most]})"
    )


def chain_inv_metric(chain):
    m = getattr(chain.warmup, "inv_metric", None)
    if m is None:
        return None
    m = np.asarray(m, dtype=float)
    return np.diag(m) if m.ndim == 2 else m.ravel()


def print_inv_metrics(fit, sampler, per_line=RUN.metric_per_line):
    ms = [chain_inv_metric(c) for c in fit]
    if all(m is None for m in ms):
        print("inverse metric: not available")
        return
    print("inverse metric diagonal (unconstrained scale):")
    for i, m in enumerate(ms):
        if m is None:
            print(f"  chain {i}: unavailable")
            continue
        if sampler == "Nutpie":
            m = m**2
        print(
            f"  chain {i}  n={m.size}  min {m.min():.4f}"
            f"  median {np.median(m):.4f}  max {m.max():.4f}"
        )
        # for j in range(0, m.size, per_line):
        #     print(f"    {j:>5}: " + " ".join(f"{v:8.3f}" for v in m[j : j + per_line]))


def summary_row(label, sampler, wall, cols=None, st=None, steps=None):
    if cols is None:
        return dict(model=label, sampler=sampler, time_s=wall, max_rhat=np.nan,
                    min_ess=np.nan, med_ess=np.nan, mean_step=np.nan)
    return dict(
        model=label,
        sampler=sampler,
        time_s=wall,
        max_rhat=float(cols["rhat"][st.worst]),
        min_ess=float(st.q[0]),
        med_ess=float(st.q[2]),
        mean_step=float(np.nanmean(steps)),
    )


def print_fit(fit, label, sampler, wall):
    print(f"\n\n{sampler}:  wall {wall:.2f}s")
    names, cols, s = summarize(fit, flat_names(fit))
    st = ess_rhat_stats(cols)
    print_summary(label, fit, names, cols, s, st, top=RUN.top_params)
    steps = [c.warmup.stepsize for c in fit]
    print("adapted step sizes:", [f"{x:6.4f}" for x in steps])
    print_inv_metrics(fit, sampler)
    if getattr(fit[0], "cmdstan", None) is not None:
        print("diagnostics:", stan_diagnostics(fit))
    if getattr(fit[0], "nutpie", None) is not None:
        print("diagnostics:", nutpie_diagnostics(fit))
    return summary_row(label, sampler, wall, cols, st, steps)


def format_cell(key, value):
    if key in SUMMARY_FORMATS:
        return "-" if np.isnan(value) else SUMMARY_FORMATS[key](value)
    return str(value)


def print_table(rows, text_cols=("model", "sampler")):
    keys = [*text_cols, *SUMMARY_FORMATS]
    cells = [[format_cell(k, r[k]) for k in keys] for r in rows]
    widths = [max(len(k), *(len(c[j]) for c in cells)) for j, k in enumerate(keys)]
    align = ["<" if k in text_cols else ">" for k in keys]
    line = lambda xs: "  ".join(f"{x:{a}{w}}" for x, a, w in zip(xs, align, widths))
    print("\n\n===== SUMMARY =====")
    print(line(keys))
    for c in cells:
        print(line(c))


# ======================================================================
# CmdStanPy
# ======================================================================

def stan_diagnostics(fit):
    mv = fit[0].cmdstan.method_variables()
    td = mv["treedepth__"]
    return dict(
        divergences=int(mv["divergent__"].sum()),
        treedepth_median=float(np.median(td)),
        treedepth_max=float(td.max()),
        treedepth_at_max=float((td >= td.max()).mean()),
    )


def stan_inv_metrics(csp_fit, num_chains):
    m = getattr(csp_fit, "metric", None)
    if m is None:
        return [None] * num_chains
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m[np.newaxis, :]
    return [m[c] if c < m.shape[0] else None for c in range(num_chains)]


def fit_stan(stan_file, data_file, run=RUN):
    model = CmdStanModel(stan_file=str(stan_file))
    csp_fit = model.sample(
        data=str(data_file),
        chains=run.num_chains,
        parallel_chains=run.num_chains,
        iter_warmup=run.iter_warmup,
        iter_sampling=run.iter_sampling,
        seed=run.seed,
        inits=run.init_radius,
        max_treedepth=run.max_tree_depth,
        show_progress=False,
        show_console=False,
        **STAN_ARGS,
    )
    keep = [i for i, n in enumerate(csp_fit.column_names) if not n.endswith("__")]
    names = [csp_fit.column_names[i] for i in keep]
    draws = csp_fit.draws(concat_chains=False)
    step = np.atleast_1d(np.asarray(getattr(csp_fit, "step_size", np.nan), dtype=float))
    if step.size < draws.shape[1]:
        step = np.full(draws.shape[1], np.nan)
    metrics = stan_inv_metrics(csp_fit, draws.shape[1])
    return [
        Chain(np.ascontiguousarray(draws[:, c, keep]), names, float(step[c]),
              metrics[c], cmdstan=csp_fit)
        for c in range(draws.shape[1])
    ]


# ======================================================================
# Nutpie
# ======================================================================

def nutpie_group(result, name):
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
    return np.concatenate(blocks, axis=2), names


def nutpie_last_finite(result, name):
    arrs = [
        a for grp in ("warmup_sample_stats", "sample_stats")
        if (a := nutpie_stat(nutpie_group(result, grp), name)) is not None
    ]
    if not arrs:
        return None
    a = np.concatenate(arrs, axis=1)
    out = []
    for rows in a:
        finite = np.flatnonzero(np.all(np.isfinite(rows.reshape(rows.shape[0], -1)), axis=1))
        out.append(rows[finite[-1]] if finite.size else None)
    return out


def nutpie_metrics(result):
    m = nutpie_last_finite(result, "mass_matrix_inv")
    if m is not None and any(x is not None for x in m):
        return m
    s = nutpie_last_finite(result, "mass_matrix_stds")
    return None if s is None else [None if x is None else np.asarray(x) ** 2 for x in s]


def fit_nutpie(stan_file, data_file, run=RUN):
    compiled = nutpie.compile_stan_model(filename=str(stan_file), cache=True)
    with open(data_file) as f:
        compiled = compiled.with_data(**json.load(f))
    result = nutpie.sample(
        compiled,
        draws=run.iter_sampling,
        tune=run.iter_warmup,
        chains=run.num_chains,
        cores=run.num_chains,
        seed=run.seed,
        maxdepth=run.max_tree_depth,
        save_warmup=True,
        progress_bar=False,
        store_mass_matrix=True,
        **NUTPIE_ARGS,
    )
    data, names = nutpie_draws(nutpie_group(result, "posterior"))
    step = nutpie_last_finite(result, "step_size")
    metrics = nutpie_metrics(result)
    return [
        Chain(
            np.ascontiguousarray(data[c]),
            names,
            float(step[c]) if step is not None and step[c] is not None else np.nan,
            metrics[c] if metrics is not None else None,
            nutpie=result,
        )
        for c in range(data.shape[0])
    ]


# ======================================================================
# Walnutpie
# ======================================================================

def fit_walnutpie(stan_file, data_file, run=RUN):
    model = bridgestan.StanModel(
        str(stan_file),
        str(data_file),
        make_args=["STAN_THREADS=1"],
        capture_stan_prints=False,
        seed=run.seed,
    )
    args = dict(
        min_warmup_iter=run.iter_warmup,
        max_warmup_iter=run.iter_warmup,
        min_sampling_iter=run.iter_sampling,
        max_sampling_iter=run.iter_sampling,
        max_trajectory_doublings=run.max_tree_depth,
        init_radius=run.init_radius,
    ) | WALNUTPIE_ARGS
    return walnutpie.walnuts_stan(
        model,
        num_chains=run.num_chains,
        seed=run.seed,
        save_inv_metric=True,
        refresh=0,
        **args,
    )


# ======================================================================
# Driver
# ======================================================================

SAMPLERS = {
    "Walnutpie": fit_walnutpie,
    "Nutpie": fit_nutpie,
    "CmdStanPy": fit_stan,
}


def run_one(sampler, stan_path, data_path):
    label = model_label(stan_path, data_path)
    t0 = time.perf_counter()
    try:
        fit = SAMPLERS[sampler](stan_path, data_path, RUN)
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"\n\n{sampler}: FIT FAILED after {wall:.2f}s ({type(e).__name__}: {e})")
        return summary_row(label, sampler, wall)
    wall = time.perf_counter() - t0
    try:
        return print_fit(fit, label, sampler, wall)
    except Exception as e:
        print(f"\n{sampler}: REPORT FAILED ({type(e).__name__}: {e})")
        traceback.print_exc(file=sys.stdout)
        return summary_row(label, sampler, wall)


def main():
    rows = []
    for stan_file, data_file in MODELS:
        print(f"\n\n***** {stan_file=} {data_file=} *****")
        for sampler in ACTIVE_SAMPLERS:
            rows.append(run_one(sampler, pathlib.Path(stan_file), pathlib.Path(data_file)))
    print_table(rows)
    print("")


if __name__ == "__main__":
    main()
