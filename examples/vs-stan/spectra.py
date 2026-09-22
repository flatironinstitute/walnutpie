import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import plotnine as pn
import bridgestan as bs

PDB = Path("~/github/stan-dev/posteriordb/posterior_database").expanduser()
PRECONDITIONERS = ["unit", "Stan", "Nutpie"]


def identity(frame):
    return frame


class Posterior(NamedTuple):
    name: str
    stan: Path
    data: Path
    draws: Path
    recover: Callable = identity


def pdb_posterior(model, data, draws=None, recover=identity):
    name = f"{data}-{model}"
    return Posterior(
        name,
        PDB / "models" / "stan" / f"{model}.stan",
        PDB / "data" / "data" / f"{data}.json.zip",
        PDB / "reference_posteriors" / "draws" / "draws" / f"{draws or name}.json.zip",
        recover,
    )


def noncentered_eight_schools(frame):
    thetas = [c for c in frame.columns if c.startswith("theta[")]
    trans = {c.replace("theta", "theta_trans"): (frame[c] - frame.mu) / frame.tau for c in thetas}
    return frame.assign(**trans)


POSTERIORS = [
    pdb_posterior("radon_pooled", "radon_mn"),
    # pdb_posterior("radon_pooled", "radon_all"),
    # pdb_posterior("blr","sblrc"),
    # pdb_posterior("wells_dist", "wells_data"),
    # pdb_posterior("kidscore_interaction", "kidiq"),
    # pdb_posterior("low_dim_gauss_mix", "low_dim_gauss_mix"),
    # pdb_posterior("logmesquite_logvolume", "mesquite"),
    # pdb_posterior("nes", "nes1980"),
    # pdb_posterior("nes", "nes2000"),
    # pdb_posterior("mesquite", "mesquite"),
    # pdb_posterior("accel_gp", "mcycle_gp"),
    # pdb_posterior("kilpisjarvi", "kilpisjarvi_mod"),
    # pdb_posterior("hmm_example", "hmm_example"),
    # pdb_posterior("logearn_height", "earnings"),
    # pdb_posterior("earn_height", "earnings"),
    # pdb_posterior("gp_pois_regr", "gp_pois_regr"),
    # pdb_posterior("arK", "arK"),
    # pdb_posterior("irt_2pl", "irt_2pl"),
    # pdb_posterior("arma11", "arma"),
    # pdb_posterior("eight_schools_centered", "eight_schools", draws="eight_schools-eight_schools_noncentered"),
    # pdb_posterior(
    #    "eight_schools_noncentered", "eight_schools", recover=noncentered_eight_schools
    # ),
    # pdb_posterior("diamonds", "diamonds"), # SLOW

    ### pdb_posterior("Rate_1_model", "Rate_1_data"),  # ONE PARAMETER, NO CONDITION
    ### pdb_posterior("hmm_drive_0", "bball_drive_event_0"), # EIGENVALUES FAIL
    ### pdb_posterior("lotka_volterra", "hudson_lynx_hare"), # NO HESSIANS
    ### pdb_posterior("GLMM_Poisson_model", "GLMM_Poisson_data"), # NO REFERENCE
]

def zip_member(z, path):
    names = [n for n in z.namelist() if n.endswith(".json") and "__MACOSX" not in n]
    exact = [n for n in names if Path(n).name == path.stem]
    (member,) = exact or names
    return member


def read_json(path):
    path = Path(path)
    if path.suffix != ".zip":
        return json.loads(path.read_text())
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(zip_member(z, path)))


def pdb_name(bs_name):
    base, *idx = bs_name.split(".")
    return f"{base}[{','.join(idx)}]" if idx else base


def load_model(posterior):
    data = json.dumps(read_json(posterior.data))
    return bs.StanModel(str(posterior.stan), data, make_args=["BRIDGESTAN_AD_HESSIAN=true"])


def draws_frame(path):
    return pd.concat([pd.DataFrame(chain) for chain in read_json(path)], ignore_index=True)


def select_params(frame, names):
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise KeyError(f"draws lack {missing}; available: {list(frame.columns)}")
    return np.ascontiguousarray(frame[names].to_numpy())


def constrained_draws(posterior, model):
    frame = posterior.recover(draws_frame(posterior.draws))
    return select_params(frame, [pdb_name(n) for n in model.param_names()])


def unconstrain(model, draws):
    return np.array([model.param_unconstrain(d) for d in draws])


def scores(model, draws):
    return np.array([model.log_density_gradient(d)[1] for d in draws])


def inverse_metrics(draws, scores):
    var = draws.var(axis=0)
    return {
        "unit": np.ones(draws.shape[1]),
        "Stan": var,
        "Nutpie": np.sqrt(var / scores.var(axis=0)),
    }


def neg_hessian(model, draw):
    try:
        return -model.log_density_hessian(draw)[2], None
    except RuntimeError as err:
        return np.full((len(draw), len(draw)), np.nan), str(err)


def hessian_problem(hessian, error):
    if error:
        return error
    if not np.isfinite(hessian).all():
        return "non-finite Hessian entries"
    return None


def precondition(hessian, inv_metric):
    root = np.sqrt(inv_metric)
    return root[:, None] * hessian * root[None, :]


def eigenvalues(matrix):
    failed = np.full(len(matrix), np.nan)
    if not np.isfinite(matrix).all():
        return failed
    try:
        return np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return failed


def spectra(model, draws, inv_metrics):
    eigs = {name: [] for name in inv_metrics}
    problems = Counter()
    for draw in draws:
        hess, error = neg_hessian(model, draw)
        if problem := hessian_problem(hess, error):
            problems[problem] += 1
        for name, m in inv_metrics.items():
            eigs[name].append(eigenvalues(precondition(hess, m)))
    return {name: np.array(e) for name, e in eigs.items()}, problems


def spectrum_frame(posterior_name, eigs):
    frames = [
        pd.DataFrame({
            "posterior": posterior_name,
            "preconditioner": name,
            "draw": np.repeat(np.arange(e.shape[0]), e.shape[1]),
            "eigenvalue": e.ravel(),
        })
        for name, e in eigs.items()
    ]
    return pd.concat(frames, ignore_index=True)


def analyze(posterior, thin=1):
    model = load_model(posterior)
    draws = unconstrain(model, constrained_draws(posterior, model))
    inv_metrics = inverse_metrics(draws, scores(model, draws))
    eigs, problems = spectra(model, draws[::thin], inv_metrics)
    return spectrum_frame(posterior.name, eigs), problems


def draw_summaries(df):
    s = df.groupby(["preconditioner", "draw"]).eigenvalue.agg(lo="min", hi="max").reset_index()
    return s.assign(
        step=2 / np.sqrt(s.hi.where(s.hi > 0)),
        condition=(s.hi / s.lo).where(s.lo > 0),
    )


def decade_exponents(x):
    return (
        int(np.floor(min(np.log10(x.min()), 0))),
        int(np.ceil(max(np.log10(x.max()), 0))),
    )


def decade_label(k):
    return "1" if k == 0 else f"1e{k}"


def log_density_plot(df, x, xlabel, title, reference=None):
    df = df[df[x] > 0].assign(
        preconditioner=lambda d: pd.Categorical(d.preconditioner, PRECONDITIONERS, ordered=True)
    )
    if df.empty:
        return None
    lo, hi = decade_exponents(df[x])
    ks = range(lo, hi + 1)
    plot = (
        pn.ggplot(df, pn.aes(x))
        + pn.geom_density()
        + pn.scale_x_log10(
            limits=(10.0**lo, 10.0**hi),
            breaks=[10.0**k for k in ks],
            labels=[decade_label(k) for k in ks],
        )
        + pn.facet_wrap("~preconditioner", nrow=1)
        + pn.labs(x=xlabel, y="density", title=title)
        + pn.theme_bw()
        + pn.theme(strip_background=pn.element_rect(fill="#f0f0f0"), figure_size=(12, 3))
    )
    return plot + pn.geom_vline(xintercept=reference, linetype="dashed") if reference else plot


def failure_counts(df):
    failed = df[df.eigenvalue.isna()].groupby("preconditioner").draw.nunique()
    return failed.reindex(PRECONDITIONERS, fill_value=0)


def failure_note(df):
    counts = failure_counts(df)
    if not counts.any():
        return ""
    n = df.draw.nunique()
    return "; failed of " + str(n) + ": " + ", ".join(f"{k} {v}" for k, v in counts.items())


def posterior_plots(df):
    name = df.posterior.iat[0]
    summaries = draw_summaries(df)
    pct = 100 * (df.eigenvalue.dropna() <= 0).mean()
    return {
        "eigenvalues": log_density_plot(
            df, "eigenvalue", "eigenvalue of preconditioned negative Hessian",
            name + failure_note(df), reference=1,
        ),
        "step": log_density_plot(
            summaries, "step", "leapfrog stability limit 2 / sqrt(max eigenvalue)", name
        ),
        "condition": log_density_plot(
            summaries, "condition", "condition number",
            f"{name}: {pct:.2g}% nonpositive eigenvalues (draws with any omitted)",
        ),
    }


def report_problems(problems, shown=3):
    for problem, count in problems.most_common(shown):
        print(f"  {count} draws: {problem}")


def missing_files(posterior):
    return [str(p) for p in (posterior.stan, posterior.data, posterior.draws) if not Path(p).exists()]


if __name__ == "__main__":
    for posterior in POSTERIORS:
        if missing := missing_files(posterior):
            print(f"skipping {posterior.name}; missing {missing}")
            continue
        try:
            df, problems = analyze(posterior, thin=10)
        except KeyError as err:
            print(f"skipping {posterior.name}; {err}")
            continue
        if note := failure_note(df):
            print(f"{posterior.name}{note}")
        report_problems(problems)
        for kind, plot in posterior_plots(df).items():
            if plot is None:
                print(f"  no {kind} plot; nothing finite and positive to plot")
                continue
            plot.save(f"{posterior.name}-{kind}.pdf")
