# Walnuts vs. Nuts

Side-by-side fits of the same Stan programs with **walnutpie** (Walnuts, through
BridgeStan) and **CmdStanPy** (Nuts, through CmdStan) with summaries
provided by `walnutpie.Summarizer` in both cases.

`fit.py` is the top-level script. The models live in subdirectories
next to it, each with its Stan program and a JSON data file and
control over which models get fit by editing `fit.py` directly.

## Prerequisite: C++ toolchain and GNU make

Both BridgeStan and CmdStan compile Stan programs to native
code. These are usually present in Linux and can be installed on macOS
and Windows as follows.

- macOS: `xcode-select --install`
- Windows: RTools, which can be provided by [CmdStanPy's installer](https://mc-stan.org/cmdstanpy/installation.html).
  
## Setup

From the repository root:

```sh
python3 -m venv .venv-vs-stan
source .venv-vs-stan/bin/activate    # Windows: .venv-vs-stan\Scripts\activate
pip install -U pip
pip install numpy bridgestan cmdstanpy walnutpie
install_cmdstan                      # Windows: install_cmdstan --compiler
```

`install_cmdstan` comes with CmdStanPy and installs CmdStan in
`~/.cmdstan` unless the `CMDSTAN` environment variable says
otherwise. The command `install_cmdstan --help` lists the version and
directory options.

BridgeStan needs no separate step — it downloads its C++ sources on
first model compile. For details and troubleshooting, see

* [CmdStanPy installation](https://mc-stan.org/cmdstanpy/installation.html) and

* [BridgeStan getting started](https://roualdes.us/bridgestan/latest/getting-started.html)

To inspect the environment:

```sh
python -c "import walnutpie, bridgestan, cmdstanpy
print(walnutpie.__version__, walnutpie.__file__)
print(bridgestan.__version__)
print(cmdstanpy.__version__, cmdstanpy.cmdstan_path())"
```

`walnutpie.__file__` must fall inside `.venv-vs-stan`. If it points somewhere
like `~/Library/Python/3.x/...`, an earlier `pip install --user walnutpie` is
shadowing the virtual environment; remove it, or run `python -m pip` against the
venv's interpreter by absolute path. To get past an early bug,
**walnutpie must be 0.0.4 or newer.** 

## Running against local Walnutpie
 
To test changes to walnutpie itself rather than the released package, install it
from this checkout instead of from PyPI, following
[walnutpie's installation page](https://flatironinstitute.github.io/walnutpie/latest/install.html)
— that path needs CMake and the `thirdparty/bridgestan` submodule.

## Running

With the environment activated:

```sh
cd examples/vs-stan
python fit.py
```

Edit `STAN_JSON_PAIRS` at the top of `fit.py` to select which models to fit; all
but the first are commented out. Paths there are relative to this directory, so
run from here.

Each pair produces two blocks — one per engine — giving per-parameter mean, sd,
MCSE, ESS and R-hat for the first 25 parameters, then the worst R-hat and lowest
ESS over all of them, the adapted step size per chain, and the wall time. The
CmdStanPy block adds divergence and treedepth diagnostics.

The first run of any model pays for two compilations, one per engine. Time the
second run, not the first.

## Notes

Sampler settings for experimentation can be set directly in `fit.py`.  

