
``walnutpie``
======================================================================

``walnutpie`` is a Python package for Markov chain Monte Carlo
(MCMC) sampling of differentiable target log densities.

- ``walnutpie`` can run models coded in Stan, PyMC, NumPyro, JAX,
  Numba, as well as models coded directly in Python.

- The underlying sampler is `Walnuts
  <https://www.jmlr.org/beta/papers/v27/25-1452.html>`__, which adds
  dynamic step-size adaptation to the `no-U-turn sampler
  <https://jmlr.org/beta/papers/v15/hoffman14a.html>`__ (Nuts), which in turn is
  based on `Hamiltonian Monte Carlo <https://arxiv.org/abs/1206.1901>`__ (HMC).

- The mass matrix and step-size adaptation scheme use an online
  variant of `Nutpie <https://arxiv.org/abs/2603.18845v1>`__

- For estimating the maximum step size during warmup, Walnuts uses `Adam
  <https://arxiv.org/abs/1412.6980>`__ rather than dual averaging for
  stochastic gradient descent.

- Chain execution is multithreaded with optional convergence detection
  for warmup and sampling through lock-free buffers.

- Posterior analysis tools are included for the varying-length
  chains produced by asynchronous automatic stopping.


Python library
--------------
.. toctree::
   :maxdepth: 2

   install
   py
   example.ipynb

``walnutpie`` accepts target log densities and gradients directly in Python,
including those coded in `NumPyro <https://num.pyro.ai/en/latest/index.html>`__,
`PyMC <https://www.pymc.io/welcome.html>`__, or `JAX
<https://docs.jax.dev/en/latest/>`__, even with foreign function calls. `Stan
<https://mc-stan.org>`__ models can be accessed directly at the C++ level
through `BridgeStan <https://roualdes.us/bridgestan/latest/>`__.


C++ interface documentation
---------------------------

``walnutpie`` is implemented in multi-threaded C++20 with a stable
client-facing API.

.. toctree::
   :maxdepth: 2

   cpp


License
-------

- ``walnutpie`` is distributed under the `MIT License <https://opensource.org/license/mit>`__.


Bug reports and feature requests
--------------------------------

Bug reports and feature requests are handled through GitHub.

- `Walnutpie issue tracker <https://github.com/flatironinstitute/walnutpie/issues>`__


Developers and other contributors
---------------------------------

We welcome new developers to the project and try to maintain a
friendly and constructive environment. To get started, see the
developers guide on GitHub:

- `Contributing to walnutpie <https://github.com/flatironinstitute/walnutpie/blob/main/CONTRIBUTING.md>`__
