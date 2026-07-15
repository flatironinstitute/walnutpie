Python API
==========


Sampling functions
------------------


.. autofunction:: walnutpie.walnuts_pyfunc
.. autofunction:: walnutpie.walnuts_stan


Note on reproducibility
_______________________

When the adaptive stopping criteria are used (i.e., when ``min_warmup_iter !=
max_warmup_iter`` and ``min_sampling_iter != max_sampling_iter``), thread
scheduling will result in different draws being produced for the same seed. When
precise reproducibility is required, setting these numbers to the same value
will turn off adaptive stopping and result in reproducible runs.


Output Classes
______________

.. autoclass:: walnutpie.pyfunc::WalnutsOutputArray
   :members:
   :show-inheritance:

.. autoclass:: walnutpie.stan::StanOutput
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __getitem__

.. autoclass:: walnutpie.stan::StanOutputBase

.. autoclass:: walnutpie.util::WarmupInfo
   :members:

Posterior Analysis Functions
----------------------------

When adaptive stopping is used, it is likely that each chain will end with a
different number of draws, which may by a challenge to process using existing
tools. The following common posterior analysis functions are implemented in such
a manner to account for this.

.. autoclass:: walnutpie.Summarizer
   :members:

.. autofunction:: walnutpie.ess
.. autofunction:: walnutpie.r_hat
.. autofunction:: walnutpie.mcse
.. autofunction:: walnutpie.mean
.. autofunction:: walnutpie.variance
.. autofunction:: walnutpie.standard_deviation
