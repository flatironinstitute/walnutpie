C++ API
=======

Sampling functionality
----------------------

Top-level call
______________

This function will spawn threads to perform end-to-end sampling

.. doxygenfunction:: walnutpie::walnuts


Iterator-style samplers
_______________________

These classes implement Walnuts in an iteration-per-call style iterator.

.. doxygenclass:: walnutpie::WalnutsSampler
   :members:
.. doxygenclass:: walnutpie::AdaptiveWalnuts
   :members:

Configuration
-------------

The following classes (and their builders) are used to configure Walnuts.

.. doxygenclass:: walnutpie::WalnutsConfig
   :members:

.. doxygenclass:: walnutpie::InitConfigBuilder
   :members:
.. doxygenclass:: walnutpie::WarmupConfigBuilder
   :members:
.. doxygenclass:: walnutpie::SamplingConfigBuilder
   :members:


.. doxygenclass:: walnutpie::InitConfig
.. doxygenclass:: walnutpie::WarmupConfig
.. doxygenclass:: walnutpie::SamplingConfig


Concepts
--------

The following concepts describe the types expected by `walnutpie`.

.. doxygenconcept:: walnutpie::LogpGrad
.. doxygenconcept:: walnutpie::ErrorCallback
.. doxygenconcept:: walnutpie::SampleHandler
.. doxygenconcept:: walnutpie::ChainHandler
.. doxygenconcept:: walnutpie::GlobalHandler
.. doxygenconcept:: walnutpie::InterruptCallback

Endpoint reuse (opt-in)
-----------------------

``WalnutsSampler`` and ``AdaptiveWalnuts`` accept a final constructor argument,
``walnutpie::EndpointReuse::Deterministic``. The default is ``Disabled``.
For a stable target, reuse avoids one density/gradient evaluation per transition
following the first. A finite warmup endpoint also transfers to the fixed sampler.
With zero warmup, the fixed sampler evaluates its first endpoint normally.

Opt in only when the target returns the same results for the same position.
Target results must not depend on evaluation counts or handler side effects.
After changing target data between draws, call ``invalidate_endpoint_cache()``
on every affected sampler. A warmup-complete handler that changes target data
must invalidate the adaptive sampler before it returns, or the caller must
invalidate the returned fixed sampler before drawing. Invalidation does not
make samples from a changing target suitable for inference.

Reuse retains only finite density and gradient values. Failed or nonfinite
endpoints are reevaluated. Draw and warmup callbacks retain their order and
frequency; error callbacks report actual thrown evaluations, not cache hits.
Caches are copied by value; RNG, target, and handler references retain their
existing ownership rules. The high-level configuration and Python API do not
enable this option.
