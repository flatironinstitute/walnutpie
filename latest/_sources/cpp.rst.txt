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
