Measurement error regression

24 Apr 2026

Data are simulated.

Background on the problem is at https://statmodeling.stat.columbia.edu/2025/05/02/measurement-error-model-stan-fitting-struggle-the-funnel-again-rears-its-ugly-head/

measurement_error.stan:  Stan program implementing a latent-data measurement-error regression

measurement_error.R:  R script that simulates the data and fits the model.  We fit the same model to two different simulated datasets.  In the first simulation, the standard deviation of the measurement error is 1.  In the second simulation, the standard deviation of the measurement error is 5.  In both cases, the measurement error is assumed known; that is, the true value of the measurement error is passed as data to Stan.  For the first simulation, the posterior is well behaved, and HMC mixes well.  For the second simulation, there'a funnel problem with the latent data, and HMC does not mix well.
