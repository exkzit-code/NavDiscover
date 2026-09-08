# Historical code with known numerical errors

These files preserve the previous implementation and manuscript package unchanged.
Their singularity, self-similarity and special-separation claims are unsupported.
They are not the active solver and their output must not be used as physical evidence.

The original source is also preserved in Git at commit
`bff8d412d373338b655f4d31e0e3217eb0feb812`.

Known failures include a pressure correction that amplifies divergence, a separate
wrong-sign Kida projection, finite-threshold extrapolation called a blowup time,
resolution-dependent FFT/filter handling, and a growing profile difference
labelled as a decay rate. See [the correction note](../../docs/CORRECTION.md).

The archived scripts use their historical paths and extra dependencies. The new
top-level requirements intentionally support only the corrected active workflow.
