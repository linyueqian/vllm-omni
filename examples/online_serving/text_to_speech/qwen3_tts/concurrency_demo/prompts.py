"""Locked prompt used by every stream in the concurrency demo.

A single fixed English string is used for both the c=1 serial reference and
all N parallel streams. This makes serial_eta = N * t_observed an exact
identity rather than an extrapolation across length-varying prompts.
"""

DEMO_PROMPT: str = (
    "Modern text-to-speech models can stream audio in real time. "
    "When many users speak to the same model at once, batching makes the "
    "throughput multiply, not divide. Watch the streams finish together."
)
