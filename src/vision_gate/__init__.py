"""vision-gate: a fail-open change-detection front door for vision models.

Cheap frame differencing decides whether anything visual actually changed,
and where, before any vision-model call is spent. See gate.py for the CLI
and the fail-open rule.
"""
from .gate import compare_frames, endpoint_decide, region_options

__version__ = "1.0.0"
__all__ = ["compare_frames", "endpoint_decide", "region_options", "__version__"]
