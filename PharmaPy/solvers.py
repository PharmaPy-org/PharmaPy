"""Public lazy constructors for PharmaPy's optional Assimulo backend.

Importing this module does not import Assimulo. The optional dependency is
loaded only when one of the exported constructors is called. CVode and IDA
preserve the first model callback exception (including KeyboardInterrupt and
SystemExit), stop retries from bypassing it, and expose it with the native
solver diagnostic as its cause. Normal event termination is unchanged.
"""

from PharmaPy._assimulo import CVode, Explicit_Problem, IDA, Implicit_Problem

__all__ = ["CVode", "IDA", "Explicit_Problem", "Implicit_Problem"]
