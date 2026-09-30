"""Import this before any torch import in this package.

Works around a known torch/sympy incompatibility seen on some Colab images:
torch.utils._sympy references sympy.printing.StrPrinter as a type
annotation, but `import sympy` alone doesn't always pull in the `printing`
submodule first, raising
"AttributeError: module 'sympy' has no attribute 'printing'" the moment
torch/torchvision is first imported.
"""

try:
    import sympy.printing  # noqa: F401
except ImportError:
    pass
