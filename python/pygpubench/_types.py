import dataclasses

from typing import Any, Callable, Tuple

Tensor = "torch.Tensor"
ExpectedSpec = Tensor | Tuple[Tensor] | Tuple[Tensor, float, float]
ExpectedResult = Tuple[ExpectedSpec, ...]
BenchmarkCase = Tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class OutputArg:
    value: Any
    expected: ExpectedSpec
    uses_current_value: bool = False

KernelFunction = Callable[..., None]
TestGeneratorInterface = Callable[..., BenchmarkCase]

__all__ = [
    "BenchmarkCase",
    "ExpectedResult",
    "ExpectedSpec",
    "KernelFunction",
    "OutputArg",
    "TestGeneratorInterface",
]
