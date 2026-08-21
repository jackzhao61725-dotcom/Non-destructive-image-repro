"""Load project modules without executing the package initializer.

The selected module and its relative imports are resolved beneath a synthetic
private package.  This keeps task-scoped tools independent of the eager imports
in ``non_destructive_image.__init__`` while preserving normal relative-import
semantics for the scientific module chain.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import re
import sys
import types
from pathlib import Path
from types import ModuleType


DEFAULT_PACKAGE_DIRECTORY = Path(__file__).resolve().parent / "non_destructive_image"


def load_isolated_non_destructive_image_module(
    module_name: str,
    *,
    namespace: str,
    package_directory: Path = DEFAULT_PACKAGE_DIRECTORY,
) -> ModuleType:
    """Import one project module beneath a private synthetic package."""

    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name) is None:
        raise ValueError("isolated module name is invalid")
    if (
        re.fullmatch(r"_[A-Za-z_]\w*", namespace) is None
        or namespace == "non_destructive_image"
        or namespace.startswith("non_destructive_image.")
    ):
        raise ValueError("isolated namespace is invalid")
    package_directory = package_directory.resolve()
    if not package_directory.is_dir():
        raise FileNotFoundError(package_directory)

    absent = object()
    canonical_before = sys.modules.get("non_destructive_image", absent)
    private_package = sys.modules.get(namespace)
    if private_package is None:
        private_package = types.ModuleType(namespace)
        private_package.__package__ = namespace
        private_package.__path__ = [str(package_directory)]
        specification = importlib.machinery.ModuleSpec(
            namespace, loader=None, is_package=True
        )
        specification.submodule_search_locations = [str(package_directory)]
        private_package.__spec__ = specification
        sys.modules[namespace] = private_package
    elif list(getattr(private_package, "__path__", ())) != [str(package_directory)]:
        raise RuntimeError("isolated namespace is already bound to another package")

    module = importlib.import_module(f"{namespace}.{module_name}")
    canonical_after = sys.modules.get("non_destructive_image", absent)
    if canonical_after is not canonical_before:
        raise RuntimeError("isolated import modified the canonical package module")
    source = Path(str(module.__file__)).resolve()
    try:
        source.relative_to(package_directory)
    except ValueError as error:
        raise RuntimeError("isolated module resolved outside the project package") from error
    package_name = str(module.__package__)
    if package_name != namespace and not package_name.startswith(f"{namespace}."):
        raise RuntimeError("isolated module escaped its private package namespace")
    return module


__all__ = ["load_isolated_non_destructive_image_module"]
