# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compatibility utilities for different versions of transformers library.
"""

import importlib.metadata
import types
from functools import lru_cache
from typing import Optional

from packaging import version

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


def patch_transformers_auto_docstring_uniontype() -> bool:
    """
    Patch transformers auto_docstring for Python 3.10+ ``types.UnionType`` annotations.

    Some transformers versions build parameter type strings using ``annotation.__name__``.
    This fails for ``A | B`` annotations because ``types.UnionType`` does not expose
    ``__name__`` and raises ``AttributeError`` while importing remote model code.
    """
    union_type = getattr(types, "UnionType", None)
    if union_type is None:
        return False

    try:
        from transformers.utils import auto_docstring as auto_docstring_module
    except ImportError:
        return False

    original = getattr(auto_docstring_module, "_process_parameter_type", None)
    if original is None or getattr(original, "_verl_uniontype_patch", False):
        return False

    def _patched_process_parameter_type(param, param_name, func):
        try:
            return original(param, param_name, func)
        except AttributeError as exc:
            annotation = getattr(param, "annotation", None)
            if not (isinstance(annotation, union_type) and "__name__" in str(exc)):
                raise

            param_type = str(annotation).replace("transformers.", "~")
            optional = "None" in param_type or "NoneType" in param_type
            for none_marker in ("NoneType | ", "None | ", " | NoneType", " | None"):
                param_type = param_type.replace(none_marker, "")
            if optional:
                param_type = f"Optional[{param_type}]"
            return param_type, optional

    _patched_process_parameter_type._verl_uniontype_patch = True
    auto_docstring_module._process_parameter_type = _patched_process_parameter_type
    return True
