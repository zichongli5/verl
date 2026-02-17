#!/usr/bin/env python3
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
Swap parquet prompt columns:
- "prompt" -> "raw_prompt"
- "source_prompt" -> "prompt"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swap prompt-related keys in a parquet file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file path. Supports Unix paths and Windows paths (e.g. C:\\\\... ).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet file path. If omitted, writes '<input_stem>.prompt_swapped.parquet'.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file in place (writes to a temp file and then replaces).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing --output file.",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec for output, e.g. snappy, zstd, gzip, none.",
    )
    return parser.parse_args()


def normalize_path(path_str: str) -> Path:
    expanded = os.path.expanduser(path_str)
    windows_path = re.match(r"^([A-Za-z]):[\\/](.*)$", expanded)
    if windows_path and os.name != "nt":
        drive = windows_path.group(1).lower()
        rest = windows_path.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(expanded)


def get_output_path(input_path: Path, output_arg: str | None, inplace: bool) -> Path:
    if inplace and output_arg is not None:
        raise ValueError("Do not pass --output together with --inplace.")
    if inplace:
        return input_path
    if output_arg:
        return normalize_path(output_arg)
    return input_path.with_name(f"{input_path.stem}.prompt_swapped{input_path.suffix}")


def get_mode(column_names: list[str]) -> str:
    has_prompt = "prompt" in column_names
    has_source_prompt = "source_prompt" in column_names
    has_raw_prompt = "raw_prompt" in column_names

    if has_prompt and has_source_prompt and not has_raw_prompt:
        return "swap"
    if has_prompt and has_raw_prompt and not has_source_prompt:
        return "already_done"
    if has_prompt and has_source_prompt and has_raw_prompt:
        raise ValueError(
            "Column 'raw_prompt' already exists, so renaming 'prompt' to 'raw_prompt' would conflict."
        )

    raise ValueError(
        "Expected columns 'prompt' and 'source_prompt'. "
        f"Current columns: {column_names}"
    )


def swap_columns(column_names: list[str]) -> list[str]:
    renamed = []
    for col in column_names:
        if col == "prompt":
            renamed.append("raw_prompt")
        elif col == "source_prompt":
            renamed.append("prompt")
        else:
            renamed.append(col)
    return renamed


def main() -> int:
    args = parse_args()
    input_path = normalize_path(args.input)
    output_path = get_output_path(input_path=input_path, output_arg=args.output, inplace=args.inplace)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and output_path != input_path and not args.overwrite:
        print(f"Output exists: {output_path}. Use --overwrite to replace it.", file=sys.stderr)
        return 1

    compression = None if str(args.compression).lower() == "none" else args.compression

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        print("Missing dependency: pyarrow. Install it with `pip install pyarrow`.", file=sys.stderr)
        return 1

    table = pq.read_table(input_path)
    mode = get_mode(table.column_names)

    if mode == "already_done":
        print("No change needed: file already has 'raw_prompt' and no 'source_prompt'.")
        return 0

    renamed_table = table.rename_columns(swap_columns(table.column_names))

    if output_path == input_path:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(renamed_table, tmp_path, compression=compression)
        tmp_path.replace(output_path)
    else:
        pq.write_table(renamed_table, output_path, compression=compression)

    print(f"Done. Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
