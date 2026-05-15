import argparse
import os
from typing import List, Tuple


class ExtEntrypointArgs(argparse.Namespace):
    ext_flags_json: str
    ext_flags_file: str


def extract_ext_args(argv: List[str]) -> Tuple[ExtEntrypointArgs, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ext_flags_json", type=str, default="")
    parser.add_argument("--ext_flags_file", type=str, default="")
    args, remaining = parser.parse_known_args(argv, namespace=ExtEntrypointArgs())
    return args, remaining


def apply_ext_args(args: ExtEntrypointArgs):
    if getattr(args, "ext_flags_json", ""):
        os.environ["STREAMVLN_EXT_FLAGS"] = args.ext_flags_json
    if getattr(args, "ext_flags_file", ""):
        os.environ["STREAMVLN_EXT_FLAGS_FILE"] = args.ext_flags_file
