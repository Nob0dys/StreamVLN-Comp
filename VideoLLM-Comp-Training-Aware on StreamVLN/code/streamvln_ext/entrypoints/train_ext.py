import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREAMVLN_ROOT = os.path.join(PROJECT_ROOT, "streamvln")
for path in (PROJECT_ROOT, STREAMVLN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamvln import streamvln_train as train_mod
from streamvln_ext.entrypoints.common import apply_ext_args, extract_ext_args
from streamvln_ext.model import StreamVLNForCausalLMExt


def main():
    ext_args, remaining = extract_ext_args(sys.argv[1:])
    apply_ext_args(ext_args)

    train_mod.StreamVLNForCausalLM = StreamVLNForCausalLMExt

    sys.argv = [sys.argv[0]] + remaining
    train_mod.train()


if __name__ == "__main__":
    main()
