import os
import sys

import torch.distributed as dist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREAMVLN_ROOT = os.path.join(PROJECT_ROOT, "streamvln")
for path in (PROJECT_ROOT, STREAMVLN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamvln import streamvln_eval as eval_mod
from streamvln_ext.entrypoints.common import apply_ext_args, extract_ext_args
from streamvln_ext.model import StreamVLNForCausalLMExt


def _patch_dist_for_single_process():
    if dist.is_available() and dist.is_initialized():
        return

    def _all_gather(output_tensor_list, tensor, group=None, async_op=False):
        for idx in range(len(output_tensor_list)):
            output_tensor_list[idx].copy_(tensor)
        return None

    def _barrier(*args, **kwargs):
        return None

    eval_mod.dist.all_gather = _all_gather
    eval_mod.dist.barrier = _barrier


def main():
    ext_args, remaining = extract_ext_args(sys.argv[1:])
    apply_ext_args(ext_args)

    eval_mod.StreamVLNForCausalLM = StreamVLNForCausalLMExt
    _patch_dist_for_single_process()

    sys.argv = [sys.argv[0]] + remaining
    eval_mod.eval()


if __name__ == "__main__":
    main()
