import argparse
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


def _extract_quick_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_episodes", type=int, default=0)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def _patch_max_episodes(max_episodes: int):
    if max_episodes <= 0:
        return

    original_config_env = eval_mod.VLNEvaluator.config_env

    def _config_env_with_limit(self):
        env = original_config_env(self)
        if hasattr(env, "episodes") and len(env.episodes) > max_episodes:
            env.episodes = env.episodes[:max_episodes]
        return env

    eval_mod.VLNEvaluator.config_env = _config_env_with_limit


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
    ext_args, stage1_remaining = extract_ext_args(sys.argv[1:])
    quick_args, remaining = _extract_quick_args(stage1_remaining)
    apply_ext_args(ext_args)

    eval_mod.StreamVLNForCausalLM = StreamVLNForCausalLMExt
    _patch_max_episodes(quick_args.max_episodes)
    _patch_dist_for_single_process()

    sys.argv = [sys.argv[0]] + remaining
    eval_mod.eval()


if __name__ == "__main__":
    main()
