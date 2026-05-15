import argparse
import copy
import json
import math
import os
import re
import sys
import time
import traceback
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import transformers
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from llava import conversation as conversation_lib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREAMVLN_ROOT = os.path.join(PROJECT_ROOT, "streamvln")
for path in (PROJECT_ROOT, STREAMVLN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamvln.dataset.vln_action_dataset import preprocess  # noqa: E402
from streamvln.utils.utils import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
)
from streamvln_ext.entrypoints.common import apply_ext_args, extract_ext_args  # noqa: E402
from streamvln_ext.model import StreamVLNForCausalLMExt  # noqa: E402


PROTOCOL_LEGACY_TF_FIRST_FRAME = "legacy_tf_first_frame"
PROTOCOL_AURORA_REPLAY_GT = "aurora_replay_gt"
AURORA_PROTOCOL_NAME = "AuroraReplay-GT"


def _mean(values: List[float]) -> float:
    """计算数值列表的均值；空列表返回 0.0，避免汇总阶段除零。"""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: List[float], q: float) -> float:
    """计算数值列表的百分位数，常用于延迟 p50/p95；空列表返回 0.0。"""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _load_rgb_or_blank(path: str, fallback_size=(256, 256)) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return Image.new("RGB", fallback_size, (0, 0, 0))


def _estimate_tflops_per_step(model, total_tokens: int) -> float:
    """粗略估算单步 Transformer 解码计算量。

    该函数只基于模型配置中的 hidden_size、num_hidden_layers 和当前 token
    数估算 TFLOPs，用于不同压缩/剪枝方案之间的相对比较，不等价于精确 profiler。
    """
    if total_tokens <= 0:
        return 0.0

    cfg = getattr(model, "config", None)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    if hidden <= 0 or layers <= 0:
        return 0.0

    t = float(total_tokens)
    h = float(hidden)
    l = float(layers)

    # 近似 causal decoder 每层注意力与 MLP 的主要 FLOPs。
    flops = l * (8.0 * t * h * h + 4.0 * t * t * h)
    return float(flops / 1e12)


def _get_gpu_peak_stats(device: torch.device) -> Dict[str, float]:
    """读取评估期间的 CUDA 峰值显存统计；CPU 环境下返回 0 和设备名 cpu。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_peak_allocated_mib": 0.0,
            "gpu_peak_reserved_mib": 0.0,
            "gpu_device_name": "cpu",
        }

    allocated = float(torch.cuda.max_memory_allocated(device=device) / (1024**2))
    reserved = float(torch.cuda.max_memory_reserved(device=device) / (1024**2))
    name = torch.cuda.get_device_name(device)
    return {
        "gpu_peak_allocated_mib": allocated,
        "gpu_peak_reserved_mib": reserved,
        "gpu_device_name": name,
    }


class OfflineTeacherForcingActionAccuracyEvaluator:
    """离线动作准确率评估器。

    该类负责读取离线 VLN 轨迹数据、构造模型输入、执行两种评估协议，并把逐条预测结果与
    汇总指标写入输出目录。动作空间固定为 STOP、前进、左转、右转四类。
    """

    def __init__(self, model, tokenizer, args):
        """保存模型、分词器和命令行参数，并初始化动作映射与运行时统计缓存。"""
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = model.get_vision_tower().image_processor

        self.idx2actions = {
            0: "STOP",
            1: "↑",
            2: "←",
            3: "→",
        }
        self.actions2idx = OrderedDict(
            {
                "STOP": [0],
                "↑": [1],
                "←": [2],
                "→": [3],
            }
        )
        self.step_latency_ms: List[float] = []
        self.step_total_tokens: List[int] = []
        self.step_visual_tokens: List[int] = []
        self.step_memory_tokens: List[int] = []
        self.step_tflops: List[float] = []
        self.eval_protocol = args.eval_protocol
        self.aurora_decode_max_new_tokens = args.aurora_decode_max_new_tokens
        self.aurora_batch_size = max(int(args.aurora_batch_size), 1)
        self.aurora_step_mode = args.aurora_step_mode
        self.aurora_precompute_vision = bool(args.aurora_precompute_vision)
        self.aurora_vision_batch_size = max(int(args.aurora_vision_batch_size), 1)
        self.save_step_debug = bool(args.save_step_debug)
        self.debug_max_steps_per_episode = max(int(args.debug_max_steps_per_episode), 0)
        self._current_step_debug: List[Dict] = []
        self.prompt_tokenizer = copy.deepcopy(tokenizer)
        self.prompt_tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        self.prompt_tokenizer.add_tokens([DEFAULT_MEMORY_TOKEN], special_tokens=True)
        self.prompt_image_token_id = self.prompt_tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.prompt_memory_token_id = self.prompt_tokenizer.convert_tokens_to_ids(DEFAULT_MEMORY_TOKEN)
        self.prompt_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.invalid_prediction_count = 0

    def parse_actions(self, text: str) -> List[int]:
        """从模型输出文本中抽取动作 token，并映射为动作编号列表。

        支持大小写 STOP 以及三个方向符号：↑=1、←=2、→=3；无法识别的文本会被忽略。
        """
        regex = re.compile(r"STOP|Stop|stop|↑|←|→")
        matches = regex.findall(text)
        out = []
        for token in matches:
            if token in {"STOP", "Stop", "stop"}:
                out.append(0)
            elif token == "↑":
                out.append(1)
            elif token == "←":
                out.append(2)
            elif token == "→":
                out.append(3)
        return out

    def normalize_actions(self, raw_actions: List[int]) -> List[int]:
        """清洗数据集中的原始动作序列。

        只保留 0/1/2/3 四类合法动作；若首个动作为 -1 则视为占位并丢弃；根据参数决定
        是否在缺少 STOP 时自动补齐终止动作。
        """
        actions = [int(a) for a in raw_actions if isinstance(a, int)]
        if len(actions) > 0 and actions[0] == -1:
            actions = actions[1:]
        actions = [a for a in actions if a in {0, 1, 2, 3}]
        if self.args.append_stop_if_missing and (len(actions) == 0 or actions[-1] != 0):
            actions.append(0)
        return actions

    def actions2text(self, actions: List[int]) -> str:
        """把动作编号序列转换为训练/评估 prompt 中使用的动作字符串。"""
        return "".join(self.idx2actions[int(a)] for a in actions)

    def load_items(self):
        """读取 annotations.json，并展开为逐 instruction 的评估样本。

        一个轨迹可能包含多条自然语言指令；本函数会把每条 instruction 拆成独立 record，
        同时附带轨迹 id、视频目录和清洗后的动作序列。
        """
        anno_path = os.path.join(self.args.dataset_root, "annotations.json")
        with open(anno_path, "r", encoding="utf-8") as f:
            annos = json.load(f)

        records = []
        for item in annos:
            instructions = item.get("instructions", item.get("instruction", None))
            if instructions is None:
                continue
            if not isinstance(instructions, list):
                instructions = [instructions]

            actions = self.normalize_actions(item.get("actions", []))
            if len(actions) == 0:
                continue

            for ins_idx, ins in enumerate(instructions):
                records.append(
                    {
                        "id": item.get("id", len(records)),
                        "ins_idx": ins_idx,
                        "instruction": ins,
                        "video": item.get("video", ""),
                        "actions": actions,
                    }
                )

        return records

    def build_prompt(self, instruction: str) -> str:
        """根据导航指令构造模型输入 prompt。

        prompt 会说明可用动作集合，并追加图像占位符 DEFAULT_IMAGE_TOKEN，让 preprocess
        后续把视觉特征插入到对应位置。
        """
        return (
            "You are an autonomous navigation assistant. "
            f"Your task is to {instruction}. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP. "
            f"{DEFAULT_IMAGE_TOKEN}."
        )

    def build_dummy_geom(self, num_views: int = 1):
        """构造占位几何输入。

        StreamVLN 前向接口需要 depth、pose、intrinsic；离线动作准确率评估只关注 RGB
        和文本动作，因此这里用零深度图、单位位姿矩阵和单位内参矩阵填充接口。
        """
        crop_h = self.image_processor.crop_size["height"]
        crop_w = self.image_processor.crop_size["width"]
        depth = torch.zeros((1, num_views, crop_h, crop_w), dtype=torch.float32, device=self.device)
        pose = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        intrinsic = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        pose = pose.repeat(1, num_views, 1, 1)
        intrinsic = intrinsic.repeat(1, num_views, 1, 1)
        return depth, pose, intrinsic

    def _offline_geom_mode(self) -> str:
        flags = getattr(self.model, "ext_flags", None)
        return str(getattr(flags, "voxel_spatial_offline_geom_mode", "dummy")).strip().lower()

    @staticmethod
    def _to_intrinsic_4x4(intrinsic: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape == (4, 4):
            return intrinsic
        if intrinsic.shape == (3, 3):
            out = np.eye(4, dtype=np.float32)
            out[:3, :3] = intrinsic
            return out
        raise ValueError(f"Unexpected intrinsic shape: {intrinsic.shape}")

    def _load_saved_frame_geom(self, episode_dir: str, frame_name: str):
        stem = os.path.splitext(frame_name)[0]
        depth_path = os.path.join(episode_dir, "depth", f"{stem}.png")
        pose_path = os.path.join(episode_dir, "pose", f"{stem}.npy")
        intrinsic_path = os.path.join(episode_dir, "intrinsic", f"{stem}.npy")
        if not (os.path.isfile(depth_path) and os.path.isfile(pose_path) and os.path.isfile(intrinsic_path)):
            return None

        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        pose = np.load(pose_path).astype(np.float32)
        intrinsic = self._to_intrinsic_4x4(np.load(intrinsic_path))
        return (
            torch.from_numpy(depth).float().to(self.device),
            torch.from_numpy(pose).float().to(self.device),
            torch.from_numpy(intrinsic).float().to(self.device),
        )

    def _build_pinhole_intrinsic(self, width: int, height: int) -> torch.Tensor:
        flags = getattr(self.model, "ext_flags", None)
        hfov_deg = float(getattr(flags, "voxel_spatial_offline_hfov_deg", 79.0))
        hfov_rad = math.radians(max(hfov_deg, 1e-3))
        fx = (float(width) / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        cx = (float(width) - 1.0) / 2.0
        cy = (float(height) - 1.0) / 2.0
        intrinsic = torch.eye(4, dtype=torch.float32, device=self.device)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        return intrinsic

    def _odometry_pose_for_frame(self, frame_idx: int, actions: List[int]) -> torch.Tensor:
        x = 0.0
        z = 0.0
        yaw = 0.0
        turn = math.radians(15.0)
        step = 0.25

        for action in actions[: max(int(frame_idx), 0)]:
            action = int(action)
            if action == 1:
                x += math.sin(yaw) * step
                z += math.cos(yaw) * step
            elif action == 2:
                yaw += turn
            elif action == 3:
                yaw -= turn

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        pose = torch.eye(4, dtype=torch.float32, device=self.device)
        pose[0, 0] = cos_y
        pose[0, 2] = sin_y
        pose[2, 0] = -sin_y
        pose[2, 2] = cos_y
        pose[0, 3] = x
        pose[2, 3] = z
        return pose

    def _build_offline_geom(
        self,
        selected_indices: List[int],
        actions: List[int],
        video_dir: str = None,
        frame_files: List[str] = None,
    ):
        flags = getattr(self.model, "ext_flags", None)
        use_saved_geometry = bool(getattr(flags, "enable_offline_saved_geometry", True))
        if use_saved_geometry and video_dir is not None and frame_files is not None:
            episode_dir = os.path.dirname(video_dir)
            loaded = []
            for idx in selected_indices:
                if idx >= len(frame_files):
                    loaded = []
                    break
                item = self._load_saved_frame_geom(episode_dir, frame_files[idx])
                if item is None:
                    loaded = []
                    break
                loaded.append(item)
            if loaded:
                depths, poses, intrinsics = zip(*loaded)
                return (
                    torch.stack(list(depths), dim=0).unsqueeze(0),
                    torch.stack(list(poses), dim=0).unsqueeze(0),
                    torch.stack(list(intrinsics), dim=0).unsqueeze(0),
                )

        mode = self._offline_geom_mode()
        if mode not in {"odometry", "odometry_unit_depth", "synthetic_odometry"}:
            return self.build_dummy_geom(num_views=len(selected_indices))

        crop_h = self.image_processor.crop_size["height"]
        crop_w = self.image_processor.crop_size["width"]
        flags = getattr(self.model, "ext_flags", None)
        unit_depth = float(getattr(flags, "voxel_spatial_offline_unit_depth_m", 2.0))
        depth = torch.full(
            (1, len(selected_indices), crop_h, crop_w),
            fill_value=max(unit_depth, 1e-3),
            dtype=torch.float32,
            device=self.device,
        )
        poses = torch.stack(
            [self._odometry_pose_for_frame(idx, actions) for idx in selected_indices],
            dim=0,
        ).unsqueeze(0)
        intrinsic = self._build_pinhole_intrinsic(crop_w, crop_h)
        intrinsics = intrinsic.unsqueeze(0).unsqueeze(0).repeat(1, len(selected_indices), 1, 1)
        return depth, poses, intrinsics

    def _select_history_indices(self, step_idx: int) -> List[int]:
        """为当前 step 选择历史帧下标。

        当 num_history 为空时使用全部历史帧；当历史帧数量超过上限时，使用 linspace 在
        [0, step_idx - 1] 范围内均匀采样，保证既覆盖早期观察也覆盖近期观察。
        """
        if step_idx <= 0:
            return []

        history_cap = self.args.num_history
        if history_cap is None:
            return list(range(step_idx))

        history_cap = max(int(history_cap), 0)
        if history_cap == 0:
            return []
        if step_idx <= history_cap:
            return list(range(step_idx))

        sampled = np.linspace(0, step_idx - 1, num=history_cap, dtype=np.int32)
        return [int(v) for v in sampled.tolist()]

    def _build_visual_inputs(self, video_dir: str, frame_files: List[str], step_idx: int, actions: List[int] = None):
        """读取当前 step 的视觉输入，并返回图像张量与占位几何信息。

        返回的图像维度为 [batch=1, views, channels, height, width]；views 由历史帧
        加当前帧组成，最后一个返回值 history_count 表示其中有多少帧是历史观察。
        """
        frame_idx = min(step_idx, len(frame_files) - 1)
        history_indices = self._select_history_indices(frame_idx)
        selected_indices = history_indices + [frame_idx]

        images = []
        for idx in selected_indices:
            frame_path = os.path.join(video_dir, frame_files[idx])
            image = _load_rgb_or_blank(frame_path)
            image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
            images.append(image_tensor)

        image_tensor = torch.stack(images, dim=0).unsqueeze(0).to(self.device)
        depth, pose, intrinsic = self._build_offline_geom(
            selected_indices,
            actions or [],
            video_dir=video_dir,
            frame_files=frame_files,
        )
        return image_tensor, depth, pose, intrinsic, len(history_indices)

    def _build_prompt_for_step(self, instruction: str, history_count: int) -> str:
        """为 AuroraReplay-GT 的单步预测构造 prompt。

        如果当前输入包含历史帧，则额外加入 DEFAULT_MEMORY_TOKEN，让模型知道本轮存在历史
        观察记忆；没有历史帧时只使用基础导航 prompt。
        """
        prompt = self.build_prompt(instruction)
        if history_count > 0:
            prompt += f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
        return prompt

    def _replace_multimodal_token_ids(self, input_ids: List[int]) -> torch.Tensor:
        """把 tokenizer 中的图像/记忆 token id 替换为 StreamVLN 使用的负数占位符。"""
        out = []
        for token_id in input_ids:
            if token_id == self.prompt_image_token_id:
                out.append(IMAGE_TOKEN_INDEX)
            elif token_id == self.prompt_memory_token_id:
                out.append(MEMORY_TOKEN_INDEX)
            else:
                out.append(int(token_id))
        return torch.tensor(out, dtype=torch.long)

    def _build_generation_input_ids(self, instruction: str, history_count: int, gt_prefix_text: str) -> torch.Tensor:
        """构造用于逐步生成/next-token 打分的 ChatML prompt。

        训练用 preprocess() 会把 assistant 消息写成
        ``<|im_start|>assistant\n{content}<|im_end|>\n``，这适合 teacher-forcing loss，
        但不适合“给定动作前缀后继续生成下一步动作”：如果输入以 ``<|im_end|>`` 结尾，
        模型会在一个已经结束的 assistant turn 后续写，常导致不可解析输出。这里显式让
        输入停在 ``<|im_start|>assistant\n{gt_prefix}``，即 assistant 内容尚未结束的位置。
        """
        prompt = self._build_prompt_for_step(instruction, history_count)
        input_ids: List[int] = []
        input_ids += self.prompt_tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."}]
        )
        input_ids += self.prompt_tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
        input_ids += self.prompt_tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        ).input_ids
        if gt_prefix_text:
            input_ids += self.prompt_tokenizer(gt_prefix_text, add_special_tokens=False).input_ids
        return self._replace_multimodal_token_ids(input_ids)

    def _record_step_debug(
        self,
        step_idx: int,
        history_count: int,
        gt_prefix_text: str,
        generated_text: str,
        parsed_action: int,
    ) -> None:
        if not self.save_step_debug:
            return
        if len(self._current_step_debug) >= self.debug_max_steps_per_episode:
            return
        self._current_step_debug.append(
            {
                "step_idx": int(step_idx),
                "history_count": int(history_count),
                "gt_prefix_text": gt_prefix_text,
                "generated_text": generated_text,
                "parsed_action": int(parsed_action),
            }
        )

    def _first_parsed_action_or_invalid(self, text: str) -> int:
        """返回输出中的首个动作；无法解析时记为无效预测，而不是默认 STOP。"""
        parsed_actions = self.parse_actions(text.strip())
        if len(parsed_actions) == 0:
            self.invalid_prediction_count += 1
            return -1
        return int(parsed_actions[0])

    def _predict_actions_aurora_replay_gt(self, item: Dict, video_dir: str, frame_files: List[str]) -> List[int]:
        """按 AuroraReplay-GT 协议逐步预测动作。

        每个 step 都使用 ground-truth 动作前缀作为已执行历史，只让模型生成下一步动作；
        视觉输入由当前帧和采样历史帧组成。该协议更接近在线 rollout 的逐步决策方式，但
        仍通过 GT 前缀避免早期错误持续扩散。
        """
        if (
            self.aurora_batch_size > 1
            or self.aurora_step_mode == "next_token_logits"
            or self.aurora_precompute_vision
        ):
            return self._predict_actions_aurora_replay_gt_batched(item, video_dir, frame_files)

        gt_actions = item["actions"]
        pred_actions: List[int] = []
        decode_tokens = max(int(self.aurora_decode_max_new_tokens), 1)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        base_num_history = int(getattr(self.model.model, "num_history", 0) or 0)
        self._current_step_debug = []

        for step_idx in range(len(gt_actions)):
            self.model.reset_for_env(0)
            images, depth, pose, intrinsic, history_count = self._build_visual_inputs(
                video_dir, frame_files, step_idx, item["actions"]
            )

            # Teacher forcing 前缀：当前 step 之前的动作来自标注，不使用模型历史预测。
            gt_prefix_text = self.actions2text(gt_actions[:step_idx])
            input_ids = self._build_generation_input_ids(
                item["instruction"],
                history_count,
                gt_prefix_text,
            ).unsqueeze(0).to(self.device)

            if history_count > 0:
                self.model.model.num_history = history_count
            else:
                self.model.model.num_history = base_num_history

            # generate() 只生成下一步动作；env/time/task 字段保持与 StreamVLN 模型接口一致。
            input_dict = {
                "images": images.to(dtype),
                "depths": depth.to(dtype),
                "poses": pose.to(dtype),
                "intrinsics": intrinsic.to(dtype),
                "inputs": input_ids,
                "env_id": 0,
                "time_ids": [[step_idx]],
                "task_type": [0],
            }

            step_start = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **input_dict,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=decode_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            step_latency_ms = (time.perf_counter() - step_start) * 1000.0
            self.step_latency_ms.append(float(step_latency_ms))

            total_tokens = int(input_ids.shape[1])
            if hasattr(self.model, "cache"):
                try:
                    cache_slot = self.model.cache[0]
                    if isinstance(cache_slot, dict) and "inputs_embeds" in cache_slot:
                        total_tokens = int(cache_slot["inputs_embeds"].shape[1])
                except Exception:
                    pass

            visual_tokens = int(input_dict["images"].shape[1])
            memory_tokens = int(max(0, visual_tokens - 1))
            self.step_visual_tokens.append(visual_tokens)
            self.step_memory_tokens.append(memory_tokens)
            self.step_total_tokens.append(total_tokens)
            self.step_tflops.append(_estimate_tflops_per_step(self.model, total_tokens))

            generated_ids = outputs.sequences[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
            pred_action = self._first_parsed_action_or_invalid(generated_text)
            self._record_step_debug(step_idx, history_count, gt_prefix_text, generated_text, pred_action)
            pred_actions.append(pred_action)

        self.model.model.num_history = base_num_history
        return pred_actions

    def _precompute_projected_frame_features(self, video_dir: str, frame_files: List[str], include_raw: bool = False):
        """Encode each RGB frame once for the current episode and cache visual tokens.

        Training-aware compressors still need raw vision-tower tokens and, for LongVU,
        the preprocessed current image.  When include_raw is true we return those
        alongside the projected pooled features used by the normal precompute path.
        """
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        projected_batches = []
        raw_batches = []
        image_batches = []

        for offset in range(0, len(frame_files), self.aurora_vision_batch_size):
            chunk_files = frame_files[offset : offset + self.aurora_vision_batch_size]
            images = []
            for frame_name in chunk_files:
                frame_path = os.path.join(video_dir, frame_name)
                image = _load_rgb_or_blank(frame_path)
                image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
                images.append(image_tensor)

            image_batch = torch.stack(images, dim=0).to(self.device)
            with torch.inference_mode():
                raw_features = self.model.get_model().get_vision_tower()(image_batch.to(dtype))
                projected = self.model.get_model().mm_projector(raw_features)
                projected = self.model.get_2dPool(projected, 2)
            projected_batches.append(projected.detach())
            if include_raw:
                raw_batches.append(raw_features.detach())
                image_batches.append(image_batch.detach())

        projected_features = torch.cat(projected_batches, dim=0)
        if not include_raw:
            return projected_features
        return {
            "projected": projected_features,
            "raw": torch.cat(raw_batches, dim=0),
            "images": torch.cat(image_batches, dim=0),
        }

    def _predict_actions_aurora_replay_gt_batched(
        self,
        item: Dict,
        video_dir: str,
        frame_files: List[str],
    ) -> List[int]:
        """批量版 AuroraReplay-GT。

        每个 step 仍只使用该 step 的 GT 动作前缀；批处理只合并 history_count 相同的独立
        step，避免改变视觉历史切分语义。
        """
        gt_actions = item["actions"]
        pred_actions: List[int] = [-1 for _ in gt_actions]
        decode_tokens = max(int(self.aurora_decode_max_new_tokens), 1)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        base_num_history = int(getattr(self.model.model, "num_history", 0) or 0)
        projected_frame_features = None
        raw_frame_features = None
        precomputed_frame_images = None
        self._current_step_debug = []
        if self.aurora_precompute_vision:
            include_raw = bool(getattr(getattr(self.model, "ext_flags", None), "enable_training_aware_video_compressor", False))
            precomputed = self._precompute_projected_frame_features(video_dir, frame_files, include_raw=include_raw)
            if include_raw:
                projected_frame_features = precomputed["projected"]
                raw_frame_features = precomputed["raw"]
                precomputed_frame_images = precomputed["images"]
            else:
                projected_frame_features = precomputed

        steps_by_history: Dict[int, List[int]] = {}
        for step_idx in range(len(gt_actions)):
            frame_idx = min(step_idx, len(frame_files) - 1)
            history_count = len(self._select_history_indices(frame_idx))
            steps_by_history.setdefault(history_count, []).append(step_idx)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        for history_count in sorted(steps_by_history):
            if history_count > 0:
                self.model.model.num_history = history_count
            else:
                self.model.model.num_history = base_num_history

            step_indices = steps_by_history[history_count]
            for offset in range(0, len(step_indices), self.aurora_batch_size):
                chunk = step_indices[offset : offset + self.aurora_batch_size]
                image_tensors = []
                depth_tensors = []
                pose_tensors = []
                intrinsic_tensors = []
                input_id_tensors = []
                precomputed_image_features = []
                precomputed_memory_features = []
                precomputed_raw_image_features = []
                precomputed_raw_memory_features = []
                precomputed_current_images = []

                for step_idx in chunk:
                    if projected_frame_features is None:
                        images, depth, pose, intrinsic, cur_history_count = self._build_visual_inputs(
                            video_dir,
                            frame_files,
                            step_idx,
                            item["actions"],
                        )
                    else:
                        frame_idx = min(step_idx, len(frame_files) - 1)
                        history_indices = self._select_history_indices(frame_idx)
                        cur_history_count = len(history_indices)
                        selected_indices = history_indices + [frame_idx]
                        current_feature = projected_frame_features[frame_idx : frame_idx + 1]
                        if history_indices:
                            history_feature = projected_frame_features.index_select(
                                0,
                                torch.tensor(history_indices, dtype=torch.long, device=projected_frame_features.device),
                            )
                            memory_feature = history_feature.flatten(0, 1).unsqueeze(0)
                        else:
                            memory_feature = None
                        precomputed_image_features.append(current_feature)
                        precomputed_memory_features.append(memory_feature)
                        if raw_frame_features is not None:
                            current_raw = raw_frame_features[frame_idx : frame_idx + 1]
                            if history_indices:
                                history_raw = raw_frame_features.index_select(
                                    0,
                                    torch.tensor(history_indices, dtype=torch.long, device=raw_frame_features.device),
                                )
                                raw_memory_feature = history_raw.flatten(0, 1).unsqueeze(0)
                            else:
                                raw_memory_feature = None
                            precomputed_raw_image_features.append(current_raw)
                            precomputed_raw_memory_features.append(raw_memory_feature)
                            precomputed_current_images.append(precomputed_frame_images[frame_idx : frame_idx + 1])
                        depth, pose, intrinsic = self._build_offline_geom(
                            selected_indices,
                            item["actions"],
                            video_dir=video_dir,
                            frame_files=frame_files,
                        )

                    if cur_history_count != history_count:
                        raise RuntimeError(
                            f"Unexpected history_count mismatch: grouped={history_count}, built={cur_history_count}"
                        )

                    gt_prefix_text = self.actions2text(gt_actions[:step_idx])
                    step_input_ids = self._build_generation_input_ids(
                        item["instruction"],
                        history_count,
                        gt_prefix_text,
                    )

                    if projected_frame_features is None:
                        image_tensors.append(images.squeeze(0))
                    depth_tensors.append(depth.squeeze(0))
                    pose_tensors.append(pose.squeeze(0))
                    intrinsic_tensors.append(intrinsic.squeeze(0))
                    input_id_tensors.append(step_input_ids)

                max_input_len = max(int(ids.shape[0]) for ids in input_id_tensors)
                input_ids = torch.full(
                    (len(chunk), max_input_len),
                    int(pad_token_id),
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros(
                    (len(chunk), max_input_len),
                    dtype=torch.long,
                    device=self.device,
                )
                for row_idx, ids in enumerate(input_id_tensors):
                    ids = ids.to(self.device)
                    cur_len = int(ids.shape[0])
                    input_ids[row_idx, :cur_len] = ids
                    attention_mask[row_idx, :cur_len] = 1

                if projected_frame_features is None:
                    images = torch.stack(image_tensors, dim=0).to(self.device)
                    depths = torch.stack(depth_tensors, dim=0).to(self.device)
                    poses = torch.stack(pose_tensors, dim=0).to(self.device)
                    intrinsics = torch.stack(intrinsic_tensors, dim=0).to(self.device)
                else:
                    crop_h = self.image_processor.crop_size["height"]
                    crop_w = self.image_processor.crop_size["width"]
                    images = torch.empty((len(chunk), 1, 3, crop_h, crop_w), dtype=torch.float32, device=self.device)
                    depths = torch.stack(depth_tensors, dim=0).to(self.device)
                    poses = torch.stack(pose_tensors, dim=0).to(self.device)
                    intrinsics = torch.stack(intrinsic_tensors, dim=0).to(self.device)
                time_ids = [[int(step_idx)] for step_idx in chunk]

                input_dict = {
                    "images": images.to(dtype),
                    "depths": depths.to(dtype),
                    "poses": poses.to(dtype),
                    "intrinsics": intrinsics.to(dtype),
                    "inputs": input_ids,
                    "attention_mask": attention_mask,
                    "time_ids": time_ids,
                    "task_type": [0 for _ in chunk],
                }

                step_start = time.perf_counter()
                chunk_total_tokens = [int(max_input_len) for _ in chunk]
                if projected_frame_features is not None:
                    if raw_frame_features is not None:
                        precomputed_rgbd = {
                            "image_features": precomputed_image_features,
                            "memory_features": precomputed_memory_features,
                            "raw_image_features": precomputed_raw_image_features,
                            "raw_memory_features": precomputed_raw_memory_features,
                            "raw_meta": {
                                "raw_tokens_per_frame": int(raw_frame_features.shape[1]),
                                "raw_hidden_size": int(raw_frame_features.shape[-1]),
                                "current_images": precomputed_current_images,
                            },
                        }
                    else:
                        precomputed_rgbd = (precomputed_image_features, precomputed_memory_features)
                    setattr(self.model, "_precomputed_rgbd_batch", precomputed_rgbd)
                with torch.inference_mode():
                    try:
                        if self.aurora_step_mode == "next_token_logits":
                            (
                                _,
                                position_ids,
                                expanded_attention_mask,
                                _,
                                inputs_embeds,
                                _,
                            ) = self.model.prepare_inputs_labels_for_multimodal(
                                input_ids,
                                None,
                                attention_mask,
                                None,
                                None,
                                input_dict["images"],
                                None,
                                input_dict["depths"],
                                input_dict["poses"],
                                input_dict["intrinsics"],
                                time_ids,
                                input_dict["task_type"],
                            )
                            outputs = self.model(
                                input_ids=None,
                                attention_mask=expanded_attention_mask,
                                position_ids=position_ids,
                                inputs_embeds=inputs_embeds,
                                use_cache=False,
                                return_dict=True,
                            )
                            if expanded_attention_mask is None:
                                last_indices = torch.full(
                                    (len(chunk),),
                                    int(outputs.logits.shape[1] - 1),
                                    dtype=torch.long,
                                    device=outputs.logits.device,
                                )
                                chunk_total_tokens = [int(outputs.logits.shape[1]) for _ in chunk]
                            else:
                                expanded_lengths = expanded_attention_mask.long().sum(dim=1)
                                chunk_total_tokens = [int(v) for v in expanded_lengths.detach().cpu().tolist()]
                                last_indices = expanded_lengths.to(outputs.logits.device) - 1
                            batch_indices = torch.arange(len(chunk), device=outputs.logits.device)
                            next_token_ids = outputs.logits[batch_indices, last_indices].argmax(dim=-1)
                            generated_texts = self.tokenizer.batch_decode(
                                next_token_ids.unsqueeze(1),
                                skip_special_tokens=False,
                            )
                        else:
                            outputs = self.model.generate(
                                **input_dict,
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=decode_tokens,
                                use_cache=True,
                                return_dict_in_generate=True,
                            )
                            generated_ids = outputs.sequences[:, input_ids.shape[1] :]
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                    finally:
                        if projected_frame_features is not None:
                            setattr(self.model, "_precomputed_rgbd_batch", None)
                batch_latency_ms = (time.perf_counter() - step_start) * 1000.0
                per_step_latency_ms = float(batch_latency_ms / max(len(chunk), 1))
                self.step_latency_ms.extend([per_step_latency_ms for _ in chunk])

                visual_tokens = int(input_dict["images"].shape[1])
                memory_tokens = int(max(0, visual_tokens - 1))
                self.step_visual_tokens.extend([visual_tokens for _ in chunk])
                self.step_memory_tokens.extend([memory_tokens for _ in chunk])
                self.step_total_tokens.extend(chunk_total_tokens)
                self.step_tflops.extend([_estimate_tflops_per_step(self.model, total_tokens) for total_tokens in chunk_total_tokens])

                for step_idx, generated_text in zip(chunk, generated_texts):
                    pred_action = self._first_parsed_action_or_invalid(generated_text)
                    gt_prefix_text = self.actions2text(gt_actions[:step_idx])
                    pred_actions[step_idx] = pred_action
                    self._record_step_debug(
                        step_idx,
                        history_count,
                        gt_prefix_text,
                        generated_text,
                        pred_action,
                    )

        self.model.model.num_history = base_num_history
        return pred_actions

    def evaluate(self):
        """执行完整离线评估并写出结果。

        输出包括 predictions.jsonl（每条 instruction/轨迹的预测统计）和 summary.json
        （整体准确率、分动作准确率、吞吐、延迟、token 数、近似 TFLOPs、显存等指标）。
        """
        self.step_latency_ms.clear()
        self.step_total_tokens.clear()
        self.step_visual_tokens.clear()
        self.step_memory_tokens.clear()
        self.step_tflops.clear()
        self.invalid_prediction_count = 0

        items = self.load_items()
        if self.args.max_episodes > 0:
            items = items[: self.args.max_episodes]

        os.makedirs(self.args.output_path, exist_ok=True)

        total_compared = 0
        total_correct = 0
        class_total = {0: 0, 1: 0, 2: 0, 3: 0}
        class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        error_count = 0

        predictions_path = os.path.join(self.args.output_path, "predictions.jsonl")
        start = time.time()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        with open(predictions_path, "w", encoding="utf-8") as fout:
            for item in tqdm(items, desc="offline_action_acc_tf"):
                try:
                    video_dir = os.path.join(self.args.dataset_root, item["video"], "rgb")
                    if not os.path.isdir(video_dir):
                        raise FileNotFoundError(f"Missing rgb directory: {video_dir}")
                    frame_files = sorted(os.listdir(video_dir))
                    if len(frame_files) == 0:
                        raise RuntimeError(f"Empty rgb directory: {video_dir}")

                    if self.eval_protocol == PROTOCOL_AURORA_REPLAY_GT:
                        # AuroraReplay-GT：逐步生成每个动作，再与 GT 在最短公共长度上比较。
                        gt_actions = [int(a) for a in item["actions"]]
                        pred_actions = self._predict_actions_aurora_replay_gt(item, video_dir, frame_files)

                        compare_len = min(len(gt_actions), len(pred_actions))
                        correct = 0
                        per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}
                        per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
                        for gt, pd in zip(gt_actions[:compare_len], pred_actions[:compare_len]):
                            gt = int(gt)
                            pd = int(pd)
                            per_class_total[gt] += 1
                            if gt == pd:
                                correct += 1
                                per_class_correct[gt] += 1

                        total_compared += compare_len
                        total_correct += correct
                        for k in [0, 1, 2, 3]:
                            class_total[k] += per_class_total[k]
                            class_correct[k] += per_class_correct[k]

                        rec = {
                            "id": item["id"],
                            "ins_idx": item["ins_idx"],
                            "video": item["video"],
                            "instruction": item["instruction"],
                            "gt_actions": gt_actions,
                            "pred_actions": pred_actions,
                            "num_invalid_pred_actions": sum(1 for action in pred_actions if int(action) not in {0, 1, 2, 3}),
                            "compare_len": compare_len,
                            "correct": correct,
                            "episode_action_acc": (correct / compare_len) if compare_len > 0 else 0.0,
                            "gt_len": len(gt_actions),
                            "pred_len": len(pred_actions),
                            "eval_protocol": self.eval_protocol,
                        }
                        if self.save_step_debug:
                            rec["step_debug"] = self._current_step_debug
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue

                    # Legacy-TF-FirstFrame：只用首帧和完整 GT 序列做一次 teacher-forcing 前向。
                    first_frame = os.path.join(video_dir, frame_files[0])
                    image = _load_rgb_or_blank(first_frame)
                    image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                    gt_action_text = self.actions2text(item["actions"])
                    conversation = [
                        {"from": "human", "value": self.build_prompt(item["instruction"])},
                        {"from": "gpt", "value": gt_action_text},
                    ]
                    data_dict = preprocess([conversation], self.tokenizer, True)
                    input_ids = data_dict["input_ids"][0].unsqueeze(0).to(self.device)
                    labels = data_dict["labels"][0].unsqueeze(0).to(self.device)
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

                    depth, pose, intrinsic = self.build_dummy_geom()

                    step_start = time.perf_counter()
                    with torch.inference_mode():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            images=image_tensor.to(torch.bfloat16 if self.device.type == "cuda" else torch.float32),
                            depths=depth.to(torch.bfloat16 if self.device.type == "cuda" else torch.float32),
                            poses=pose.to(torch.bfloat16 if self.device.type == "cuda" else torch.float32),
                            intrinsics=intrinsic.to(torch.bfloat16 if self.device.type == "cuda" else torch.float32),
                            time_ids=[[0]],
                            task_type=[0],
                            return_dict=True,
                        )
                    step_latency_ms = (time.perf_counter() - step_start) * 1000.0
                    self.step_latency_ms.append(float(step_latency_ms))

                    pred_ids = outputs.logits.argmax(dim=-1)
                    valid_mask = labels[0] != IGNORE_INDEX
                    valid_positions = torch.where(valid_mask)[0].tolist()

                    input_len = input_ids.shape[1]
                    logits_len = pred_ids.shape[1]
                    image_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].tolist()
                    memory_positions = torch.where(input_ids[0] == MEMORY_TOKEN_INDEX)[0].tolist()
                    total_extra = max(logits_len - input_len, 0)
                    extra_per_image = (
                        (total_extra // len(image_positions)) if len(image_positions) > 0 else 0
                    )

                    visual_tokens = int(total_extra)
                    memory_tokens = int(len(memory_positions))
                    total_tokens = int(logits_len)
                    self.step_visual_tokens.append(visual_tokens)
                    self.step_memory_tokens.append(memory_tokens)
                    self.step_total_tokens.append(total_tokens)
                    self.step_tflops.append(_estimate_tflops_per_step(self.model, total_tokens))

                    # 模型会把图像占位符展开为视觉 token；这里把 label 的文本位置映射到
                    # logits 序列中的真实位置，才能逐 token 对齐 GT 和 argmax 预测。
                    mapped_positions = []
                    for pos in valid_positions:
                        shift = 0
                        for img_pos in image_positions:
                            if pos > img_pos:
                                shift += extra_per_image
                        mapped_pos = pos + shift
                        if mapped_pos < logits_len:
                            mapped_positions.append((pos, mapped_pos))

                    gt_token_ids = [labels[0][pos].item() for pos, _ in mapped_positions]
                    pred_token_ids = [pred_ids[0][mapped].item() for _, mapped in mapped_positions]

                    gt_text = self.tokenizer.decode(gt_token_ids, skip_special_tokens=False)
                    pred_text = self.tokenizer.decode(pred_token_ids, skip_special_tokens=False)

                    # 解码为文本后再抽取动作符号，最后按最短公共长度计算 micro action accuracy。
                    gt_actions = self.parse_actions(gt_text)
                    pred_actions = self.parse_actions(pred_text)

                    compare_len = min(len(gt_actions), len(pred_actions))
                    correct = 0
                    per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}
                    per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
                    for gt, pd in zip(gt_actions[:compare_len], pred_actions[:compare_len]):
                        per_class_total[gt] += 1
                        if gt == pd:
                            correct += 1
                            per_class_correct[gt] += 1

                    total_compared += compare_len
                    total_correct += correct
                    for k in [0, 1, 2, 3]:
                        class_total[k] += per_class_total[k]
                        class_correct[k] += per_class_correct[k]

                    rec = {
                        "id": item["id"],
                        "ins_idx": item["ins_idx"],
                        "video": item["video"],
                        "instruction": item["instruction"],
                        "gt_actions": gt_actions,
                        "pred_actions": pred_actions,
                        "num_invalid_pred_actions": sum(1 for action in pred_actions if int(action) not in {0, 1, 2, 3}),
                        "compare_len": compare_len,
                        "correct": correct,
                        "episode_action_acc": (correct / compare_len) if compare_len > 0 else 0.0,
                        "gt_len": len(gt_actions),
                        "pred_len": len(pred_actions),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                except Exception as exc:
                    # 单条样本失败不终止整个评估；错误信息写入 jsonl 便于后续排查数据问题。
                    error_count += 1
                    rec = {
                        "id": item.get("id", -1),
                        "ins_idx": item.get("ins_idx", 0),
                        "video": item.get("video", ""),
                        "instruction": item.get("instruction", ""),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        elapsed = time.time() - start
        overall_acc = (total_correct / total_compared) if total_compared > 0 else 0.0
        per_class_acc = {
            str(k): (class_correct[k] / class_total[k]) if class_total[k] > 0 else 0.0
            for k in [0, 1, 2, 3]
        }

        runtime_summary = {}
        if hasattr(self.model, "consume_runtime_metrics_summary"):
            try:
                runtime_summary = self.model.consume_runtime_metrics_summary(reset=True)
            except Exception:
                runtime_summary = {}

        avg_visual_tokens = float(runtime_summary.get("avg_visual_tokens_after", _mean(self.step_visual_tokens)))
        avg_memory_tokens = float(runtime_summary.get("avg_memory_tokens_after", _mean(self.step_memory_tokens)))
        avg_total_tokens = float(runtime_summary.get("avg_total_tokens_after", _mean(self.step_total_tokens)))
        avg_visual_tokens_before = float(runtime_summary.get("avg_visual_tokens_before", avg_visual_tokens))
        avg_memory_tokens_before = float(runtime_summary.get("avg_memory_tokens_before", avg_memory_tokens))
        avg_total_tokens_before = float(runtime_summary.get("avg_total_tokens_before", avg_total_tokens))

        token_reduction_ratio = 0.0
        if self.args.baseline_avg_total_tokens_per_step > 0:
            token_reduction_ratio = (
                self.args.baseline_avg_total_tokens_per_step - avg_total_tokens
            ) / self.args.baseline_avg_total_tokens_per_step

        fps = (total_compared / elapsed) if elapsed > 0 else 0.0
        invalid_prediction_rate = (
            float(self.invalid_prediction_count / total_compared) if total_compared > 0 else 0.0
        )
        latency_mean = _mean(self.step_latency_ms)
        latency_p50 = _percentile(self.step_latency_ms, 50.0)
        latency_p95 = _percentile(self.step_latency_ms, 95.0)
        runtime_tflops = float(runtime_summary.get("approx_tflops_per_step", 0.0))
        avg_tflops = runtime_tflops if runtime_tflops > 0 else _mean(self.step_tflops)

        gpu_stats = _get_gpu_peak_stats(self.device)

        summary = {
            "metric": "offline_action_micro_accuracy_min_len_teacher_forcing",
            "eval_protocol": self.eval_protocol,
            "eval_protocol_name": (
                AURORA_PROTOCOL_NAME
                if self.eval_protocol == PROTOCOL_AURORA_REPLAY_GT
                else "Legacy-TF-FirstFrame"
            ),
            "dataset_root": self.args.dataset_root,
            "num_episodes_eval": len(items),
            "num_episodes_error": error_count,
            "num_actions_compared": total_compared,
            "num_actions_correct": total_correct,
            "num_invalid_predictions": int(self.invalid_prediction_count),
            "invalid_prediction_rate": invalid_prediction_rate,
            "overall_action_acc": overall_acc,
            "per_class_action_acc": per_class_acc,
            "per_class_total": {str(k): v for k, v in class_total.items()},
            "elapsed_seconds": elapsed,
            "model_path": self.args.model_path,
            "token_pruning_flags": os.environ.get("STREAMVLN_EXT_FLAGS", ""),
            "avg_visual_tokens_before": avg_visual_tokens_before,
            "avg_memory_tokens_before": avg_memory_tokens_before,
            "avg_total_tokens_before": avg_total_tokens_before,
            "avg_visual_tokens_per_step": avg_visual_tokens,
            "avg_memory_tokens_per_step": avg_memory_tokens,
            "avg_total_tokens_per_step": avg_total_tokens,
            "runtime_token_reduction_ratio": float(runtime_summary.get("token_reduction_ratio", 0.0)),
            "num_runtime_steps": float(runtime_summary.get("num_runtime_steps", 0.0)),
            "token_reduction_ratio_vs_baseline": token_reduction_ratio,
            "fps": fps,
            "latency_ms_mean": latency_mean,
            "latency_ms_p50": latency_p50,
            "latency_ms_p95": latency_p95,
            "approx_tflops_per_step": avg_tflops,
            "gpu_peak_allocated_mib": gpu_stats["gpu_peak_allocated_mib"],
            "gpu_peak_reserved_mib": gpu_stats["gpu_peak_reserved_mib"],
            "gpu_device_name": gpu_stats["gpu_device_name"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        summary.update(
            {
                key: value
                for key, value in runtime_summary.items()
                if key.startswith("voxel_spatial_")
            }
        )

        with open(os.path.join(self.args.output_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_model(args):
    """根据模型目录内容加载评估模型与 tokenizer。

    支持三种路径形态：训练感知视频压缩器 checkpoint、LoRA/PEFT adapter、完整模型。
    加载完成后会切到 eval 模式、关闭梯度，并初始化模型的环境缓存。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    compressor_checkpoint_path = os.path.join(args.model_path, "training_aware_video_compressor.bin")
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.isfile(compressor_checkpoint_path):
        # 只保存压缩器参数时，需要先加载 base model，再把压缩器 state_dict 填回扩展模块。
        meta_path = os.path.join(args.model_path, "training_aware_compressor_meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        base_model_path = args.base_model_path or meta.get("base_model_path", "")
        if not base_model_path:
            raise ValueError(
                "A training-aware compressor checkpoint needs --base_model_path "
                "or training_aware_compressor_meta.json with base_model_path."
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
        if tokenizer.pad_token is None and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        config = transformers.AutoConfig.from_pretrained(base_model_path)
        model = StreamVLNForCausalLMExt.from_pretrained(
            base_model_path,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            config=config,
            low_cpu_mem_usage=False,
        )
        compressor = getattr(model, "training_aware_video_compressor", None)
        if compressor is None:
            raise ValueError(
                "STREAMVLN_EXT_FLAGS must enable the same training-aware compressor "
                "when loading training_aware_video_compressor.bin."
            )
        state = torch.load(compressor_checkpoint_path, map_location="cpu")
        prefix = "training_aware_video_compressor."
        state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
        missing, unexpected = compressor.load_state_dict(state, strict=False)
        print(
            f"[load_model] loaded training-aware compressor from {compressor_checkpoint_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    elif os.path.isfile(adapter_config_path):
        # LoRA/PEFT adapter 路径：加载 base model 后合并 adapter，得到普通推理模型。
        from peft import PeftConfig, PeftModel
        from peft import import_utils as peft_import_utils
        from peft.tuners.lora import model as peft_lora_model

        # 禁用 bitsandbytes 探测，避免评估环境缺少 bnb 时 PEFT 误走量化加载分支。
        peft_import_utils.is_bnb_available.cache_clear()
        peft_import_utils.is_bnb_4bit_available.cache_clear()
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False

        peft_cfg = PeftConfig.from_pretrained(args.model_path)
        base_model_path = args.base_model_path or peft_cfg.base_model_name_or_path

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
        if tokenizer.pad_token is None and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        config = transformers.AutoConfig.from_pretrained(base_model_path)
        base_model = StreamVLNForCausalLMExt.from_pretrained(
            base_model_path,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            config=config,
            low_cpu_mem_usage=False,
        )
        peft_model = PeftModel.from_pretrained(base_model, args.model_path)
        model = peft_model.merge_and_unload()
    else:
        # 完整模型路径：直接从 model_path 加载扩展后的 StreamVLN 模型。
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
        if tokenizer.pad_token is None and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        config = transformers.AutoConfig.from_pretrained(args.model_path)
        model = StreamVLNForCausalLMExt.from_pretrained(
            args.model_path,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            config=config,
            low_cpu_mem_usage=False,
        )

    model.model.num_history = args.num_history

    # 确保 preprocess() 使用明确的 Qwen 对话模板，避免 default_conversation 为 None。
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)
    return model, tokenizer


def main():
    """命令行入口：解析扩展参数和评估参数，加载模型，然后启动离线动作准确率评估。"""
    ext_args, remaining = extract_ext_args(sys.argv[1:])
    apply_ext_args(ext_args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default=PROTOCOL_AURORA_REPLAY_GT,
        choices=[PROTOCOL_LEGACY_TF_FIRST_FRAME, PROTOCOL_AURORA_REPLAY_GT],
        help=(
            "legacy_tf_first_frame: original single-forward first-frame TF diagnostic; "
            "aurora_replay_gt: AuroraReplay-GT with current+history visuals and GT action prefix conditioning."
        ),
    )
    parser.add_argument(
        "--aurora_decode_max_new_tokens",
        type=int,
        default=16,
        help="Decode budget per step for AuroraReplay-GT protocol.",
    )
    parser.add_argument(
        "--aurora_batch_size",
        type=int,
        default=1,
        help="Batch size for independent AuroraReplay-GT step predictions with matching history_count.",
    )
    parser.add_argument(
        "--aurora_step_mode",
        type=str,
        default="generate",
        choices=["generate", "next_token_logits"],
        help=(
            "generate uses Transformers generation; next_token_logits uses the greedy next-token logits, "
            "equivalent to the first generated token when max_new_tokens=1 and do_sample=False."
        ),
    )
    parser.add_argument(
        "--aurora_precompute_vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute projected vision features once per episode and reuse them across Aurora steps.",
    )
    parser.add_argument(
        "--aurora_vision_batch_size",
        type=int,
        default=16,
        help="Batch size for per-episode vision feature precomputation.",
    )
    parser.add_argument(
        "--save_step_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a small per-episode Aurora step debug trace into predictions.jsonl.",
    )
    parser.add_argument(
        "--debug_max_steps_per_episode",
        type=int,
        default=8,
        help="Maximum number of Aurora steps to include when --save_step_debug is enabled.",
    )
    parser.add_argument("--baseline_avg_total_tokens_per_step", type=float, default=0.0)
    parser.add_argument(
        "--append_stop_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(remaining)

    model, tokenizer = load_model(args)
    evaluator = OfflineTeacherForcingActionAccuracyEvaluator(model, tokenizer, args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
