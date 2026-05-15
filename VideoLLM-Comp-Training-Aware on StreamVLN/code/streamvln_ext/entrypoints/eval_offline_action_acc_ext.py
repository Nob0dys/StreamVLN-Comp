import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STREAMVLN_ROOT = os.path.join(PROJECT_ROOT, "streamvln")
for path in (PROJECT_ROOT, STREAMVLN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamvln.utils.utils import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    dict_to_cuda,
)
from streamvln_ext.entrypoints.common import apply_ext_args, extract_ext_args  # noqa: E402
from streamvln_ext.model import StreamVLNForCausalLMExt  # noqa: E402


PROTOCOL_LEGACY_AR_MODEL_HISTORY = "legacy_ar_model_history"
PROTOCOL_AURORA_REPLAY_GT = "aurora_replay_gt"
AURORA_PROTOCOL_NAME = "AuroraReplay-GT"


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: List[float], q: float) -> float:
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
    flops = l * (8.0 * t * h * h + 4.0 * t * t * h)
    return float(flops / 1e12)


def _get_gpu_peak_stats(device: torch.device) -> Dict[str, float]:
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


class OfflineActionAccuracyEvaluator:
    def __init__(self, model, tokenizer, args: argparse.Namespace):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = model.get_vision_tower().image_processor

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

        self.actions2idx = OrderedDict(
            {
                "STOP": [0],
                "↑": [1],
                "←": [2],
                "→": [3],
            }
        )
        self.idx2actions = {
            0: "STOP",
            1: "↑",
            2: "←",
            3: "→",
        }
        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]

        prompt = (
            "<video>\nYou are an autonomous navigation assistant. "
            "Your task is to <instruction>. Devise an action sequence to follow "
            "the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        self.conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ""},
        ]
        self.step_latency_ms: List[float] = []
        self.step_total_tokens: List[int] = []
        self.step_visual_tokens: List[int] = []
        self.step_memory_tokens: List[int] = []
        self.step_tflops: List[float] = []
        self.eval_protocol = args.eval_protocol
        self.aurora_decode_max_new_tokens = args.aurora_decode_max_new_tokens
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

    def parse_actions(self, output: str) -> List[int]:
        regex = re.compile(r"STOP|Stop|stop|↑|←|→")
        matches = regex.findall(output)
        parsed = []
        for token in matches:
            if token in {"STOP", "Stop", "stop"}:
                parsed.append(0)
            elif token == "↑":
                parsed.append(1)
            elif token == "←":
                parsed.append(2)
            elif token == "→":
                parsed.append(3)
        return parsed

    def _normalize_actions(self, raw_actions: List[int]) -> List[int]:
        actions = [int(a) for a in raw_actions if isinstance(a, (int, np.integer))]
        if len(actions) > 0 and actions[0] == -1:
            actions = actions[1:]
        actions = [a for a in actions if a in {0, 1, 2, 3}]
        if self.args.append_stop_if_missing and (len(actions) == 0 or actions[-1] != 0):
            actions.append(0)
        return actions

    def actions2text(self, actions: List[int]) -> str:
        return "".join(self.idx2actions[int(a)] for a in actions)

    def preprocess_qwen(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        roles = {"human": "user", "gpt": "assistant"}

        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
        unmask_tokens_idx = [198, im_start, im_end]

        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for source in sources:
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."

            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_message}]
                )

            for conv in source:
                role = roles.get(conv.get("role", conv.get("from", "")), conv.get("role", ""))
                content = conv.get("content", conv.get("value", ""))
                conv_wrapped = [{"role": role, "content": content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv_wrapped)
                input_id += encode_id

            for idx, token_id in enumerate(input_id):
                if token_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if token_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                if input_id[idx] in unmask_tokens_idx:
                    continue

            input_ids.append(input_id)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    def _load_eval_items(self):
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

            normalized_actions = self._normalize_actions(item.get("actions", []))
            if len(normalized_actions) == 0:
                continue

            for ins_idx, instruction in enumerate(instructions):
                records.append(
                    {
                        "id": item.get("id", len(records)),
                        "ins_idx": ins_idx,
                        "instruction": instruction,
                        "video": item.get("video", ""),
                        "actions": normalized_actions,
                    }
                )

        return records

    def _build_dummy_intrinsic(self, width: int, height: int) -> torch.Tensor:
        intrinsic = torch.eye(4, dtype=torch.float32)
        intrinsic[0, 0] = width / 2.0
        intrinsic[1, 1] = height / 2.0
        intrinsic[0, 2] = (width - 1.0) / 2.0
        intrinsic[1, 2] = (height - 1.0) / 2.0
        return intrinsic

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
            torch.from_numpy(depth).float(),
            torch.from_numpy(pose).float(),
            torch.from_numpy(intrinsic).float(),
        )

    def _build_pinhole_intrinsic(self, width: int, height: int) -> torch.Tensor:
        flags = getattr(self.model, "ext_flags", None)
        hfov_deg = float(getattr(flags, "voxel_spatial_offline_hfov_deg", 79.0))
        hfov_rad = math.radians(max(hfov_deg, 1e-3))
        fx = (float(width) / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        cx = (float(width) - 1.0) / 2.0
        cy = (float(height) - 1.0) / 2.0
        intrinsic = torch.eye(4, dtype=torch.float32)
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
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 0] = cos_y
        pose[0, 2] = sin_y
        pose[2, 0] = -sin_y
        pose[2, 2] = cos_y
        pose[0, 3] = x
        pose[2, 3] = z
        return pose

    def _build_offline_frame_geom(
        self,
        frame_idx: int,
        actions: List[int],
        episode_dir: str = None,
        frame_name: str = None,
    ):
        flags = getattr(self.model, "ext_flags", None)
        use_saved_geometry = bool(getattr(flags, "enable_offline_saved_geometry", True))
        if use_saved_geometry and episode_dir is not None and frame_name is not None:
            loaded = self._load_saved_frame_geom(episode_dir, frame_name)
            if loaded is not None:
                return loaded

        crop_h = self.image_processor.crop_size["height"]
        crop_w = self.image_processor.crop_size["width"]
        mode = self._offline_geom_mode()
        if mode not in {"odometry", "odometry_unit_depth", "synthetic_odometry"}:
            return (
                torch.zeros((crop_h, crop_w), dtype=torch.float32),
                torch.eye(4, dtype=torch.float32),
                self._build_dummy_intrinsic(crop_w, crop_h),
            )

        unit_depth = float(getattr(flags, "voxel_spatial_offline_unit_depth_m", 2.0))
        depth = torch.full((crop_h, crop_w), fill_value=max(unit_depth, 1e-3), dtype=torch.float32)
        pose = self._odometry_pose_for_frame(frame_idx, actions)
        intrinsic = self._build_pinhole_intrinsic(crop_w, crop_h)
        return depth, pose, intrinsic

    def _select_history_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 1:
            return []

        previous_count = total_frames - 1
        if self.num_history is None:
            return list(range(previous_count))

        history_cap = max(int(self.num_history), 0)
        if history_cap == 0:
            return []
        if previous_count <= history_cap:
            return list(range(previous_count))

        sampled = np.linspace(0, previous_count - 1, num=history_cap, dtype=np.int32)
        return [int(v) for v in sampled.tolist()]

    def _build_sources(self, instruction: str, include_memory: bool, gt_prefix_text: str):
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace(
            " Where should you go next to stay on track?",
            " Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.",
        )
        if include_memory:
            sources[0]["value"] += f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
        sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + "\n", "")
        sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
        sources[1]["value"] = gt_prefix_text
        return sources

    def _build_aurora_prompt(self, instruction: str, include_memory: bool) -> str:
        prompt = self.conversation[0]["value"]
        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN + "\n", "")
        prompt = prompt.replace("<instruction>.", instruction)
        prompt += f" {DEFAULT_IMAGE_TOKEN}."
        if include_memory:
            prompt += f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
        return prompt

    def _replace_multimodal_token_ids(self, input_ids: List[int]) -> torch.Tensor:
        out = []
        for token_id in input_ids:
            if token_id == self.prompt_image_token_id:
                out.append(IMAGE_TOKEN_INDEX)
            elif token_id == self.prompt_memory_token_id:
                out.append(MEMORY_TOKEN_INDEX)
            else:
                out.append(int(token_id))
        return torch.tensor(out, dtype=torch.long)

    def _build_generation_input_ids(
        self,
        instruction: str,
        include_memory: bool,
        gt_prefix_text: str,
    ) -> torch.Tensor:
        prompt = self._build_aurora_prompt(instruction, include_memory)
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

    def _first_parsed_action_or_invalid(self, text: str) -> int:
        parsed_actions = self.parse_actions(text.strip())
        if len(parsed_actions) == 0:
            self.invalid_prediction_count += 1
            return -1
        return int(parsed_actions[0])

    def _episode_predict_aurora_replay_gt(self, record: Dict) -> Dict:
        video_path = os.path.join(self.args.dataset_root, record["video"])
        rgb_path = os.path.join(video_path, "rgb")

        if not os.path.isdir(rgb_path):
            return {
                "id": record["id"],
                "ins_idx": record["ins_idx"],
                "video": record["video"],
                "instruction": record["instruction"],
                "error": f"Missing rgb directory: {rgb_path}",
            }

        frame_files = sorted(os.listdir(rgb_path))
        if len(frame_files) == 0:
            return {
                "id": record["id"],
                "ins_idx": record["ins_idx"],
                "video": record["video"],
                "instruction": record["instruction"],
                "error": f"Empty rgb directory: {rgb_path}",
            }

        gt_actions = record["actions"]
        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsic_list = []
        pred_actions = []

        decode_tokens = max(int(self.aurora_decode_max_new_tokens), 1)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        base_num_history = int(getattr(self.model.model, "num_history", 0) or 0)

        try:
            for step_id in range(len(gt_actions)):
                self.model.reset_for_env(0)

                frame_idx = min(step_id, len(frame_files) - 1)
                frame_file = os.path.join(rgb_path, frame_files[frame_idx])

                image = _load_rgb_or_blank(frame_file)
                image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]

                depth_tensor, pose_tensor, intrinsic_tensor = self._build_offline_frame_geom(
                    frame_idx,
                    gt_actions,
                    episode_dir=video_path,
                    frame_name=frame_files[frame_idx],
                )

                rgb_list.append(image_tensor)
                depth_list.append(depth_tensor)
                pose_list.append(pose_tensor)
                intrinsic_list.append(intrinsic_tensor)

                history_indices = self._select_history_indices(len(rgb_list))
                history_count = len(history_indices)

                images = [rgb_list[idx] for idx in history_indices] + [rgb_list[-1]]
                depths = [depth_list[idx] for idx in history_indices] + [depth_list[-1]]
                poses = [pose_list[idx] for idx in history_indices] + [pose_list[-1]]
                intrinsics = [intrinsic_list[idx] for idx in history_indices] + [intrinsic_list[-1]]

                if history_count > 0:
                    self.model.model.num_history = history_count
                else:
                    self.model.model.num_history = base_num_history

                gt_prefix_text = self.actions2text(gt_actions[:step_id])
                input_ids = self._build_generation_input_ids(
                    instruction=record["instruction"],
                    include_memory=history_count > 0,
                    gt_prefix_text=gt_prefix_text,
                ).unsqueeze(0)

                input_dict = {
                    "images": torch.stack(images).unsqueeze(0),
                    "depths": torch.stack(depths).unsqueeze(0),
                    "poses": torch.stack(poses).unsqueeze(0),
                    "intrinsics": torch.stack(intrinsics).unsqueeze(0),
                    "inputs": input_ids,
                    "env_id": 0,
                    "time_ids": [[step_id]],
                    "task_type": [0],
                }

                input_dict = dict_to_cuda(input_dict, self.device)
                if self.device.type == "cuda":
                    for key in ["images", "depths", "poses", "intrinsics"]:
                        input_dict[key] = input_dict[key].to(dtype)

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

                input_len = int(input_dict["inputs"].shape[1])
                total_tokens = input_len
                if hasattr(self.model, "cache"):
                    try:
                        cache_slot = self.model.cache[0]
                        if isinstance(cache_slot, dict) and "inputs_embeds" in cache_slot:
                            total_tokens = int(cache_slot["inputs_embeds"].shape[1])
                    except Exception:
                        pass

                visual_tokens = int(input_dict["images"].shape[1])
                memory_tokens = int(max(0, visual_tokens - 1))
                self.step_total_tokens.append(total_tokens)
                self.step_visual_tokens.append(visual_tokens)
                self.step_memory_tokens.append(memory_tokens)
                self.step_tflops.append(_estimate_tflops_per_step(self.model, total_tokens))

                generated_ids = outputs.sequences[:, input_len:]
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
                pred_actions.append(self._first_parsed_action_or_invalid(generated_text))

            compare_len = min(len(pred_actions), len(gt_actions))
            correct = 0
            per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}
            per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
            for gt, pred in zip(gt_actions[:compare_len], pred_actions[:compare_len]):
                per_class_total[int(gt)] += 1
                if int(gt) == int(pred):
                    correct += 1
                    per_class_correct[int(gt)] += 1

            episode_acc = (correct / compare_len) if compare_len > 0 else 0.0

            return {
                "id": record["id"],
                "ins_idx": record["ins_idx"],
                "video": record["video"],
                "instruction": record["instruction"],
                "gt_actions": gt_actions,
                "pred_actions": pred_actions,
                "num_invalid_pred_actions": sum(1 for action in pred_actions if int(action) not in {0, 1, 2, 3}),
                "compare_len": compare_len,
                "correct": correct,
                "episode_action_acc": episode_acc,
                "per_class_total": per_class_total,
                "per_class_correct": per_class_correct,
                "gt_len": len(gt_actions),
                "pred_len": len(pred_actions),
            }
        finally:
            self.model.model.num_history = base_num_history
            self.model.reset_for_env(0)

    def _episode_predict(self, record: Dict) -> Dict:
        if self.eval_protocol == PROTOCOL_AURORA_REPLAY_GT:
            return self._episode_predict_aurora_replay_gt(record)

        video_path = os.path.join(self.args.dataset_root, record["video"])
        rgb_path = os.path.join(video_path, "rgb")

        if not os.path.isdir(rgb_path):
            return {
                "id": record["id"],
                "ins_idx": record["ins_idx"],
                "video": record["video"],
                "instruction": record["instruction"],
                "error": f"Missing rgb directory: {rgb_path}",
            }

        frame_files = sorted(os.listdir(rgb_path))
        if len(frame_files) == 0:
            return {
                "id": record["id"],
                "ins_idx": record["ins_idx"],
                "video": record["video"],
                "instruction": record["instruction"],
                "error": f"Empty rgb directory: {rgb_path}",
            }

        gt_actions = record["actions"]

        self.model.reset_for_env(0)

        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsic_list = []
        time_ids = []
        action_seq = []
        pred_actions = []
        output_ids = None
        past_key_values = None

        step_id = 0
        while step_id < len(gt_actions):
            frame_idx = min(step_id, len(frame_files) - 1)
            frame_file = os.path.join(rgb_path, frame_files[frame_idx])

            image = _load_rgb_or_blank(frame_file)
            image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")[
                "pixel_values"
            ][0]

            depth_tensor, pose_tensor, intrinsic_tensor = self._build_offline_frame_geom(
                frame_idx,
                gt_actions,
                episode_dir=video_path,
                frame_name=frame_files[frame_idx],
            )

            rgb_list.append(image_tensor)
            depth_list.append(depth_tensor)
            pose_list.append(pose_tensor)
            intrinsic_list.append(intrinsic_tensor)
            time_ids.append(step_id)

            if len(action_seq) == 0:
                if output_ids is None:
                    sources = copy.deepcopy(self.conversation)
                    sources[0]["value"] = sources[0]["value"].replace(
                        " Where should you go next to stay on track?",
                        " Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.",
                    )
                    if step_id != 0:
                        sources[0]["value"] += f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                    sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + "\n", "")
                    sources[0]["value"] = sources[0]["value"].replace(
                        "<instruction>.", record["instruction"]
                    )
                    add_system = True
                else:
                    sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                    add_system = False

                input_ids, _ = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                if output_ids is not None:
                    input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

                images = rgb_list[-1:]
                depths = depth_list[-1:]
                poses = pose_list[-1:]
                intrinsics = intrinsic_list[-1:]

                if step_id != 0 and step_id % self.num_frames == 0:
                    if self.num_history is None:
                        history_ids = slice(0, time_ids[0], self.num_future_steps)
                    else:
                        history_stride = max(time_ids[0] // self.num_history, 1)
                        history_ids = slice(0, time_ids[0], history_stride)

                    images = rgb_list[history_ids] + images
                    depths = depth_list[history_ids] + depths
                    poses = pose_list[history_ids] + poses
                    intrinsics = intrinsic_list[history_ids] + intrinsics

                input_dict = {
                    "images": torch.stack(images).unsqueeze(0),
                    "depths": torch.stack(depths).unsqueeze(0),
                    "poses": torch.stack(poses).unsqueeze(0),
                    "intrinsics": torch.stack(intrinsics).unsqueeze(0),
                    "inputs": input_ids,
                    "env_id": 0,
                    "time_ids": [time_ids],
                    "task_type": [0],
                }

                input_dict = dict_to_cuda(input_dict, self.device)
                if self.device.type == "cuda":
                    for key in ["images", "depths", "poses", "intrinsics"]:
                        input_dict[key] = input_dict[key].to(torch.bfloat16)

                step_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **input_dict,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=self.args.max_new_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values,
                    )
                step_latency_ms = (time.perf_counter() - step_start) * 1000.0
                self.step_latency_ms.append(float(step_latency_ms))

                total_tokens = int(input_dict["inputs"].shape[1])
                if hasattr(self.model, "cache"):
                    try:
                        cache_slot = self.model.cache[0]
                        if isinstance(cache_slot, dict) and "inputs_embeds" in cache_slot:
                            total_tokens = int(cache_slot["inputs_embeds"].shape[1])
                    except Exception:
                        pass

                visual_tokens = int(input_dict["images"].shape[1])
                memory_tokens = int(max(0, visual_tokens - 1))

                self.step_total_tokens.append(total_tokens)
                self.step_visual_tokens.append(visual_tokens)
                self.step_memory_tokens.append(memory_tokens)
                self.step_tflops.append(_estimate_tflops_per_step(self.model, total_tokens))

                output_ids = outputs.sequences
                past_key_values = outputs.past_key_values
                llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                action_seq = self.parse_actions(llm_outputs)
                if len(action_seq) == 0:
                    self.invalid_prediction_count += 1
                    action_seq = [-1]

            pred_action = int(action_seq.pop(0))
            pred_actions.append(pred_action)
            step_id += 1

            if step_id % self.num_frames == 0:
                self.model.reset_for_env(0)
                output_ids = None
                past_key_values = None
                time_ids = []

        compare_len = min(len(pred_actions), len(gt_actions))
        correct = 0
        per_class_total = {0: 0, 1: 0, 2: 0, 3: 0}
        per_class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        for gt, pred in zip(gt_actions[:compare_len], pred_actions[:compare_len]):
            per_class_total[int(gt)] += 1
            if int(gt) == int(pred):
                correct += 1
                per_class_correct[int(gt)] += 1

        episode_acc = (correct / compare_len) if compare_len > 0 else 0.0

        return {
            "id": record["id"],
            "ins_idx": record["ins_idx"],
            "video": record["video"],
            "instruction": record["instruction"],
            "gt_actions": gt_actions,
            "pred_actions": pred_actions,
            "num_invalid_pred_actions": sum(1 for action in pred_actions if int(action) not in {0, 1, 2, 3}),
            "compare_len": compare_len,
            "correct": correct,
            "episode_action_acc": episode_acc,
            "per_class_total": per_class_total,
            "per_class_correct": per_class_correct,
            "gt_len": len(gt_actions),
            "pred_len": len(pred_actions),
        }

    def evaluate(self):
        self.step_latency_ms.clear()
        self.step_total_tokens.clear()
        self.step_visual_tokens.clear()
        self.step_memory_tokens.clear()
        self.step_tflops.clear()
        self.invalid_prediction_count = 0

        records = self._load_eval_items()
        if self.args.max_episodes > 0:
            records = records[: self.args.max_episodes]

        os.makedirs(self.args.output_path, exist_ok=True)

        results = []
        total_compared = 0
        total_correct = 0
        class_total = {0: 0, 1: 0, 2: 0, 3: 0}
        class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
        error_count = 0

        start_time = time.time()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        for record in tqdm(records, desc="offline_action_acc"):
            try:
                result = self._episode_predict(record)
            except Exception as exc:
                result = {
                    "id": record.get("id", -1),
                    "ins_idx": record.get("ins_idx", 0),
                    "video": record.get("video", ""),
                    "instruction": record.get("instruction", ""),
                    "error": str(exc),
                }

            if "error" in result:
                error_count += 1
                results.append(result)
                continue

            total_compared += result["compare_len"]
            total_correct += result["correct"]
            for act in [0, 1, 2, 3]:
                class_total[act] += result["per_class_total"][act]
                class_correct[act] += result["per_class_correct"][act]
            results.append(result)

        elapsed = time.time() - start_time
        overall_acc = (total_correct / total_compared) if total_compared > 0 else 0.0
        per_class_acc = {
            str(act): (class_correct[act] / class_total[act]) if class_total[act] > 0 else 0.0
            for act in [0, 1, 2, 3]
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
            "metric": "offline_action_micro_accuracy_min_len",
            "eval_protocol": self.eval_protocol,
            "eval_protocol_name": (
                AURORA_PROTOCOL_NAME
                if self.eval_protocol == PROTOCOL_AURORA_REPLAY_GT
                else "Legacy-AR-ModelHistory"
            ),
            "dataset_root": self.args.dataset_root,
            "num_episodes_eval": len(records),
            "num_episodes_success": len(results) - error_count,
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
            "avg_visual_tokens_per_step": avg_visual_tokens,
            "avg_memory_tokens_per_step": avg_memory_tokens,
            "avg_total_tokens_per_step": avg_total_tokens,
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

        with open(os.path.join(self.args.output_path, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(os.path.join(self.args.output_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_model(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    compressor_checkpoint_path = os.path.join(args.model_path, "training_aware_video_compressor.bin")
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.isfile(compressor_checkpoint_path):
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
        from peft import PeftConfig, PeftModel
        from peft import import_utils as peft_import_utils
        from peft.tuners.lora import model as peft_lora_model

        peft_import_utils.is_bnb_available.cache_clear()
        peft_import_utils.is_bnb_4bit_available.cache_clear()
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False

        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = args.base_model_path or peft_config.base_model_name_or_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
        config = transformers.AutoConfig.from_pretrained(args.model_path)
        model = StreamVLNForCausalLMExt.from_pretrained(
            args.model_path,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            config=config,
            low_cpu_mem_usage=False,
        )

    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)
    return model, tokenizer


def main():
    ext_args, stage1_remaining = extract_ext_args(sys.argv[1:])
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
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default=PROTOCOL_LEGACY_AR_MODEL_HISTORY,
        choices=[PROTOCOL_LEGACY_AR_MODEL_HISTORY, PROTOCOL_AURORA_REPLAY_GT],
        help=(
            "legacy_ar_model_history: original autoregressive rollout using model-generated history; "
            "aurora_replay_gt: AuroraReplay-GT with current+history visuals and GT action prefix conditioning."
        ),
    )
    parser.add_argument(
        "--aurora_decode_max_new_tokens",
        type=int,
        default=16,
        help="Decode budget per step for AuroraReplay-GT protocol.",
    )
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--baseline_avg_total_tokens_per_step", type=float, default=0.0)
    parser.add_argument(
        "--append_stop_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args(stage1_remaining)

    model, tokenizer = load_model(args)
    evaluator = OfflineActionAccuracyEvaluator(model=model, tokenizer=tokenizer, args=args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
