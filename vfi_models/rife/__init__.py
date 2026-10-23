import gc
import torch
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList, generic_frame_loop
import typing
import comfy.model_management as model_management
from comfy.model_management import get_torch_device, soft_empty_cache, unload_all_models
from packaging import version

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
    # Arch 4.10 doesn't work due to state dict mismatch
    # "rife410.pth": "4.10",
    # "rife411.pth": "4.10",
    # "rife412.pth": "4.10"
}

DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Module-level model cache: avoids reloading weights on every node execution.
# Key: (ckpt_name, dtype, torch_compile) — invalidated if any of these change.
# Holds at most one model, kept on CPU between runs so it doesn't pin VRAM
# outside of ComfyUI's memory management.
_model_cache: typing.Dict[typing.Tuple, torch.nn.Module] = {}


def _is_oom(e: Exception) -> bool:
    # model_management.is_oom only exists in newer ComfyUI builds
    is_oom = getattr(model_management, "is_oom", None)
    if is_oom is not None:
        return is_oom(e)
    return isinstance(e, torch.cuda.OutOfMemoryError)


class RIFE_VFI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife49.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                # Opt 1: Precision — float32 is safe, float16 is ~2x faster/half VRAM,
                #         bfloat16 is safer than float16 on Ampere+ GPUs (RTX 30xx+)
                "dtype": (DTYPE_OPTIONS, {"default": "float32"}),
                # Opt 2: torch.compile() — JIT-compiles the model graph for 10-30% speedup.
                #         First inference will be slow (compilation), subsequent ones faster.
                #         Requires PyTorch 2.0+. Disable if you hit errors on older builds.
                "torch_compile": ("BOOLEAN", {"default": False,
                                              "tooltip": "Compile the model with torch.compile() for 10-30% faster inference "
                                                         "after the first (warm-up) run. Requires PyTorch 2.0+."}),
                # Opt 7: Batch tasks — process multiple (pair, timestep) tasks per GPU call.
                #         Each task is one interpolated frame. Batching amortises kernel launch
                #         overhead; higher values improve throughput but use more VRAM.
                #         1 = one interpolated frame per call (safest).
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Number of interpolation tasks per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Set to 1 for the most conservative behaviour."}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames: int = 10,
        multiplier: typing.SupportsInt = 2,
        fast_mode: bool = False,
        ensemble: bool = False,
        scale_factor: float = 1.0,
        dtype: str = "float32",
        torch_compile: bool = False,
        batch_size: int = 1,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        """
        Perform video frame interpolation using a given checkpoint model.

        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
            dtype (str, optional): Floating-point precision for inference.
                "float32" — full precision, safest (default).
                "float16" — half precision, ~2x faster and ~50% less VRAM, may have minor artifacts.
                "bfloat16" — brain float16, better numeric range than float16, requires Ampere+ GPU (RTX 30xx+).
            torch_compile (bool, optional): Compile the model with torch.compile() for 10-30% speedup.
                The first inference call will be slow (compilation warm-up); subsequent calls are faster.
                Requires PyTorch 2.0+. Defaults to False.
            batch_size (int, optional): Number of interpolation tasks per GPU call. Each task produces one
                intermediate frame. Higher values improve throughput but use more VRAM. Defaults to 1.

        Returns:
            tuple: A tuple containing the output interpolated frames.
        """
        from .rife_arch import IFNet

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()

        # sudo_rife4 effectively ignores the timestep (measured: t=0.25/0.75 come out as t=0.5),
        # so multipliers above 2 are built by recursive halving instead
        midpoint_only = ckpt_name.startswith("sudo_rife4") and any(
            int(m) > 2 for m in (multiplier if isinstance(multiplier, list) else [multiplier]))

        #  4.26 disable ensemble
        if arch_ver == "4.26":
            ensemble = False  # 4.26 版本不支持 ensemble

        # Cache the model by (ckpt_name, dtype, torch_compile) so repeated node
        # executions skip reading and loading the checkpoint.
        cache_key = (ckpt_name, dtype, torch_compile)
        if cache_key not in _model_cache:
            _model_cache.clear()
            interpolation_model = IFNet(arch_ver=arch_ver)
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            # 4.26 ships its training-only distillation branch; Practical-RIFE ignores it too (strict=False)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("teacher.", "caltime."))}
            interpolation_model.load_state_dict(state_dict)
            if torch_dtype != torch.float32:
                interpolation_model = interpolation_model.to(torch_dtype)
            interpolation_model.eval()
            # Opt 2: torch.compile() — JIT-compiles the model for 10-30% faster inference.
            # The first call triggers compilation (slow); all subsequent calls use the cached graph.
            if torch_compile:
                interpolation_model = torch.compile(interpolation_model)
            _model_cache[cache_key] = interpolation_model
            print(f"Comfy-VFI: Loaded and cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")
        else:
            interpolation_model = _model_cache[cache_key]
            print(f"Comfy-VFI: Using cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")

        frames = preprocess_frames(frames)

        # Normalise multiplier to a per-pair list
        n_pairs = len(frames) - 1
        if isinstance(multiplier, int):
            multipliers = [int(multiplier)] * n_pairs
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (n_pairs - len(multipliers))

         #  4.26 scale_list
        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        if midpoint_only:
            def return_middle_frame(frame_0, frame_1, _timestep, model):
                t = torch.full((frame_0.shape[0], 1, 1, 1), 0.5, dtype=torch_dtype, device=device)
                return model(frame_0.to(torch_dtype), frame_1.to(torch_dtype), t, scale_list, fast_mode, ensemble).clamp(0, 1).float()

            interpolation_model.to(device)
            try:
                with torch.inference_mode():
                    out = generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame,
                                             interpolation_model, interpolation_states=optional_interpolation_states,
                                             use_timestep=False, dtype=torch.float32)
            finally:
                interpolation_model.to("cpu")
            return (postprocess_frames(out),)

        # Build a flat list of all (pair_idx, timestep) tasks, skipping excluded pairs.
        # Each task produces exactly one intermediate frame.
        tasks: typing.List[typing.Tuple[int, float]] = []
        tasks_remaining_per_pair: typing.Dict[int, int] = {}
        for pair_idx in range(n_pairs):
            if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                tasks_remaining_per_pair[pair_idx] = 0
                continue
            m = multipliers[pair_idx]
            n_steps = max(m - 1, 0)
            tasks_remaining_per_pair[pair_idx] = n_steps
            for step in range(1, m):
                tasks.append((pair_idx, step / m))

        # Storage for intermediate frames, keyed by pair index
        results: typing.Dict[int, typing.List[torch.Tensor]] = {i: [] for i in range(n_pairs)}

        frames_processed_since_cache_clear = 0
        pos = 0
        cur_batch_size = batch_size
        unloaded_comfy_models = False

        def run_batch(batch_tasks):
            frame0_list, frame1_list, timestep_list = [], [], []
            for pair_idx, dt in batch_tasks:
                frame0_list.append(frames[pair_idx : pair_idx + 1])
                frame1_list.append(frames[pair_idx + 1 : pair_idx + 2])
                timestep_list.append(dt)

            frame0_batch = torch.cat(frame0_list, dim=0).to(device, dtype=torch_dtype)
            frame1_batch = torch.cat(frame1_list, dim=0).to(device, dtype=torch_dtype)
            # Batched timestep tensor: shape [B, 1, 1, 1] — IFNet expands it internally
            timestep_tensor = torch.tensor(timestep_list, dtype=torch_dtype, device=device).view(-1, 1, 1, 1)

            return interpolation_model(
                frame0_batch,
                frame1_batch,
                timestep_tensor,
                scale_list,
                fast_mode,
                ensemble,
            ).clamp(0, 1).detach().cpu()

        interpolation_model.to(device)
        try:
            # Opt 5: inference_mode disables autograd tracking entirely — lower overhead than no_grad().
            # Opt 7: process batch_size tasks per GPU call. IFNet supports batched tensor timesteps,
            #         so multiple (pair, timestep) combinations can be inferred in a single forward pass.
            with torch.inference_mode():
                while pos < len(tasks):
                    batch_tasks = tasks[pos : pos + cur_batch_size]

                    oom = False
                    try:
                        middle_frames = run_batch(batch_tasks)
                    except Exception as e:
                        if not _is_oom(e) or (unloaded_comfy_models and cur_batch_size == 1):
                            raise
                        oom = True

                    # Recover outside the except block so the traceback's tensors are released first.
                    # First unload models ComfyUI keeps resident (e.g. the video model), then halve the batch.
                    if oom:
                        if not unloaded_comfy_models:
                            print("Comfy-VFI: Out of VRAM, unloading other models and retrying...")
                            unload_all_models()
                            unloaded_comfy_models = True
                        else:
                            cur_batch_size = max(1, cur_batch_size // 2)
                            print(f"Comfy-VFI: Out of VRAM, retrying with batch_size={cur_batch_size}...")
                        gc.collect()
                        soft_empty_cache()
                        continue

                    for idx, (pair_idx, _) in enumerate(batch_tasks):
                        results[pair_idx].append(middle_frames[idx : idx + 1].float())
                        tasks_remaining_per_pair[pair_idx] -= 1
                        # Clear cache after finishing each pair's worth of tasks
                        if tasks_remaining_per_pair[pair_idx] == 0:
                            frames_processed_since_cache_clear += 1
                            if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                                print("Comfy-VFI: Clearing cache...", end=' ')
                                soft_empty_cache()
                                gc.collect()
                                frames_processed_since_cache_clear = 0
                                print("Done cache clearing")

                    pos += len(batch_tasks)
        finally:
            # Offload so the cached model doesn't hold VRAM between runs
            interpolation_model.to("cpu")

        # Assemble output: each original frame followed by its interpolated frames.
        # Source frames stay in their original fp32 so they pass through unchanged.
        output_frames: typing.List[torch.Tensor] = []
        for pair_idx in range(n_pairs):
            output_frames.append(frames[pair_idx : pair_idx + 1])
            for mid in results[pair_idx]:
                output_frames.append(mid)
        output_frames.append(frames[-1:])

        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")
        print(f"Comfy-VFI done! {sum(len(v) for v in results.values()) + len(frames)} frames generated")

        # Always return float32 — numpy and all downstream ComfyUI nodes require it
        out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)
        return (postprocess_frames(out_tensor),)
