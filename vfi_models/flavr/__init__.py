import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames, assert_batch_size
import pathlib
import warnings
from .flavr_arch import UNet_3D_3D, InputPadder
import gc

device = get_torch_device()
NBR_FRAME = 4

def build_flavr(model_path):
    sd = torch.load(model_path, map_location='cpu')['state_dict']
    sd = {k.partition("module.")[-1]:v for k,v in sd.items()}

    #Ref: Class UNet_3D_3D
    model = UNet_3D_3D("unet_18", n_inputs=NBR_FRAME, n_outputs=sd["outconv.1.weight"].shape[0] // 3, joinType="concat" , upmode="transpose")
    model.load_state_dict(sd)
    model.to(device).eval()
    del sd
    return model

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["FLAVR_2x.pth", "FLAVR_4x.pth", "FLAVR_8x.pth"]

class FLAVR_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8,
                                       "tooltip": "Set by the checkpoint: FLAVR_2x=2, FLAVR_4x=4, FLAVR_8x=8"}),
                "duplicate_first_last_frames": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"        

    #Reference: https://github.com/danier97/ST-MFNet/blob/main/interpolate_yuv.py#L93
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        duplicate_first_last_frames: bool = False,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        assert_batch_size(frames, batch_size=4, vfi_name="FLAVR")
        interpolation_states = optional_interpolation_states
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        model = build_flavr(model_path)
        # Each checkpoint predicts a fixed number of middle frames (1, 3 or 7)
        ckpt_multiplier = model.n_outputs + 1
        if multiplier != ckpt_multiplier:
            warnings.warn(f"{ckpt_name} always interpolates {ckpt_multiplier}x, ignoring multiplier={multiplier}. Choose FLAVR_2x/4x/8x.pth to change the factor.")
        frames = preprocess_frames(frames)
        padder = InputPadder(frames.shape, 16)
        frames = padder.pad(frames)

        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        for frame_itr in range(len(frames) - 3):
            #Ensure that input frames are in fp32 - the same dtype as model
            frame0, frame1, frame2, frame3 = (
                frames[frame_itr:frame_itr+1].float(),
                frames[frame_itr+1:frame_itr+2].float(), 
                frames[frame_itr+2:frame_itr+3].float(), 
                frames[frame_itr+3:frame_itr+4].float()
            )
            #Does skipping frame i+1 make sanse in this case?
            # A skipped window still emits its source frames, just without the new middle frame
            skipped = interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr) and interpolation_states.is_frame_skipped(frame_itr + 1)

            if frame_itr == 0:
                output_frames.append(frame0)
                if duplicate_first_last_frames:
                    output_frames.append(frame0) # repeat the first frame
                output_frames.append(frame1)
            if not skipped:
                new_frames = model([frame0.to(device), frame1.to(device), frame2.to(device), frame3.to(device)])
                number_of_frames_processed_since_last_cleared_cuda_cache += 2
                output_frames.extend(new_frame.detach().clamp(0, 1).cpu() for new_frame in new_frames)
            output_frames.append(frame2)
            if frame_itr == len(frames) - 4:
                output_frames.append(frame3)
                if duplicate_first_last_frames:
                    output_frames.append(frame3) # repeat the last frame

            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end = ' ')
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")
            gc.collect()
        
        dtype = torch.float32
        output_frames = [frame.cpu().to(dtype=dtype) for frame in output_frames] #Ensure all frames are in cpu
        out = torch.cat(output_frames, dim=0)
        out = padder.unpad(out)
        # clear cache for courtesy
        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )
