import torch
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames
import typing
from comfy.model_management import get_torch_device
from vfi_utils import generic_frame_loop, InterpolationStateList

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["IFRNet_S_Vimeo90K.pth", "IFRNet_L_Vimeo90K.pth", "IFRNet_S_GoPro.pth", "IFRNet_L_GoPro.pth"]

class IFRNet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
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
        clear_cache_after_n_frames: typing.SupportsInt = 1,
        multiplier: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        from .IFRNet_S_arch import IRFNet_S
        from .IFRNet_L_arch import IRFNet_L
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IRFNet_S() if 'S' in ckpt_name else IRFNet_L()
        interpolation_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        # Vimeo90K checkpoints were trained on triplets and only predict the middle frame (measured:
        # t=0.25/0.75 come out as t=0.5), so build higher multipliers by recursive halving instead
        use_timestep = "GoPro" in ckpt_name

        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor):
            # forward() takes (img0, img1, scale_factor, timestep), so pass by name
            return model(frame_0, frame_1, scale_factor=scale_factor, timestep=0.5 if timestep is None else timestep)
        
        args = [interpolation_model, scale_factor]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, use_timestep=use_timestep, dtype=torch.float32)
        )
        return (out,)
