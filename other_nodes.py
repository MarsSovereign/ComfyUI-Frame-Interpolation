import comfy.samplers
import nodes
import torch

class Gradually_More_Denoise_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    
                    "start_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_increment": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                    "denoise_increment_steps": ("INT", {"default": 20, "min": 1, "max": 10000})
                     },
                "optional": { "optional_vae": ("VAE",) }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "ComfyUI-Frame-Interpolation/others"

    def sample(self, model, positive, negative, latent_image,
               seed, steps, cfg, sampler_name, scheduler,start_denoise, denoise_increment, denoise_increment_steps, optional_vae=None):
        if start_denoise + denoise_increment * denoise_increment_steps > 1.0:
            raise Exception(f"Max denoise strength can't over 1.0 (start_denoise={start_denoise}, denoise_increment={denoise_increment}, denoise_increment_steps={denoise_increment_steps}")

        copied_latent = latent_image.copy()
        out_samples = []
        
        for latent_sample in copied_latent["samples"]:
            # Keep the batch dim so this works for image (NCHW) and video (NCTHW) latents
            latent = {"samples": latent_sample.unsqueeze(0)}
            gradually_denoising_samples = [
                nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=start_denoise + denoise_increment * i
                )[0]["samples"]
                for i in range(denoise_increment_steps)
            ]
            out_samples.extend(gradually_denoising_samples)

        copied_latent["samples"] = torch.cat(out_samples, dim=0)
        return (model, positive, negative, copied_latent, optional_vae)