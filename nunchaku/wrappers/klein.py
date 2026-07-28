"""
ComfyFlux2KleinWrapper - bridges nunchaku's FLUX.2 Klein transformer with ComfyUI.

This module is auto-generated and embedded in Magic Assistant.
"""

from typing import Callable, Optional, Tuple

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.model_patcher import ModelPatcher
from einops import rearrange, repeat
from torch import nn

from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context

class ComfyFlux2KleinWrapper(nn.Module):
    """
    Wrapper for Nunchaku FLUX.2 Klein transformer to support ComfyUI workflows,
    LoRA composition, and caching.

    Parameters
    ----------
    model : NunchakuFlux2Transformer2DModel
        The underlying Nunchaku FLUX.2 Klein model to wrap.
    config : dict
        Model configuration dictionary.
    pulid_pipeline : object, optional
        Optional pipeline for Pulid integration.
    customized_forward : Callable, optional
        Optional custom forward function.
    forward_kwargs : dict, optional
        Additional keyword arguments for the forward pass.
    ctx_for_copy : dict
        A dict that holds initialization context for later duplication of this object.
    """

    def __init__(
        self,
        model: NunchakuFlux2Transformer2DModel,
        config: dict,
        pulid_pipeline=None,
        customized_forward: Callable = None,
        forward_kwargs: Optional[dict] = None,
        ctx_for_copy: Optional[dict] = None,
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.pulid_pipeline = pulid_pipeline
        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs

        self.ctx_for_copy = (ctx_for_copy or {}).copy()

        self._prev_timestep = None
        self._cache_context = None

    # ------------------------------------------------------------------ #
    # LoRA management - delegates to the underlying nunchaku transformer
    # via SVDQLoRAMixin (update_lora_params / set_lora_strength / reset_lora)
    # ------------------------------------------------------------------ #

    def update_lora_params(self, lora_path_or_state_dict, strength: float = 1.0) -> None:
        """Load and apply LoRA weights via the nunchaku transformer's native API."""
        self.model.update_lora_params(lora_path_or_state_dict, strength=strength)

    def set_lora_strength(self, strength: float = 1.0) -> None:
        """Adjust LoRA strength without reloading weights."""
        self.model.set_lora_strength(strength)

    def reset_lora(self) -> None:
        """Remove all LoRA effects and restore original weights."""
        self.model.reset_lora()

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        """
        Preprocess an input image tensor for the model.

        Pads and rearranges the image into patches and generates corresponding image IDs.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
        index : int, optional
            Index for image ID encoding.
        h_offset : int, optional
            Height offset for patch IDs.
        w_offset : int, optional
            Width offset for patch IDs.

        Returns
        -------
        img : torch.Tensor
            Rearranged image tensor of shape (batch, num_patches, patch_dim).
        img_ids : torch.Tensor
            Image ID tensor of shape (batch, num_patches, num_axes).
        """
        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 1)
        axes_dim = self.config.get("axes_dim", [32, 32, 32, 32])
        num_axes = len(axes_dim)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        id_dtype = torch.float32
        img_ids = torch.zeros((h_len, w_len, num_axes), device=x.device, dtype=id_dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=id_dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=id_dtype
        ).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context,
        y=None,
        guidance=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        """
        Forward pass for the wrapped FLUX.2 Klein model.

        Handles LoRA composition, caching, PuLID integration, and reference latents.
        """
        if transformer_options is None:
            transformer_options = {}
        if y is None:
            y = transformer_options.get("y")
        if guidance is None:
            guidance = kwargs.get("guidance")

        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFlux2Transformer2DModel)

        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 1)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]

        ref_latents = kwargs.get("ref_latents")
        ref_index_scale = float(self.config.get("ref_index_scale", 10.0))
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0.0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                index += ref_index_scale
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        axes_dim = self.config.get("axes_dim", [32, 32, 32, 32])
        num_axes = len(axes_dim)
        id_dtype = torch.float32
        txt_ids = torch.zeros((bs, context.shape[1], num_axes), device=x.device, dtype=id_dtype)
        txt_ids_dims = self.config.get("txt_ids_dims", [3])
        if len(txt_ids_dims) > 0:
            seq = context.shape[1]
            for i in txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(
                    0, seq - 1, steps=seq, device=x.device, dtype=id_dtype
                )

        if getattr(model, "residual_diff_threshold_multi", 0) != 0 or getattr(model, "_is_cached", False):
            cache_invalid = False

            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                if self.customized_forward is None:
                    out = model(
                        hidden_states=img,
                        encoder_hidden_states=context,
                        guidance=guidance if self.config["guidance_embed"] else None,
                    ).sample
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        **self.forward_kwargs,
                    ).sample
        else:
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    **self.forward_kwargs,
                ).sample

        if self.pulid_pipeline is not None and hasattr(model, "transformer_blocks"):
            self.model.transformer_blocks[0].pulid_ca = None

        out = out[:, :img_tokens]
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h_orig, :w_orig]

        self._prev_timestep = timestep_float
        return out

def copy_with_ctx(model_wrapper: ComfyFlux2KleinWrapper) -> Tuple[ComfyFlux2KleinWrapper, ModelPatcher]:
    """
    Duplicates a ComfyFlux2KleinWrapper object with its initialization context.

    Also creates a ModelPatcher object that holds the model_base object.
    """
    from comfy.model_base import BaseModel

    ctx_for_copy = model_wrapper.ctx_for_copy
    ret_model_wrapper = ComfyFlux2KleinWrapper(
        model_wrapper.model,
        config=ctx_for_copy["comfy_config"]["model_config"],
        ctx_for_copy={
            "comfy_config": ctx_for_copy["comfy_config"],
            "model_config": ctx_for_copy["model_config"],
            "device": ctx_for_copy["device"],
            "device_id": ctx_for_copy["device_id"],
        },
    )
    model_base = ctx_for_copy["model_config"].get_model({})
    model_base.diffusion_model = ret_model_wrapper
    ret_model = ModelPatcher(model_base, ctx_for_copy["device"], ctx_for_copy["device_id"])
    return ret_model_wrapper, ret_model
