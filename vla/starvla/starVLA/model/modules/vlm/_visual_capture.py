"""Capture a Qwen3-VL HF model's merged image ViT tokens (`image_embeds`) from the
model's OWN forward, without recomputing the vision tower and without conflating video.

Why this exists
---------------
Stock transformers' Qwen3-VL outputs do not expose the flat `[num_image_tokens, hidden]`
merged+projected visual tokens: the model computes them internally as
`get_image_features(...).pooler_output`, `torch.cat`s them, masked_scatters them into
`inputs_embeds`, and discards the standalone tensor. The vendored modeling file (used on
transformers 4.x) adds an `image_embeds` output field; on stock modeling this shim captures
the same tensor instead.

How
---
For both models, `Model.forward` does `image_embeds = self.get_image_features(pixel_values,
image_grid_thw).pooler_output` (a tuple of per-image tensors) then `torch.cat`. We instance-wrap the
Model's `get_image_features` to stash the cat'd image-only tensor. `get_video_features` *delegates*
to `get_image_features` (video is "same implementation as images"), so naively wrapping
get_image_features would also catch video; we set a flag while inside get_video_features and skip
capture then -> image_embeds stays image-only.

Notes
-----
- One vision pass: we reuse the model's own get_image_features call; no recompute.
- Backprop-safe: the captured tensor keeps its grad_fn (it is the same per-image tensors the model
  cats and scatters), so gradients flow into the vision tower for an unfrozen VLM. The model's cat
  and ours share the underlying per-image tensors -> autograd sums the two consumers correctly.
- Multi-VIEW safe: several images per sample -> one get_image_features call -> pooler_output is the
  concatenation of every view's tokens, split back per-image downstream via image_grid_thw.
- Per-process state: `cached` lives on the wrapper instance (one per rank), repopulated each
  forward; no cross-rank interaction (DeepSpeed ZeRO grad/param sharding is orthogonal).
"""
import torch


class ImageEmbedsCapture:
    def __init__(self, hf_model):
        # hf_model: Qwen3VLForConditionalGeneration
        # hf_model.model: the inner *Model* whose forward scatters image_embeds and owns
        #                 get_image_features / get_video_features.
        inner = hf_model.model
        self.cached = None
        self._in_video = False

        _orig_image = inner.get_image_features
        _orig_video = getattr(inner, "get_video_features", None)

        def _wrapped_image(*args, **kwargs):
            out = _orig_image(*args, **kwargs)
            if not self._in_video:            # skip capture when called from the video path (see below)
                pe = getattr(out, "pooler_output", out)
                if isinstance(pe, (tuple, list)):
                    pe = torch.cat(pe, dim=0)
                self.cached = pe
            return out

        inner.get_image_features = _wrapped_image

        # VIDEO IS NOT A SUPPORTED INPUT here. starVLA's VLA pipeline feeds IMAGES ONLY
        # (multi-camera views are passed as separate images; see build_qwenvl_inputs), so
        # get_video_features is not exercised today. We still guard it defensively because Qwen's
        # get_video_features DELEGATES to get_image_features internally -- if a video path ever ran,
        # the wrapper above would capture video tokens too. So this wrapper flags _in_video=True
        # around the video call to EXCLUDE it from capture -> self.cached stays image-only and is
        # never polluted by video tokens (which would be mis-split by image_grid_thw).
        if _orig_video is not None:
            def _wrapped_video(*args, **kwargs):
                self._in_video = True
                try:
                    return _orig_video(*args, **kwargs)
                finally:
                    self._in_video = False

            inner.get_video_features = _wrapped_video

    def reset(self):
        """Call at the start of each interface forward() so a no-image (text-only) step doesn't
        attach a stale tensor from a previous step."""
        self.cached = None
