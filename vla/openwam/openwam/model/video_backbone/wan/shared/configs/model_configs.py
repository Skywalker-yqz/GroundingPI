wan_series = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "openwam.model.video_backbone.wan.models.text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.models.dit.WanModel",
        "extra_kwargs": {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "num_layers": 30,
            "eps": 1e-06,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
        },
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "openwam.model.video_backbone.wan.models.vae.WanVideoVAE38",
        "state_dict_converter": "openwam.model.video_backbone.wan.shared.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
]

MODEL_CONFIGS = wan_series
