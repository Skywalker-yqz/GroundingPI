# Training

This module provides data preparation, tokenizer tools, runtime checks, training, and checkpoint saving and resumption.

- `data/`: JSONL row validation and bounding-box/point spatial token formats. Raw-data-to-cache conversion is not included.
- `launcher/`: consistency checks for models, data, checkpoints, and distributed launch settings.
- `vlm/`: the VLM ms-swift plugin, argument conversion, and training configuration checks.
- `runtime/`: cache loading, media path handling, and runtime utilities used during training.
- `tokenizer/`: spatial token extension and checks for consistency between tokenizer and model vocabularies.

See the [training guide](../docs/TRAINING.md). Run commands from the project root.
