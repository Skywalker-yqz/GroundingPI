"""Validate final VLM configuration and bound resources."""
import json
from pathlib import Path
from . import config_runtime


def validate(path, *, expected_world_size=None, check_resources=True):
    cfg = config_runtime.load_config(path)
    model, training, runtime = cfg['model'], cfg['training'], cfg['runtime']
    if model['type'] != 'groundingpi':
        raise ValueError('portable profile requires groundingpi')
    if runtime.get('validation_profile') != 'release':
        raise ValueError('portable profile must be explicitly selected')
    if not cfg['datasets']:
        raise ValueError('at least one prepared cache is required')
    for item in cfg['datasets']:
        if type(item.get('repeat')) is not int or item['repeat'] < 1:
            raise ValueError('dataset repeat must be a positive integer')
        count = item.get('sample_count')
        if count is not None and (type(count) is not int or count < 1 or item['repeat'] != 1):
            raise ValueError('sample_count requires a positive integer and repeat=1')
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'max_length',
                'packing_length', 'dataset_num_proc', 'packing_num_proc'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if training['packing_length'] > training['max_length']:
        raise ValueError('packing_length exceeds max_length')
    if training['packing'] and training['attn_impl'] not in ('flash_attn', 'flash_attention_2'):
        raise ValueError('GroundingPI packing requires FlashAttention')
    if training['num_train_epochs'] <= 0 or training['learning_rate'] <= 0:
        raise ValueError('epochs and learning_rate must be positive')
    if runtime['expected_world_size'] != runtime['expected_nodes'] * runtime['expected_gpus_per_node']:
        raise ValueError('invalid runtime topology')
    if expected_world_size is not None and runtime['expected_world_size'] != expected_world_size:
        raise ValueError('runtime world size differs from launcher')
    config_runtime.validate_plugin(Path(runtime['external_plugin']))
    if not check_resources:
        return cfg
    model_path = Path(model['path'])
    metadata = json.loads((model_path / 'config.json').read_text())
    if metadata.get('model_type') != 'groundingpi' or metadata.get('text_config', {}).get('model_type') != 'qwen3':
        raise ValueError('expected GroundingPI with plain Qwen3')
    for name, digest in config_runtime.EXPECTED_RUNTIME_SHA256.items():
        if config_runtime.sha256(model_path / name) != digest:
            raise ValueError(f'model implementation hash differs: {name}')
    manifest = json.loads(Path(model['manifest']).read_text())
    expected = {'status': 'PASS', 'weight_sharing': 'untied', 'coordinate_id_start': 151669,
                'coordinate_id_end': 152668, 'separator_token_id': 152669, 'vocab_size': 152670}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('tokenizer manifest violates the released spatial-token contract')
    if manifest.get('tokenizer_sha256') != config_runtime.sha256(model_path / 'tokenizer.json'):
        raise ValueError('tokenizer SHA differs from its manifest')
    index = model_path / 'model.safetensors.index.json'
    weights = [model_path / 'model.safetensors']
    if index.is_file():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if not names or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('invalid weight shard index')
        weights = [model_path / n for n in names]
    if not all(p.is_file() and p.stat().st_size for p in weights):
        raise ValueError('model weight files are incomplete')
    from models.dependency_contract import verify_dependency
    verify_dependency("swift")
    verify_dependency("transformers")
    # Also prove the installed modules come from those forks, rather than merely
    # finding an unused checkout with the correct HEAD on disk.
    import swift
    import transformers
    for module, directory in ((swift, config_runtime.SWIFT_ROOT), (transformers, config_runtime.TRANSFORMERS_ROOT)):
        if not Path(module.__file__).resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'{module.__name__} is not imported from the pinned fork')
    for item in cfg['datasets']:
        if not Path(item['path']).is_dir(): raise ValueError('prepared cache directory is missing')
        cache = json.loads(Path(item['manifest']).read_text())
        rows = cache.get('num_rows')
        if type(rows) is not int or rows < 1: raise ValueError('cache manifest has no positive num_rows')
        if item.get('sample_count', rows) > rows: raise ValueError('sample_count exceeds cache rows')
    return cfg
