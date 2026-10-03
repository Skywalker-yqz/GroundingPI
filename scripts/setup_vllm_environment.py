"""Create a vLLM serving overlay over an explicitly selected accelerator image."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv', required=True)
    p.add_argument('--platform', choices=('gpu', 'ppu'), required=True)
    p.add_argument('--index-url', default='https://pypi.org/simple')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args(argv)
    target = (ROOT / a.venv).resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        raise ValueError('use a new project-relative environment directory')
    python = str(target / 'bin/python')
    commands = [[sys.executable, '-m', 'venv', '--system-site-packages', str(target)],
                [sys.executable, str(ROOT / 'scripts/prepare_dependencies.py'),
                 '--apply', '--name', 'transformers'],
                [python, '-m', 'pip', 'install', '--index-url', a.index_url,
                 '--no-deps', '-r', str(ROOT / 'requirements/serve.txt')],
                [python, '-m', 'pip', 'install', '--index-url', a.index_url,
                 '--no-deps', '--no-build-isolation', '-e', str(ROOT / 'vendor/transformers')]]
    print(json.dumps(dict(platform=a.platform, commands=commands, apply=a.apply), indent=2))
    if not a.apply:
        return
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        raise ValueError('vLLM runtime requires Linux Python 3.12')
    version = metadata.version('vllm')
    if not version.startswith('0.18.') or ('ppu' in version) != (a.platform == 'ppu'):
        raise ValueError(f'base image vLLM {version} does not match {a.platform}')
    env = dict(os.environ)
    env.pop('PIP_CONSTRAINT', None)
    env['PIP_CONFIG_FILE'] = os.devnull
    for cmd in commands:
        subprocess.run(cmd, env=env, check=True)
    if a.platform == 'ppu':
        # ACEXT resolves its native libraries through the active interpreter's
        # purelib, even when the Python module was inherited from the image.
        base = Path(metadata.distribution('acext').locate_file('')).resolve()
        site = target / 'lib/python3.12/site-packages'
        for name in ('lib', 'include'):
            source = base / name
            if not source.is_dir():
                raise ValueError(f'PPU image is missing ACEXT {name}')
            (site / name).symlink_to(source, target_is_directory=True)
    probe = subprocess.run([python, '-c',
        'import pathlib, torch, transformers, vllm; '
        'source=pathlib.Path(transformers.__file__).resolve(); '
        f'expected=pathlib.Path({str(ROOT / "vendor/transformers/src")!r}).resolve(); '
        'assert transformers.__version__ == "5.7.0" and source.is_relative_to(expected), '
        '"serving requires the bundled Transformers 5.7.0 fork"; '
        'print(torch.__version__, transformers.__version__, source, vllm.__version__)'],
        check=True, capture_output=True, text=True)
    check = subprocess.run([python, '-m', 'pip', 'check'], capture_output=True, text=True)
    (target / 'runtime.json').write_text(json.dumps(dict(platform=a.platform, import_probe=probe.stdout,
        inherited_container=True, pip_check_exit=check.returncode, pip_check=check.stdout), indent=2)+'\n')
    freeze = subprocess.run([python, '-m', 'pip', 'freeze', '--all'], check=True, capture_output=True, text=True)
    (target / 'installed.freeze.txt').write_text(freeze.stdout)
    print(probe.stdout)
    if check.returncode:
        print('Package metadata conflicts are recorded in runtime.json.')


if __name__ == '__main__':
    main()
