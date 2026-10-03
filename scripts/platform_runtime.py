"""Copy only verified container accelerator packages to an isolated virtualenv."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import shutil

PPU = {
 'torch': ('2.9.0','ca65a3012636717e25d7dd6109b18e4aa98bb82a4c6261598cfa364ce05bbeea'),
 'torchvision': ('0.24.0','dbfa70b4c2d4e604c452669d856418b9ed17153e6638442d093a57ec41f9c28c'),
 'flash-attn': ('2.7.4.post1','08af2a4a8f327314fa3c19de68ad8741dbcd382edae2fb18436f84cedd597a40'),
 'triton': ('3.5.0+git4328cd8b','440e3cca50502421dcb5102f36fa8c1c3b19f00237ed0f13a31e012a9491e764'),
}

def inspect_runtime():
    if sys.platform != 'linux' or sys.version_info[:2] != (3,12):
        raise ValueError('this PPU runtime profile requires Linux Python 3.12')
    report={};links={}
    for name,(version,wheel_sha) in PPU.items():
        dist=metadata.distribution(name)
        origin=json.loads(dist.read_text('direct_url.json') or '{}')
        actual=origin.get('archive_info',{}).get('hashes',{}).get('sha256')
        if dist.version!=version or actual!=wheel_sha:
            raise ValueError(f'{name}: container runtime version/wheel provenance mismatch')
        base=Path(dist.locate_file('')).resolve()
        for item in dist.files or []:
            parts=Path(str(item)).parts
            if not parts or '..' in parts or Path(str(item)).is_absolute():continue
            top=parts[0]
            if top=='__pycache__':continue
            source=base/top
            if source.exists():
                if top in links and links[top]!=source:raise ValueError('platform package file collision')
                links[top]=source
        report[name]={'version':version,'wheel_sha256':wheel_sha}
    return report,links

def attach(venv):
    report,links=inspect_runtime()
    venv=Path(venv).resolve()
    config=(venv/'pyvenv.cfg').read_text().lower()
    if 'include-system-site-packages = false' not in config:
        raise ValueError('platform packages require an isolated virtualenv')
    target=venv/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if not target.is_dir():raise ValueError('virtualenv site-packages missing')
    for name in links:
        if (target/name).exists() or (target/name).is_symlink():raise ValueError(f'refusing to overwrite {name}')
    for name,source in links.items():
        if source.is_dir():
            shutil.copytree(source,target/name)
        else:
            shutil.copy2(source,target/name)
    report={'packages':report,'mode':'copied_container_packages','base_environment_inherited':False,
            'note':'Requires the same immutable vendor runtime container; not a standalone public PPU wheel distribution.'}
    (venv/'platform-runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--venv',required=True);args=parser.parse_args()
    print(json.dumps(attach(args.venv),indent=2))
