"""Save matching images without replacing existing files."""
from pathlib import Path
import re


def copy_unique(source, directory, name):
    # Drive names must never be interpreted as local paths.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).rstrip(' .') or 'photo'
    if name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}:
        name = '_' + name
    base = Path(name)
    index = 0
    while True:
        candidate = Path(directory) / (name if index == 0 else f'{base.stem} ({index}){base.suffix}')
        try:
            target = candidate.open('xb')
        except FileExistsError:
            index += 1
            continue
        try:
            with target, open(source, 'rb') as original:
                while True:
                    chunk = original.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
        return candidate
