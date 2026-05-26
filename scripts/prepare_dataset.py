"""Download and prepare dataset for training.

Usage examples:
  - Use existing extracted dataset directory:
      python scripts/prepare_dataset.py --source-dir path/to/extracted --out data --val-split 0.2

  - Download a zip from a URL and prepare:
      python scripts/prepare_dataset.py --download-url https://example.com/dataset.zip --out data --val-split 0.2

The script expects either a source directory (already extracted) or a URL to a zip archive.
If the source contains class subdirectories (class_name/*.jpg) they will be preserved.
"""
import argparse
import os
import tempfile
import zipfile
import shutil
import random
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


def download_zip(url, dest_path):
    if requests is None:
        raise RuntimeError('requests is required to download files. Install requirements.txt')
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def prepare_split(source_dir, out_dir, val_split=0.2, seed=42):
    """Create `out_dir/train/<class>/...` and `out_dir/val/<class>/...` from source_dir.

    If source_dir contains subdirectories, each subdir is treated as a class. Otherwise all images go into a single class 'images'.
    """
    random.seed(seed)
    src = Path(source_dir)
    out = Path(out_dir)
    train_dir = out / 'train'
    val_dir = out / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # detect class dirs
    class_dirs = [p for p in src.iterdir() if p.is_dir()]
    if not class_dirs:
        # single class using files directly
        class_dirs = [src]

    for cdir in class_dirs:
        class_name = cdir.name
        images = [p for p in cdir.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not images:
            continue
        random.shuffle(images)
        cut = int(len(images) * (1.0 - val_split))
        train_imgs = images[:cut]
        val_imgs = images[cut:]

        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(p, train_dir / class_name / p.name)
        for p in val_imgs:
            shutil.copy2(p, val_dir / class_name / p.name)


def main():
    parser = argparse.ArgumentParser(description='Download and prepare image dataset')
    parser.add_argument('--download-url', help='URL to a zip archive containing dataset')
    parser.add_argument('--source-dir', help='Path to already extracted dataset directory')
    parser.add_argument('--out', default='data', help='Output base directory (creates train/ and val/)')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tmpdir = None
    try:
        if args.download_url:
            tmpdir = tempfile.mkdtemp(prefix='dataset_')
            zip_path = os.path.join(tmpdir, 'dataset.zip')
            print('Downloading', args.download_url)
            download_zip(args.download_url, zip_path)
            print('Extracting to', tmpdir)
            extract_zip(zip_path, tmpdir)
            # assume extracted folder is first child
            children = [p for p in Path(tmpdir).iterdir() if p.is_dir()]
            if children:
                source = children[0]
            else:
                source = tmpdir
        elif args.source_dir:
            source = args.source_dir
        else:
            raise ValueError('Either --download-url or --source-dir must be provided')

        print('Preparing dataset from', source)
        prepare_split(source, args.out, val_split=args.val_split, seed=args.seed)
        print('Prepared dataset in', args.out)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
