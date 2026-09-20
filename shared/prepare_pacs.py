from __future__ import annotations

"""Materialize PACS from its Hugging Face copy (`flwrlabs/pacs`: the standard 9,991-image release
as one parquet file with `image`, `domain`, `label` columns) into the folder layout the loaders
and `shared/pacs_protocol.py` expect:

    <out>/<domain>/<class>/<domain>_<row>.<jpg|png>      (image bytes are written unmodified)

Original PACS file names repeat across classes and domains (pic_001.jpg ...), so files are named
by their row in the parquet, which is fixed -- the listing, and therefore the seeded split, is
identical on every machine. Run `python -m shared.verify_pacs` afterwards to check the counts.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from shared.pacs import CLASSES, DOMAINS
from shared.verify_pacs import verify

PARQUET_URL = "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/data/train-00000-of-00001.parquet"


def download_parquet(dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        print(f"downloading {PARQUET_URL} -> {dest}")
        urllib.request.urlretrieve(PARQUET_URL, dest + ".part")
        os.replace(dest + ".part", dest)
    return dest


def prepare(out_root: str, parquet_path: str) -> None:
    pf = pq.ParquetFile(parquet_path)
    hf_meta = json.loads(pf.schema_arrow.metadata[b"huggingface"].decode())
    label_names = hf_meta["info"]["features"]["label"]["names"]
    if label_names != CLASSES:
        raise ValueError(f"parquet label order {label_names} != expected {CLASSES}")

    row = 0
    for batch in pf.iter_batches(batch_size=200, columns=["image", "domain", "label"]):
        columns = batch.to_pydict()
        for image, domain, label in zip(columns["image"], columns["domain"], columns["label"]):
            if domain not in DOMAINS:
                raise ValueError(f"unexpected domain {domain!r} at row {row}")
            suffix = Path(image["path"] or "").suffix.lower() or ".jpg"
            target = Path(out_root) / domain / CLASSES[label] / f"{domain}_{row:05d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image["bytes"])
            row += 1
    print(f"wrote {row} images under {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Destination PACS root (put this on fast local disk)")
    parser.add_argument("--parquet", required=True, help="Where to cache the parquet (downloaded if missing)")
    args = parser.parse_args()

    if os.path.isdir(os.path.join(args.out, "photo")) and verify(args.out):
        print("PACS already prepared at", args.out)
    else:
        prepare(args.out, download_parquet(args.parquet))
        verify(args.out)
