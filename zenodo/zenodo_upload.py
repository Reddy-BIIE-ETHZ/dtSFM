#!/usr/bin/env python3
"""zenodo_upload.py — create the dtSFM Paper-1 Zenodo deposit and upload files.

Usage
-----
    export ZENODO_TOKEN=...            # personal access token (write scope)
    # optional: --sandbox to test against sandbox.zenodo.org first
    python zenodo_upload.py --files /path/to/bundle1.tar.zst /path/to/bundle2.tar.zst
    python zenodo_upload.py --sandbox --files ...      # dry-ish run on sandbox

What it does
------------
1. Creates a new deposition (or reuses --deposition-id).
2. Sets metadata from zenodo_metadata.json (sitting next to this script).
3. Uploads each --files entry to the deposition bucket (resumable PUT).
4. Prints the reserved concept DOI + the deposition edit URL.
   It does NOT publish — review on the web UI, then click Publish (publishing
   mints the DOI permanently and is irreversible).

Token: create at https://zenodo.org/account/settings/applications/tokens/new/
with scopes `deposit:write` and `deposit:actions`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests  # pip install requests

HERE = Path(__file__).resolve().parent


def base_url(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[], help="files to upload")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--deposition-id", type=int, default=None,
                    help="reuse an existing draft deposition instead of creating one")
    ap.add_argument("--metadata", default=str(HERE / "zenodo_metadata.json"))
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN (see header).", file=sys.stderr)
        return 2

    api = base_url(args.sandbox) + "/api"
    params = {"access_token": token}
    meta = json.loads(Path(args.metadata).read_text())

    # 1) create or reuse deposition
    if args.deposition_id:
        dep_id = args.deposition_id
        r = requests.get(f"{api}/deposit/depositions/{dep_id}", params=params)
        r.raise_for_status()
        dep = r.json()
    else:
        r = requests.post(f"{api}/deposit/depositions", params=params, json={})
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"deposition id : {dep_id}")
    print(f"edit URL      : {dep['links'].get('html')}")
    print(f"reserved DOI  : {dep.get('metadata', {}).get('prereserve_doi', {}).get('doi', '(set after metadata)')}")

    # 2) set metadata
    r = requests.put(f"{api}/deposit/depositions/{dep_id}", params=params, json=meta)
    if not r.ok:
        print("metadata error:", r.status_code, r.text, file=sys.stderr)
        return 1
    print("metadata set OK")

    # 3) upload files (resumable bucket PUT)
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"  SKIP (missing): {p}", file=sys.stderr)
            continue
        print(f"  uploading {p.name} ({p.stat().st_size/1e6:.1f} MB) ...", flush=True)
        with p.open("rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, params=params)
        r.raise_for_status()
        print(f"    done: {p.name}")

    print("\nDraft ready. Review at the edit URL above, then PUBLISH on the web UI")
    print("(publishing mints the DOI permanently — do it only when the paper is ready).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
