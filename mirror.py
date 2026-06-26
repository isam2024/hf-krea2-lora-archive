#!/usr/bin/env python3
"""Archive many source HF model repos into ONE destination repo as subfolders.

Runs on a GitHub Actions runner. Each source repo `ilkerzgi/<name>` is copied
into `ARCHIVE_REPO` under a subfolder `<name>/...` (full contents: weights,
README, preview images). Downloads are batched so several LoRAs go up in a
single commit — far fewer Hub-API calls than one-commit-per-LoRA.

Idempotent and resumable: a LoRA whose subfolder already exists in the archive
is skipped, so re-running only adds what's missing.

Because all commits target the SAME repo, shards MUST run sequentially
(workflow sets max-parallel: 1) to avoid concurrent-commit conflicts. Sharding
still helps by giving per-job checkpoints under the 6h job timeout.

Env vars:
  HF_TOKEN     token with read+write (required)
  ARCHIVE_REPO destination repo id, e.g. k2styles/krea-2-style-loras-archive (required)
  SHARD        this runner's shard index, 0-based (default 0)
  NUM_SHARDS   total number of shards (default 1)
  REPOS_FILE   newline-separated source repo ids (default repos.txt)
  BATCH_SIZE   LoRAs to bundle into one commit (default 12)
  SKIP_REPOS   comma-separated source ids to skip entirely (e.g. the index repo)
"""
import os
import sys
import shutil
import tempfile
import time
import traceback

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

try:
    import httpx
    HTTPX_ERR = (httpx.HTTPStatusError,)
except Exception:  # pragma: no cover
    HTTPX_ERR = tuple()

HF_TOKEN     = os.environ["HF_TOKEN"]
ARCHIVE_REPO = os.environ["ARCHIVE_REPO"]
SHARD        = int(os.environ.get("SHARD", "0"))
NUM_SHARDS   = int(os.environ.get("NUM_SHARDS", "1"))
REPOS_FILE   = os.environ.get("REPOS_FILE", "repos.txt")
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "12"))
SKIP_REPOS   = {s.strip() for s in os.environ.get("SKIP_REPOS", "").split(",") if s.strip()}

MAX_RETRIES = 12
MAX_SLEEP   = 320

api = HfApi(token=HF_TOKEN)


def _status_and_retry_after(exc):
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    retry_after = None
    if resp is not None:
        try:
            ra = resp.headers.get("Retry-After")
            if ra and str(ra).strip().isdigit():
                retry_after = int(ra)
        except Exception:
            pass
    if retry_after is None and "Retry after" in str(exc):
        try:
            retry_after = int(str(exc).split("Retry after")[1].split("second")[0].strip())
        except Exception:
            pass
    return status, retry_after


def with_retry(label, fn, *args, **kwargs):
    """Call fn, retrying on 429 / 5xx with backoff that respects Retry-After."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (HfHubHTTPError, *HTTPX_ERR) as e:
            status, retry_after = _status_and_retry_after(e)
            transient = status == 429 or (status is not None and 500 <= status < 600) \
                        or "429" in str(e) or "rate limit" in str(e).lower()
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = retry_after if retry_after is not None else min(MAX_SLEEP, 20 * attempt)
            wait = min(wait + 5, MAX_SLEEP)
            print(f"    [{label}] {status or '429'} — sleeping {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{label}: retries exhausted")


def subfolder_for(src_id: str) -> str:
    return src_id.split("/")[-1]


def done_subfolders() -> set:
    """Top-level subfolders already present in the archive repo."""
    try:
        files = with_retry("list-archive", api.list_repo_files,
                           ARCHIVE_REPO, repo_type="model", token=HF_TOKEN)
    except Exception as e:
        print(f"  WARN could not list archive ({e!r}); assuming empty", flush=True)
        return set()
    return {f.split("/")[0] for f in files if "/" in f}


def process_batch(batch):
    """Download each LoRA in `batch` into staging/<subfolder>/, then upload the
    whole staging dir to the archive in a single commit."""
    staging = tempfile.mkdtemp(prefix="hfbatch_")
    try:
        names = []
        for src_id in batch:
            sub = subfolder_for(src_id)
            with_retry(f"download {sub}", snapshot_download,
                       repo_id=src_id, repo_type="model",
                       local_dir=os.path.join(staging, sub), token=HF_TOKEN)
            names.append(sub)
        with_retry("upload-batch", api.upload_folder,
                   folder_path=staging, repo_id=ARCHIVE_REPO, repo_type="model",
                   token=HF_TOKEN,
                   commit_message=f"Add {len(names)} LoRAs: {', '.join(names[:6])}"
                                  + (" …" if len(names) > 6 else ""))
        return names
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    with open(REPOS_FILE) as f:
        repos = [ln.strip() for ln in f if ln.strip()]

    # this shard's slice, minus anything explicitly skipped (e.g. the index repo)
    mine = [r for i, r in enumerate(repos)
            if i % NUM_SHARDS == SHARD and r not in SKIP_REPOS]

    done = done_subfolders()
    todo = [r for r in mine if subfolder_for(r) not in done]
    print(f"[shard {SHARD}/{NUM_SHARDS}] {len(mine)} in slice, "
          f"{len(mine) - len(todo)} already in archive, {len(todo)} to add "
          f"-> {ARCHIVE_REPO} (batch={BATCH_SIZE})", flush=True)

    added = 0
    failed = []
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        try:
            names = process_batch(batch)
            added += len(names)
            print(f"[shard {SHARD}] committed {len(names)} ({added}/{len(todo)}): "
                  f"{', '.join(names)}", flush=True)
        except Exception as e:
            print(f"[shard {SHARD}] BATCH FAIL {[subfolder_for(b) for b in batch]}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.extend(batch)

    print(f"\n[shard {SHARD}] DONE: added={added} failed={len(failed)}", flush=True)
    if failed:
        print(f"[shard {SHARD}] FAILED:", flush=True)
        for r in failed:
            print(f"  {r}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
