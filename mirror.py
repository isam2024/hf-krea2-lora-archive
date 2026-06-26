#!/usr/bin/env python3
"""Copy a shard of HF model repos from a source account to a destination account.

Runs on a GitHub Actions runner. For each repo in this shard it:
  1. Checks the destination — if the mirror already has every source file, skips.
  2. Otherwise downloads the source repo to a temp dir, creates the dest repo,
     and uploads the folder. The temp dir is deleted after each repo so disk
     never fills (one ~235 MB repo at a time).

Idempotent and resumable: re-running only touches repos that are missing or
incomplete on the destination.

Rate-limit aware: HF enforces ~1000 API requests per 300s window, account-wide.
Every HF call is wrapped in `with_retry`, which on a 429 sleeps the server's
`Retry-After` (the full window, ~240s) and retries. Combined with low shard
parallelism this self-throttles to stay under the cap instead of failing.

Env vars:
  HF_TOKEN     token with read+write (read source, write destination) (required)
  DEST_USER    destination namespace, e.g. k2styles (required)
  SHARD        this runner's shard index, 0-based (default 0)
  NUM_SHARDS   total number of shards (default 1)
  PRIVATE      "true" to create dest repos private (default false)
  REPOS_FILE   path to newline-separated source repo ids (default repos.txt)
  BASE_DELAY   seconds to pause between repos to smooth bursts (default 1.5)
"""
import os
import sys
import shutil
import tempfile
import time
import traceback

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
)

try:
    import httpx
    HTTPX_ERR = (httpx.HTTPStatusError,)
except Exception:  # pragma: no cover
    HTTPX_ERR = tuple()

HF_TOKEN   = os.environ["HF_TOKEN"]
DEST_USER  = os.environ["DEST_USER"]
SHARD      = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
PRIVATE    = os.environ.get("PRIVATE", "false").lower() == "true"
REPOS_FILE = os.environ.get("REPOS_FILE", "repos.txt")
BASE_DELAY = float(os.environ.get("BASE_DELAY", "1.5"))

MAX_RETRIES = 12          # generous: a 429 window is ~240s, we may wait several
MAX_SLEEP   = 320         # cap on any single backoff sleep

api = HfApi(token=HF_TOKEN)


def _status_and_retry_after(exc):
    """Extract (http_status, retry_after_seconds) from an HF/httpx error, if any."""
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
    # fall back to parsing the message ("Retry after 232 seconds")
    if retry_after is None:
        msg = str(exc)
        if "Retry after" in msg:
            try:
                retry_after = int(msg.split("Retry after")[1].split("second")[0].strip())
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
            wait = min(wait + 5, MAX_SLEEP)  # small cushion past the stated window
            print(f"    [{label}] {status or '429'} — sleeping {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{label}: retries exhausted")


def dest_id_for(src_id: str) -> str:
    return f"{DEST_USER}/{src_id.split('/')[-1]}"


def fetch_existing_dest_names() -> set:
    """One paginated API call: the set of model names already on the dest account.

    Lets us cheaply tell 'definitely not copied yet' (skip both list calls and
    go straight to download) from 'exists, must compare files'. This is the big
    rate-limit saver: not-yet-copied repos cost 0 metadata calls here."""
    try:
        models = with_retry("list-dest-account", api.list_models, author=DEST_USER)
        return {m.id.split("/")[-1] for m in models}
    except Exception as e:
        print(f"  WARN could not pre-list dest account ({e!r}); "
              f"falling back to per-repo checks", flush=True)
        return None  # signal: fall back to per-repo existence check


def already_complete(src_id: str, dest_id: str, existing_names) -> bool:
    """True if the destination repo exists and contains every source file."""
    name = dest_id.split("/")[-1]
    if existing_names is not None and name not in existing_names:
        return False  # not on dest yet -> copy, no metadata calls spent
    try:
        dest_files = set(with_retry("list-dest", api.list_repo_files,
                                    dest_id, repo_type="model", token=HF_TOKEN))
    except RepositoryNotFoundError:
        return False
    try:
        src_files = set(with_retry("list-src", api.list_repo_files,
                                   src_id, repo_type="model", token=HF_TOKEN))
    except Exception:
        return False  # can't compare -> attempt copy
    return src_files.issubset(dest_files)


def copy_one(src_id: str, existing_names) -> str:
    dest_id = dest_id_for(src_id)
    if already_complete(src_id, dest_id, existing_names):
        return "skip"

    tmp = tempfile.mkdtemp(prefix="hfcopy_")
    try:
        with_retry("download", snapshot_download,
                   repo_id=src_id, repo_type="model",
                   local_dir=tmp, token=HF_TOKEN)
        with_retry("create", api.create_repo,
                   repo_id=dest_id, repo_type="model",
                   private=PRIVATE, exist_ok=True, token=HF_TOKEN)
        with_retry("upload", api.upload_folder,
                   folder_path=tmp, repo_id=dest_id, repo_type="model",
                   token=HF_TOKEN, commit_message=f"Archive mirror of {src_id}")
        return "copied"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    with open(REPOS_FILE) as f:
        repos = [ln.strip() for ln in f if ln.strip()]

    mine = [r for i, r in enumerate(repos) if i % NUM_SHARDS == SHARD]
    existing_names = fetch_existing_dest_names()
    print(f"[shard {SHARD}/{NUM_SHARDS}] {len(mine)} repos to process "
          f"(of {len(repos)} total) -> {DEST_USER} (private={PRIVATE}); "
          f"dest already has {len(existing_names) if existing_names is not None else '?'} repos",
          flush=True)

    copied = skipped = 0
    failed = []
    for n, src_id in enumerate(mine, 1):
        try:
            result = copy_one(src_id, existing_names)
            if result == "skip":
                skipped += 1
                print(f"[{n}/{len(mine)}] SKIP  {src_id} (already complete)", flush=True)
            else:
                copied += 1
                print(f"[{n}/{len(mine)}] OK    {src_id} -> {dest_id_for(src_id)}", flush=True)
        except GatedRepoError:
            print(f"[{n}/{len(mine)}] GATED {src_id} (cannot read) — skipping", flush=True)
            failed.append((src_id, "gated"))
        except Exception as e:
            print(f"[{n}/{len(mine)}] FAIL  {src_id}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.append((src_id, repr(e)))
        if BASE_DELAY:
            time.sleep(BASE_DELAY)

    print(f"\n[shard {SHARD}] DONE: copied={copied} skipped={skipped} failed={len(failed)}",
          flush=True)
    if failed:
        print(f"[shard {SHARD}] FAILED REPOS:", flush=True)
        for r, why in failed:
            print(f"  {r}\t{why}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
