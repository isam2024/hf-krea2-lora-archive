# HF LoRA archiver → single repo

Copies the 999 Krea-2 style LoRAs from
[`ilkerzgi`](https://huggingface.co/ilkerzgi) into **one** destination repo
(`k2styles/krea-2-style-loras-archive`), each as a subfolder with its full
contents (weights, README, preview images). Uses **GitHub Actions runners as
the intermediary** — the data never touches a local machine. Downloads are
batched so several LoRAs go up per commit.

## How it works

- `repos.txt` — the source repo ids (the index repo is skipped via `SKIP_REPOS`).
- `mirror.py` — for its shard, downloads each LoRA into `staging/<name>/` and
  uploads the batch into the archive repo in one commit. Idempotent: skips any
  LoRA whose subfolder already exists in the archive.
- `.github/workflows/archive.yml` — shards the list and runs them **sequentially**
  (`max-parallel: 1`) since all commits target the same repo.

## Run it

1. Encrypted secret `HF_TOKEN` — a read+write token for the destination account.
2. Repo variable `ARCHIVE_REPO` — the single destination repo id.
3. Repo variable `SKIP_REPOS` — source ids to skip (the index repo).
4. Actions tab → **Archive LoRAs (HF -> single repo)** → **Run workflow**.

Re-running is safe — it only adds LoRAs not already in the archive.
