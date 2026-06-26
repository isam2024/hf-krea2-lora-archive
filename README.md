# HF → HF LoRA archiver

Mirrors the 999 Krea-2 style LoRAs (plus the index repo) from
[`ilkerzgi`](https://huggingface.co/ilkerzgi) to the `k2styles` account, using
**GitHub Actions runners as the intermediary** — the data never touches a local
machine. Each runner downloads a source repo and re-uploads it to the
destination, deleting local files after every repo so disk never fills.

## How it works

- `repos.txt` — the 1000 source repo ids to mirror.
- `mirror.py` — copies one shard of the list (download → create dest repo → upload).
  Idempotent: skips any repo the destination already has in full.
- `.github/workflows/archive.yml` — fans the list out across N parallel shards.

## Run it

1. Set the destination token as an **encrypted secret** named `HF_TOKEN`
   (a write token for the destination account).
2. Set a repo **variable** `DEST_USER` to the destination namespace (`k2styles`).
3. Actions tab → **Archive LoRAs (HF -> HF)** → **Run workflow**
   (defaults: 20 shards, public destination repos).

Re-running is safe — it only copies what's missing or incomplete.
