# artifacts/

Adapters, fitted maps, latent models, cached spectral summaries and locally built tiny
bases are written here. The directory is gitignored (except this file): artifacts are
large and are regenerable from configs + manifests. Store canonical Stage-0 artifacts
externally (object storage) and record their location in the run manifests.

Layout produced by the pipeline:

```
artifacts/
├── bases/<base_name>/            # locally built tiny bases (smoke tests only)
├── runs/<run_id>/                # one LoRA run
│   ├── adapter/                  # PEFT adapter (safetensors) of the best checkpoint
│   ├── manifest.yaml             # run manifest (spec §11)
│   ├── train_log.jsonl
│   ├── delta_verification.json   # ΔW reconstruction vs PEFT forward check
│   └── spectral/                 # compact exact low-rank SVD factors + summary json
├── maps/<experiment>/<map_id>/   # fitted pair maps (+ train/holdout task IDs)
└── predictions/<experiment>/...  # predicted target updates (factored)
```
