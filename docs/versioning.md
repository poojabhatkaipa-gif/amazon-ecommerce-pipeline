# Data versioning and lineage

**Tools:** Git (code, `.dvc` pointer files, `dvc.yaml`, `dvc.lock`) + DVC (data, models). Remote: `../amazon-ecommerce-dvc-remote` (local folder; swap for S3/GDrive with `dvc remote modify`).

## What is tracked

| Layer | How | Files |
|---|---|---|
| Raw archive | `dvc add` | `data/raw/amazon_ecommerce.zip.dvc` |
| Extracted CSV | `dvc add` | `data/extracted/amazon_ecommerce_1M.csv.dvc` |
| Raw lake (all ingest partitions) | `dvc add` | `data/lake/raw.dvc` |
| cleaned / transformed / warehouse / features / model | pipeline stages | `dvc.yaml` + `dvc.lock` |
| Feature versions | feature-store registry | `data/feature_store/registry.json` (every version: created_utc, source, hash, features) |
| Experiments | MLflow | `mlflow/` (run ids, params, metrics, feature-store versions) |

## Lineage (`dvc dag`)

```
extracted csv -> clean -> transform -> analytics (warehouse, figures)
                                    -> features (feature store) -> train (models, performance)
```

`dvc.lock` stores the md5 of every dependency and output of each stage, i.e. which exact input and code produced which output.
Metadata per ingestion run (source, date, files, status) is in `data/lake/raw/manifest_<run_ts>.json`; applied transformations are
documented in `docs/data_contract_person2.md` and `docs/feature_catalog.md`.

## Workflow

```bash
source .venv/bin/activate && export DVC_NO_ANALYTICS=1
dvc status                  # is anything stale / modified?
dvc repro                   # rebuild stale stages only
dvc dag                     # show lineage
git add dvc.yaml dvc.lock data/**/*.dvc && git commit -m "data v2"    # snapshot a version
dvc push                    # upload data to the remote
git checkout <old-commit> && dvc checkout                              # restore an earlier dataset version
dvc pull                    # on a fresh clone: download data from the remote
```

Data and logs are kept out of git: `.gitignore` excludes `logs/`, caches and sample outputs, and DVC writes ignore entries for every tracked path.
