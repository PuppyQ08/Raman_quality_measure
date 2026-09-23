# Source data: download, verification, and conversion

This repository does not redistribute third-party Raman spectra. The commands
below download the source archives from their official endpoints into the
Git-ignored `data/raw/` directory and check each file's exact size and SHA-256
before use.

| Dataset | Task in the paper | Source | Terms |
|---|---|---|---|
| Bacteria-ID (Ho et al., *Nat. Commun.* 2019) | bacterial isolate classification | authors' Dropbox link from <https://github.com/zhenyingok/bacteria-ID> | no explicit dataset license located; not mirrored |
| RRUFF (Lafuente et al., 2015) | mineral identification | eight Raman ZIPs from <https://www.rruff.net/about/download-data/> | free download; redistribution not established; not mirrored |
| Raman sugar mixtures (Georgiev et al.) | concentration quantification | Zenodo <https://doi.org/10.5281/zenodo.10779223> | CC BY 4.0 at the source |

`metadata/sources.json` is the registry of official URLs, local relative paths,
byte counts, SHA-256 values, and license status. It contains no credentials or
mirrored data. Review each provider's terms and citation requirements before
downloading.

## Install

```bash
python -m venv .venv
./.venv/bin/pip install -r env/requirements.lock
./.venv/bin/pip install -r env/phase3-requirements.lock
./.venv/bin/pip install -r env/data-rebuild-requirements.lock
```

The download scripts use only the Python standard library. The additional
packages support dataset inspection and conversion.

## Inspect the registry without downloading

```bash
./.venv/bin/python scripts/data/fetch_rruff.py --list
./.venv/bin/python scripts/data/fetch_bacteria_id.py --list
./.venv/bin/python scripts/data/fetch_sugar_mixtures.py --list
```

## Download and verify

`--accept-source-terms` records your explicit choice on the command line; it
does not change or grant any upstream rights.

```bash
./.venv/bin/python scripts/data/fetch_rruff.py \
  --download --accept-source-terms --data-root data/raw

./.venv/bin/python scripts/data/fetch_bacteria_id.py \
  --download --accept-source-terms --data-root data/raw

./.venv/bin/python scripts/data/fetch_sugar_mixtures.py \
  --download --accept-source-terms --data-root data/raw
```

RRUFF downloads are spaced by two seconds by default. Use `--artifact <id>` to
download one archive at a time. Each download is written to a `.part` file and
kept only after all registered hashes pass. The Sugar archive (about 5.3 GB) is
also checked against Zenodo's MD5.

To verify files that are already present, without network access:

```bash
./.venv/bin/python scripts/data/fetch_rruff.py --verify-only --data-root data/raw
./.venv/bin/python scripts/data/fetch_bacteria_id.py --verify-only --data-root data/raw
./.venv/bin/python scripts/data/fetch_sugar_mixtures.py --verify-only --data-root data/raw
```

## Build the unified RRUFF and Bacteria-ID stores

```bash
MPLCONFIGDIR=/tmp/rpe-mpl ./.venv/bin/python tools/build_rruff_unified.py \
  --raw-root data/raw/rruff \
  --output-root data/unified

MPLCONFIGDIR=/tmp/rpe-mpl ./.venv/bin/python tools/build_bacteria_id_unified.py \
  --raw-root data/raw/bacteria_id \
  --output-root data/unified
```

Both converters work offline and stop if the source inventory or content hashes
differ from the frozen contract. The unified schema is defined in
`rpe/io/schema.py` and `rpe/io/store.py`.

## Check the Sugar archive through RamanBench

The Sugar archive is placed where `raman-data==1.2.6` expects it:
`data/raw/ramanbench/cache/10779223/Raw data.zip`.

```bash
MPLCONFIGDIR=/tmp/rpe-mpl ./.venv/bin/python -m tools.audit_ramanbench \
  --one sugar_mixtures_high_snr \
  --raw-root data/raw/ramanbench

MPLCONFIGDIR=/tmp/rpe-mpl ./.venv/bin/python -m tools.audit_ramanbench \
  --one sugar_mixtures_low_snr \
  --raw-root data/raw/ramanbench
```

The paper uses the low-SNR acquisitions. The Sugar cohort (240 wells, 32
acquisitions each, well-grouped five-fold splits) is built inside the task
runners from `rpe/io/sugar_mixtures*.py`, with the frozen cohort definition in
`experiments/phase05/configs/d4_sugar_protocol.json`.

## From source data to results

After the stores exist, the pipeline stages in the main
[README](../README.md#reproducing-the-experiments-from-source-data) produce the
aggregate CSVs in `paper/data/` and `reports/robustness/w1_axis/`. Each runner
checks the SHA-256 of its frozen configuration and parent artifacts, so its
outputs can be compared directly with the published aggregates.
