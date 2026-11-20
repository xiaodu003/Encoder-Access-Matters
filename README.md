# Paper Code README

Core code for evaluating text missingness in multimodal sentiment analysis. PRE blocks selected content before frozen BERT encoding. POST encodes the full text and then removes the same positions. Both conditions retain the same downstream mask, position indices, and audio and visual inputs.

## Code map

| Directory | Role |
| --- | --- |
| `data/` | Dataset and tokenizer identity, PRE/POST encoding, training-only normalization, deletion schedules and packed caches |
| `models/` | Text GRU, fusion GRU, visibility-rate fusion and ALMT-FEATURE |
| `training/` | Validation-selected training, early stopping, AMP checks and epoch checkpoints |
| `evaluation/` | Clean/PRE/POST, alternate masking operators, new masks, contiguous blocks and deleted-ID sensitivity |
| `analysis/` | Per-example pairing, source-cluster bootstrap, seed summaries and model rankings |
| `checks/` | Synthetic checks without external data or pretrained weights |

## Setup

Use Python 3.13 and a CUDA-enabled PyTorch build for full runs. The recorded environment in `requirements.txt` includes PyTorch 2.11.0 with CUDA 12.8. Its full notebook environment listing contains platform-specific packages. For a fresh environment, install matching PyTorch and CUDA support for the host plus NumPy, pandas, Transformers, tqdm and PyYAML. CPU runs work for small checks.

Provide the released MMSA PKL and label CSV files for MOSI, CH-SIMS and MOSEI. Supply local Hugging Face BERT model and tokenizer directories. MOSI and MOSEI use English `bert-base-uncased`, and CH-SIMS uses Chinese `bert-base-chinese`. Edit `runtime.dataset_paths` and `runtime.bert_paths` in `config.yaml` for local paths. Use trusted copies of the dataset PKLs and checkpoints.

## Training seeds

Each dataset follows the training matrix reported in the paper.

| Model | Training conditions | Training seeds per condition | Runs per dataset |
| --- | --- | --- | ---: |
| `B0_TEXT` | POST | 2026, 2027, 2028 | 3 |
| `B1_FUSION` | CLEAN, POST, PRE, MIX | 2026, 2027, 2028 | 12 |
| `B2_RATE` | POST, PRE | 2026, 2027, 2028 | 6 |
| `ALMT_FEATURE` | POST, PRE | 2026, 2027, 2028 | 6 |

Each configuration uses the three training seeds listed under `experiments`. This gives 27 checkpoints per dataset and 81 in total.

`runtime.mask_seed` sets the shared standard deletion masks (2026). `evaluation.new_mask_seed` sets the new-mask and contiguous controls (4311). The eight masks per deletion rate are separate from the number of training seeds.

## Run

```sh
python run.py selfcheck
python run.py train
python run.py evaluate
python run.py analyze
python run.py audit-access
```

- Set `datasets` and the model entries under `experiments` to run a subset.
- `CLEAN` trains on complete text. `POST`, `PRE` and `MIX` mix clean examples with their specified deletion conditions. The 20% and 50% rates each use eight paired masks. The selected checkpoint minimizes the equal-weight mean of Clean, PRE and POST validation MAE.
- `evaluate` reads selected checkpoints. Evaluation defaults to validation. For a final test pass, fix the training matrix and evaluation settings first, then set `evaluation.split` to `test` and `evaluation.allow_test` to `true`. The evaluation writes checkpoint hashes and produces predictions without checkpoint selection.
- `audit-access` uses the validation split and reports how replacing deleted IDs affects retained text states and downstream predictions. The original labels are used only for the unmodified PRE–POST error comparison. To match the published diagnostic coverage, point `--config` to a copy of the configuration whose `experiments` selects `B0_TEXT`, `B1_FUSION`, and `ALMT_FEATURE`, each with `arms: [POST]` and `seeds: [2026]`. Keep the same output root to reuse those selected checkpoints.
- `analyze` reads predictions for the selected split and suites. Results include per-seed scores and paired effects. In the pooled position analysis, source resampling follows within-sample averaging across training seeds. The seed standard deviation is reported separately.
- Outputs go to `runtime.output_root`. Use a fresh output directory when changing the scientific protocol. Cache and checkpoint identity checks guard reuse. The public runner reruns the protocol with its own deterministic masks and initialization, so its caches and checkpoints are specific to this code layout.

The ALMT adaptation retains the upstream [MIT notice](vendor/almt/LICENSE).
