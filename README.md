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

## Run

```sh
python run.py selfcheck
python run.py train
python run.py evaluate
python run.py analyze
python run.py audit-access
```

- Set `datasets` and the model entries under `experiments` to run a subset. The supplied configuration lists all 63 training runs.
- `CLEAN` trains on complete text. `POST`, `PRE` and `MIX` mix clean examples with their specified deletion conditions. The 20% and 50% rates each use eight paired masks. The selected checkpoint minimizes the equal-weight mean of Clean, PRE and POST validation MAE.
- `evaluate` reads selected checkpoints. Evaluation defaults to validation. For a final test pass, fix the training matrix and evaluation settings first, then set `evaluation.split` to `test` and `evaluation.allow_test` to `true`. The evaluation writes checkpoint hashes and produces predictions without checkpoint selection.
- `audit-access` uses the validation split and reports how replacing deleted IDs affects retained text states and downstream predictions. The original labels are used only for the unmodified PRE–POST error comparison.
- `analyze` reads predictions for the selected split and suites. Results include per-seed scores and paired effects. In the pooled position analysis, source resampling follows within-sample averaging across training seeds. The seed standard deviation is reported separately.
- Outputs go to `runtime.output_root`. Use a fresh output directory when changing the scientific protocol. Cache and checkpoint identity checks guard reuse. The public runner reruns the protocol with its own deterministic masks and initialization, so its caches and checkpoints are specific to this code layout.

The ALMT adaptation retains the upstream [MIT notice](vendor/almt/LICENSE).