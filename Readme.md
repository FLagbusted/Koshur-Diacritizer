# Koshur Diacritizer — KATHE 2026 Submission

An English → Kashmiri (Perso-Arabic) translation pipeline built for
[KATHE 2026](https://www.kaggle.com/competitions/kathe-2026), a national
translation challenge run by Gaash Lab (NIT Srinagar) with BIS and the
University of Kashmir.

**Status: training in progress.** Current best checkpoint below; will keep
updating as training continues.

## Results so far

| Run | Steps/Epochs | eval_loss |
|---|---|---|
| Initial LoRA fine-tune | 3 epochs | 2.183 |
| Resumed fine-tune (`--resume-from`) | 8 epochs | **1.840** |

Best adapter currently in repo: [`kas-lora-adapter-v2-best/`](./kas-lora-adapter-v2-best)

## Approach

Rather than training a translation model from scratch on a low-resource
language pair, this pipeline fine-tunes an existing pretrained model and
adds a diacritization post-processing pass:

1. **Base model**: [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
   (`ai4bharat/indictrans2-en-indic-dist-200M`), which already supports
   English → Kashmiri (`kas_Arab`) as one of its 22 languages, pretrained
   on AI4Bharat's [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC)
   corpus.
2. **Fine-tuning**: LoRA adapter trained on the Kashmiri-tagged subset of
   BPCC (`kas_Arab` files pulled from the `daily`/`wiki`/`ilci`/`massive`
   splits, not the full 107GB multilingual dataset).
3. **Post-processing**: output is run through the public
   [Koshur Diacritizer](https://huggingface.co/Omarrran/koshur-diacritizer-byt5-small)
   checkpoint (ByT5-based) to clean up/restore diacritics, with an
   explicit strip-then-diacritize normalization step (translation output
   already carries some diacritics from training data; feeding
   partially-diacritized text into the diacritizer without stripping
   first causes duplicate mark-stacking artifacts).

## Pipeline

```
download_kashmiri_bpcc.py   -> pulls only Kashmiri-tagged files from BPCC
prepare_data.py             -> merges into deduplicated train/val CSVs,
                                auto-detecting English vs Kashmiri columns
                                by script rather than assumed position
train_lora.py                -> LoRA fine-tunes IndicTrans2 on the prepared
                                data, with early stopping on val loss
infer_lora.py                -> translates the competition test set with the
                                trained adapter, then diacritizes the output
```

## Setup

```bash
python3 -m venv kathe-train-env
source kathe-train-env/bin/activate
pip install -r requirements_train.txt
```

## Usage

```bash
# 1. Pull Kashmiri-only data from BPCC (requires accepting the dataset's
#    access terms at huggingface.co/datasets/ai4bharat/BPCC first)
python download_kashmiri_bpcc.py --out ./bpcc_kashmiri

# 2. Build train/val splits
python prepare_data.py --in-dir ./bpcc_kashmiri --out-dir ./data

# 3. Train
python train_lora.py --train data/train.csv --val data/val.csv \
  --model ai4bharat/indictrans2-en-indic-dist-200M \
  --output ./kas-lora-adapter --batch-size 4 --grad-accum 8 \
  --epochs 40 --early-stopping-patience 4

# 4. Translate + diacritize
python infer_lora.py --input englishdev.csv --id-col ID --text-col sentence \
  --base-model ai4bharat/indictrans2-en-indic-dist-200M \
  --adapter ./kas-lora-adapter-v2-best --output submission.csv --diacritize
```

## Data & Model Attribution

- **BPCC** (Bharat Parallel Corpus Collection) — AI4Bharat, used under its
  Hugging Face dataset terms.
- **IndicTrans2** — AI4Bharat, MIT License.
  [Paper](https://arxiv.org/abs/2305.16307)
- **Koshur Diacritizer** — Haq Nawaz Malik, Nahfid Nissar, Faizan Iqbal.
  Model: `Omarrran/koshur-diacritizer-byt5-small`.
  [Paper](https://arxiv.org/abs/2606.15883) — see the original model card
  for license terms.

## Known limitations (as of last training run)

- Occasional script leakage (rare Devanagari fragments in otherwise
  Perso-Arabic output) — evidence of an undertrained model on a small
  fine-tuning set.
- Weak handling of numbers/currency in some sentences (e.g. large
  rupee amounts occasionally collapse or corrupt) — likely due to limited
  numeric examples in the small Kashmiri-tagged BPCC subset used so far.
- Diacritizer stacking bug (fixed): resolved by stripping any
  pre-existing diacritics from translation output before the
  diacritization pass, since the diacritizer expects bare undiacritized
  input.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Data and pretrained model weights retain their original upstream licenses
as noted above.
