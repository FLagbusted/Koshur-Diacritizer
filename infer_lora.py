"""
Same pipeline as translate.py, but loads your trained LoRA adapter on
top of the base IndicTrans2 model before translating.

Usage:
    python infer_lora.py --input data/test.csv --id-col id --text-col english \
        --base-model ai4bharat/indictrans2-en-indic-dist-200M \
        --adapter ./kas-lora-adapter \
        --output submission.csv

    # add diacritization post-processing pass:
    python infer_lora.py ... --diacritize
"""
import argparse
import unicodedata

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def strip_diacritics(text: str) -> str:
    """Remove Unicode combining marks (category Mn). The diacritizer
    expects bare undiacritized input; translation output may already
    carry some diacritics from training data, and feeding it
    partially-diacritized text causes mark-stacking artifacts instead
    of clean diacritization."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def load_translator(base_model_name: str, adapter_path: str):
    print(f"Loading base model {base_model_name} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, trust_remote_code=True
    ).to(DEVICE)

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # fold LoRA into base weights, faster inference
    model.eval()

    ip = IndicProcessor(inference=True)
    return tokenizer, model, ip


def translate_batch(sentences, tokenizer, model, ip, src_lang, tgt_lang, batch_size=8):
    outputs_all = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        batch = ip.preprocess_batch(chunk, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            batch, padding="longest", truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(
                generated.detach().cpu().tolist(), skip_special_tokens=True
            )
        decoded = ip.postprocess_batch(decoded, lang=tgt_lang)
        outputs_all.extend(decoded)
        print(f"  translated {min(i + batch_size, len(sentences))}/{len(sentences)}")
    return outputs_all


def diacritize_batch(sentences, batch_size=16):
    print("Loading Koshur Diacritizer for post-processing ...")
    tok = AutoTokenizer.from_pretrained("Omarrran/koshur-diacritizer-byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Omarrran/koshur-diacritizer-byt5-small"
    ).to(DEVICE)
    model.eval()

    outputs_all = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        inputs = tok(
            chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        # Diacritization only ADDS characters (marks) to existing base
        # letters, so real output is never shorter than input. Use each
        # batch's shortest real (non-pad) input length as a floor against
        # premature EOS/truncation.
        real_lengths = inputs["attention_mask"].sum(dim=1)
        min_new = max(1, int(real_lengths.min().item() * 0.9))

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_length=512,
                min_new_tokens=min_new,
                num_beams=4,
                repetition_penalty=1.05,  # ByT5 is byte-level: keep this mild,
                # no_repeat_ngram_size deliberately omitted - blocking repeated
                # BYTE n-grams can fall mid-character on multi-byte UTF-8
                # sequences (Kashmiri diacritics legitimately repeat), which
                # corrupts output into stray Latin-looking artifacts.
            )
        decoded = tok.batch_decode(generated, skip_special_tokens=True)
        outputs_all.extend(decoded)
        print(f"  diacritized {min(i + batch_size, len(sentences))}/{len(sentences)}")
    return outputs_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="english")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--base-model", default="ai4bharat/indictrans2-en-indic-dist-200M")
    ap.add_argument("--adapter", required=True, help="path to the trained LoRA adapter dir")
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-lang", default="kas_Arab")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--diacritize", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    sentences = df[args.text_col].astype(str).tolist()
    tokenizer, model, ip = load_translator(args.base_model, args.adapter)
    translations = translate_batch(
        sentences, tokenizer, model, ip, args.src_lang, args.tgt_lang, args.batch_size
    )

    # Save raw translations before diacritization, so if the diacritizer
    # pass needs redoing (e.g. bad decoding params) you don't have to
    # re-run the slower translation step.
    raw_path = args.output.rsplit(".", 1)[0] + "_raw.csv"
    pd.DataFrame({args.id_col: df[args.id_col], "translation": translations}).to_csv(
        raw_path, index=False
    )
    print(f"Saved pre-diacritization translations to {raw_path}")

    if args.diacritize:
        pre_marks = sum(
            1 for s in translations for ch in s if unicodedata.category(ch) == "Mn"
        )
        stripped = [strip_diacritics(s) for s in translations]
        print(
            f"Stripped {pre_marks:,} existing diacritic marks from translation "
            f"output before running the diacritizer (confirms whether IndicTrans2 "
            f"output already carried diacritics)."
        )
        translations = diacritize_batch(stripped)

    out_df = pd.DataFrame({args.id_col: df[args.id_col], "translation": translations})
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out_df)} rows)")
    print("Rename the output column to match Kaggle's exact expected name before submitting.")


if __name__ == "__main__":
    main()
