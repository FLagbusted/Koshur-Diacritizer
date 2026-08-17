"""
LoRA fine-tunes IndicTrans2 (En -> Indic) on the prepared Kashmiri pairs.
Sized for a single RTX 4060 (8-16GB VRAM): LoRA + fp16 + small batch +
gradient accumulation, distilled 200M model by default.

IndicTrans2 loads via trust_remote_code (custom architecture, not a
stock HF model class), so this script does NOT hardcode LoRA target
module names from memory - it inspects the actual loaded model first
and prints what it found, so a wrong guess about internal layer naming
doesn't silently produce a no-op LoRA adapter. Check that printed list
before training starts.

Usage:
    python train_lora.py --train data/train.csv --val data/val.csv \
        --model ai4bharat/indictrans2-en-indic-dist-200M \
        --output ./kas-lora-adapter --epochs 3 --batch-size 4
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from IndicTransToolkit.processor import IndicProcessor


def find_lora_target_modules(model):
    """Inspect the actual model rather than assuming layer names, since
    IndicTrans2 is loaded via trust_remote_code with a non-standard
    architecture."""
    candidates = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
                  "q_lin", "k_lin", "v_lin", "out_lin",
                  "query", "key", "value", "dense"}
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in candidates:
                found.add(leaf)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="ai4bharat/indictrans2-en-indic-dist-200M")
    ap.add_argument("--output", default="./kas-lora-adapter")
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-lang", default="kas_Arab")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--full-finetune", action="store_true",
                     help="skip LoRA, fine-tune all weights (needs more VRAM, "
                          "only worth it at ~16GB+ with the 1B model)")
    ap.add_argument("--early-stopping-patience", type=int, default=2,
                     help="stop if val loss hasn't improved for this many epochs")
    ap.add_argument("--resume-from", default=None,
                     help="path to an existing LoRA adapter to warm-start from, "
                          "instead of initializing fresh LoRA weights")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. LoRA on CPU will be very slow for a "
              "corpus of this size - double-check `nvidia-smi` inside WSL first.")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if args.full_finetune:
        print("Full fine-tune mode: training all weights, no LoRA adapter.")
        for p in model.parameters():
            p.requires_grad = True
    elif args.resume_from:
        print(f"Resuming LoRA weights from {args.resume_from} (warm start, not from scratch)")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        model.print_trainable_parameters()
    else:
        target_modules = find_lora_target_modules(model)
        print(f"Auto-detected LoRA target modules: {target_modules}")
        if not target_modules:
            raise SystemExit(
                "No matching Linear layer names found automatically. Run "
                "`for n, m in model.named_modules(): print(n, type(m))` manually, "
                "find the attention/FFN Linear layers, and pass them explicitly "
                "via LoraConfig(target_modules=[...]) below."
            )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    ip = IndicProcessor(inference=False)

    ds = load_dataset(
        "csv", data_files={"train": args.train, "validation": args.val}
    )

    def preprocess(batch):
        src = ip.preprocess_batch(
            batch["english"], src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        model_inputs = tokenizer(
            src, truncation=True, max_length=args.max_length, padding=False
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["kashmiri"], truncation=True, max_length=args.max_length, padding=False
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")


if __name__ == "__main__":
    main()
