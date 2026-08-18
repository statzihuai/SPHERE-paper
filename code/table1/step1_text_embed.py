#!/usr/bin/env python3
"""
Encode the Dreaddit corpus into the GTR-base embedding space that Table 1's
text row is measured in.

The embedding space matters more than it looks. Table 1 reports utility on the
same 768-d vectors the inversion attack operates on — the mean-pooled T5
encoder output that vec2text's "gtr-base" corrector was trained to invert. A
sentence-transformers pipeline over the same GTR checkpoint produces a
different (post-projection, normalised) vector, and a probe trained on that
one scores ~0.845 rather than the published 0.795. Utility and privacy have to
be quoted on one matrix, so this step reproduces the embedder exactly:

  * every text truncated to 32 T5 tokens (the corrector's max sequence length,
    so the vector that is attacked is the vector that is scored),
  * padded to max_length, and
  * embedded with corr.inversion_trainer.call_embedding_model.

    python3 step1_text_embed.py [--out-dir DIR]

Inputs   HuggingFace andreagasparini/dreaddit (public, ~3.5k posts)
Outputs  data/table1/text/emb_{train,test}_raw.npy   (n, 768) float32
         data/table1/text/labels_{train,test}.npy    (n,)     int64
         data/table1/text/test_trunc_orig.json       truncated reference texts
Needs    vec2text 0.0.13 + transformers 4.44.2 (the .venv-vec2text environment)
"""
import _common as C

import argparse
import json

import numpy as np

DATASET_ID = "andreagasparini/dreaddit"
CORRECTOR_ID = "gtr-base"
MAX_T5 = 32          # corrector max sequence length
BATCH = 128


def load_dreaddit():
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID)
    tr_t = [x for x in ds["train"]["text"]]
    te_t = [x for x in ds["test"]["text"]]
    tr_y = np.array(ds["train"]["label"], np.int64)
    te_y = np.array(ds["test"]["label"], np.int64)
    return tr_t, tr_y, te_t, te_y


def truncate(tok, texts):
    """Re-render each text as the string its first 32 T5 tokens decode to.

    Truncating the text rather than the token tensor keeps the attacked string
    and the embedded string identical, which is what makes the inversion
    metrics comparable to the utility metrics.
    """
    out = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=MAX_T5).input_ids
        if ids and ids[-1] == tok.eos_token_id:
            ids = ids[:-1]
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def embed(corr, dev, texts, bs=BATCH):
    import torch
    tok = corr.embedder_tokenizer
    outs = []
    for i in range(0, len(texts), bs):
        ins = tok(texts[i:i + bs], return_tensors="pt", max_length=MAX_T5,
                  truncation=True, padding="max_length")
        ins = {k: v.to(dev) for k, v in ins.items()}
        with torch.no_grad():
            e = corr.inversion_trainer.call_embedding_model(
                input_ids=ins["input_ids"], attention_mask=ins["attention_mask"])
        outs.append(e.cpu().numpy().astype(np.float32))
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(C.DATA / "text"))
    args = ap.parse_args()
    out = C.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import vec2text
    corr = vec2text.load_pretrained_corrector(CORRECTOR_ID)
    dev = next(corr.model.parameters()).device
    print(f"corrector={CORRECTOR_ID} embedder="
          f"{type(corr.inversion_trainer.model.embedder).__name__} device={dev}", flush=True)

    tr_t, tr_y, te_t, te_y = load_dreaddit()
    tok = corr.embedder_tokenizer
    tr_t, te_t = truncate(tok, tr_t), truncate(tok, te_t)
    print(f"train={len(tr_t)} test={len(te_t)}", flush=True)

    Ztr, Zte = embed(corr, dev, tr_t), embed(corr, dev, te_t)
    print(f"train {Ztr.shape}  test {Zte.shape}", flush=True)

    np.save(out / "emb_train_raw.npy", Ztr)
    np.save(out / "emb_test_raw.npy", Zte)
    np.save(out / "labels_train.npy", tr_y)
    np.save(out / "labels_test.npy", te_y)
    with open(out / "test_trunc_orig.json", "w") as f:
        json.dump(te_t, f)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
