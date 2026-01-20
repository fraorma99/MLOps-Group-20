# monitoring/data_drifting.py

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import simple_tokenizer

from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.report import Report


def load_artifacts(device):
    """Load model + vocab + label mappings from local files."""
    label_info = pd.read_pickle("data/splits/label_mappings.pkl")
    idx2label = label_info["idx2label"]
    label2idx = {v: k for k, v in idx2label.items()}

    vocab = pd.read_pickle("data/splits/vocab.pkl")
    checkpoint = torch.load("models/best_model.pt", map_location=device)

    model = LanguageClassifier(
        vocab_size=checkpoint["vocab_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, idx2label, label2idx


def encode(text: str, vocab: dict, max_len: int = 200) -> tuple[list[int], float]:
    """
    Tokenize and map tokens to indices.
    Unknown tokens -> 0 (so inference is robust to drift), and we track OOV rate.
    """
    tokens = simple_tokenizer(text)[:max_len]
    oov = 0
    idxs = []
    for t in tokens:
        try:
            idx = vocab[t]  # works for dict-like or Vocabulary objects
            idxs.append(int(idx))
        except Exception:
            idxs.append(0)
            oov += 1


    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))

    oov_rate = oov / max(1, len(tokens))
    return idxs, oov_rate


@torch.no_grad()
def predict_texts(model, vocab, texts, device, batch_size: int = 64):
    """
    Batched prediction for speed + a simple progress indicator.
    """
    all_idxs = []
    oov_rates = []
    for txt in texts:
        idxs, oov_rate = encode(txt, vocab)
        all_idxs.append(idxs)
        oov_rates.append(oov_rate)

    preds = []
    n = len(all_idxs)
    for i in range(0, n, batch_size):
        batch = all_idxs[i : i + batch_size]
        x = torch.tensor(batch, dtype=torch.long).to(device)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        print(f"processed {min(i+batch_size, n)}/{n}", end="\r", flush=True)
    print()
    return preds, oov_rates


# -------- Drift scenarios (language detection) --------
def drift_short(text: str, max_chars: int = 25) -> str:
    return text[:max_chars]


def drift_typos(text: str, p: float = 0.08) -> str:
    rng = random.random
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha() and rng() < p:
            base = "a" if ch.islower() else "A"
            offset = (ord(ch) - ord(base) + 1) % 26
            chars[i] = chr(ord(base) + offset)
    return "".join(chars)


def drift_emojis_urls(text: str) -> str:
    return f"{text}  https://example.com #mlops"


def drift_codeswitch(text: str) -> str:
    return f"{text} gracias amigo"


SCENARIOS = {
    "baseline": lambda x: x,
    "short": drift_short,
    "typos": drift_typos,
    "emoji_url": drift_emojis_urls,
    "codeswitch": drift_codeswitch,
}


def make_evidently_df(texts, y_true, y_pred, oov_rate):
    """
    Evidently dataframe.
    - content: text
    - target: true label (int index)
    - prediction: predicted label (int index)
    Plus numeric features that are useful for drift detection on text.
    """
    return pd.DataFrame(
        {
            "content": texts,
            "target": y_true,
            "prediction": y_pred,
            "oov_rate": oov_rate,
            "char_len": [len(t) for t in texts],
            "digit_frac": [sum(ch.isdigit() for ch in t) / max(1, len(t)) for t in texts],
            "non_ascii_frac": [sum(ord(ch) > 127 for ch in t) / max(1, len(t)) for t in texts],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with Text + Language columns.")
    parser.add_argument("--n", type=int, default=200, help="How many rows to use (for speed).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_col", default="Text")
    parser.add_argument("--label_col", default="Language")
    args = parser.parse_args()

    random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}", flush=True)
    model, vocab, idx2label, label2idx = load_artifacts(device)
    print("Loaded model + artifacts", flush=True)

    df = pd.read_csv(args.data)
    df = df[[args.text_col, args.label_col]].dropna().head(args.n).copy()

    texts_raw = df[args.text_col].astype(str).tolist()
    labels_raw = df[args.label_col].astype(str).tolist()
    y_true = [label2idx.get(lbl, -1) for lbl in labels_raw]

    out_dir = Path("reports/data_drifting")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference = baseline (no drift)
    print("Building reference predictions (baseline)...", flush=True)
    y_pred_ref, oov_ref = predict_texts(model, vocab, texts_raw, device)
    reference_df = make_evidently_df(texts_raw, y_true, y_pred_ref, oov_ref)
    print("Reference ready", flush=True)

    out_rows = []

    for name, fn in SCENARIOS.items():
        print(f"=== Scenario: {name} ===", flush=True)

        texts = [fn(t) for t in texts_raw]
        y_pred, oov = predict_texts(model, vocab, texts, device)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        current_df = make_evidently_df(texts, y_true, y_pred, oov)

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        report_path = out_dir / f"evidently_{name}.html"
        report.save(str(report_path))
        print(f"Saved report: {report_path}", flush=True)

        out_rows.append(
            {
                "scenario": name,
                "accuracy": acc,
                "macro_f1": f1,
                "avg_oov_rate": float(sum(oov) / max(1, len(oov))),
                "report_html": str(report_path),
            }
        )

    out = pd.DataFrame(out_rows).sort_values("scenario")
    out_path = out_dir / "robustness_metrics.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}", flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
