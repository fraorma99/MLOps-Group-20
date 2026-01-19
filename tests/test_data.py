import pandas as pd
import torch
from torch.utils.data import Dataset

from mlops_group_20.data import LanguageDataset, TextDataset

class DummyVocab(dict):
    """Minimal vocabulary for testing Vocabulary class of TextDataset."""
    def __init__(self, tokens):
        super().__init__({tok: i + 2 for i, tok in enumerate(tokens)})
        self.pad_id = 0
        self.unk_id = 1
        dict.__setitem__(self, "<pad>", self.pad_id)
        dict.__setitem__(self, "<unk>", self.unk_id)

    def __getitem__(self, token):

        return dict.get(self, token, self.unk_id)

def dummy_tokenizer(text: str):
    return text.split()

def test_languagedataset(tmp_path):
    """Correct load of CSV file and returns valid results."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "Text": ["hello world", "goodbye world", "hola mundo", "hej verden"],
            "Language": ["en", "en", "es", "da"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = LanguageDataset(str(csv_path))
    assert isinstance(ds, Dataset)
    assert len(ds) == len(df)

    sample = ds[0]
    assert sample is not None

    # LanguageDataset returns dict {"text": ..., "label": ...}
    if isinstance(sample, dict):
        assert "text" in sample
        assert "label" in sample

        assert isinstance(sample["text"], str)
        assert len(sample["text"]) > 0

        assert isinstance(sample["label"], str)
        assert sample["label"] in set(df["Language"])

        # all labels represented (in this tiny dummy dataset)
        labels = {ds[i]["label"] for i in range(len(ds))}
        assert labels == set(df["Language"])

    else:
        raise AssertionError(f"Unexpected sample type: {type(sample)} with value {sample}")

def test_languagedataset_missing_columns(tmp_path):
    """Fails when required columns are missing."""
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"NotText": ["a", "b"], "NotLanguage": ["x", "y"]}).to_csv(csv_path, index=False)

    try:
        _ = LanguageDataset(str(csv_path))
        assert False, "Expected failure when required columns are missing"
    except KeyError:
        pass

def test_textdataset():
    """Checks if TextDataset transforms text into tensor and its corresponding class label."""
    texts = ["hello world", "hello there"]
    labels = [0, 1]
    vocab = DummyVocab(tokens=["hello", "world", "there"])
    tokenizer = dummy_tokenizer

    ds = TextDataset(texts, labels, vocab, tokenizer)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2

    x, y = ds[0]

    assert x is not None
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 1
    assert x.numel() > 0

    assert y is not None
    assert isinstance(y, (int, torch.Tensor))
    assert int(y) in [0, 1]
