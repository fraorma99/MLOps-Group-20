import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mlops_group_20.model import LanguageClassifier
from mlops_group_20.data import Vocabulary, simple_tokenizer
import mlops_group_20.train as train_module


# -----------------------------------------------------------------------------------------
# Fixtures to develop a test vocab, dummy inputs and a small model to conduct the tests
# -----------------------------------------------------------------------------------------

@pytest.fixture
def small_vocab():
    """ Create a Tiny Vocabulary instance for testing purposes. """
    vocab = Vocabulary(min_freq=1)
    toy_texts = [
        simple_tokenizer("hello"),
        simple_tokenizer("world"),
        simple_tokenizer("hello world"),
    ]
    vocab.build(toy_texts)
    return vocab


@pytest.fixture
def small_inputs(small_vocab):
    """ Construct a batch of words with its indices """
    texts = ["hello", "world", "hello"]
    max_len = 5

    def encode(text: str) -> torch.Tensor:
        tokens = simple_tokenizer(text)[:max_len]
        idxs = [small_vocab[token] for token in tokens]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))
        return torch.tensor(idxs, dtype=torch.long)

    xs = [encode(t) for t in texts]
    batch_x = torch.stack(xs, dim=0)
    return batch_x


@pytest.fixture
def small_labels():
    """ Labels for the small batch """
    return torch.tensor([0, 1, 0], dtype=torch.long)


@pytest.fixture
def small_model(small_vocab):
    """ Reduced dimensions LanguageClassifier model to run tests faster """
    vocab_size = len(small_vocab)
    num_classes = 3  # arbitrary small number of classes for the tests

    model = LanguageClassifier(
        vocab_size=vocab_size,
        embed_dim=16,
        hidden_dim=32,
        num_classes=num_classes,
        num_layers=1,
        dropout=0.1,
    )
    return model


# ---------------------------------------------------------------------------
# Model tests (LanguageClassifier)
# ---------------------------------------------------------------------------

def test_model_forward_shape(small_model, small_inputs):
    """ Check LanguageClassifier forward pass returns a tensor with the expected shape """
    model = small_model
    x = small_inputs
    outputs = model(x)

    batch_size = x.size(0)
    num_classes = model.fc.out_features

    assert outputs.shape == (batch_size, num_classes)


def test_model_outputs(small_model, small_inputs):
    """ Model gives a valid output type and no constant values """
    model = small_model
    x = small_inputs
    outputs = model(x)

    assert outputs.dtype.is_floating_point

    probs = torch.softmax(outputs, dim=1)
    assert torch.var(probs) > 0.0


def test_language_classifier_backward_pass(small_model, small_inputs, small_labels):
    """ Complete test of the model computing loss and optimizing weights """
    model = small_model
    x = small_inputs
    y = small_labels

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# Training smoke test (train.train with a dummy cfg and tiny data)
# ---------------------------------------------------------------------------

class DummyCfg:
    """ Minimal configuration class to train the model """

    class DataCfg:
        path = "data/processed/processed.pkl"
        splits_dir = "data/splits"
        seed = 0

        class SplitCfg:
            train_size = 0.7
            val_size = 0.2
            test_size = 0.1
            stratify_column = "Language"

        split = SplitCfg()
        vocab_min_freq = 1
        max_len = 10

    class TrainingCfg:
        batch_size = 4
        num_epochs = 1
        device = "cpu"
        save_best_to = "tests/checkpoints/best_model.pt"
        save_history_to = "tests/checkpoints/history.pt"

    class ModelCfg:
        embed_dim = 16
        hidden_dim = 32
        num_layers = 1
        dropout = 0.1

    class OptimizerCfg:
        name = "adam"
        lr = 1e-3
        weight_decay = 0.0

    class SchedulerCfg:
        name = "none"
        mode = "min"
        patience = 2
        factor = 0.5

    class WandbCfg:
        enabled = False
        project = "dummy"
        entity = None
        run_name = None
        notes = None
        tags = []
        watch_model = False
        log_freq = 10

    data = DataCfg()
    training = TrainingCfg()
    model = ModelCfg()
    optimizer = OptimizerCfg()
    scheduler = SchedulerCfg()
    wandb = WandbCfg()



def test_train_smoke_one_epoch(monkeypatch, tmp_path):
    """
    Smoke test: verify that train(cfg) can run one epoch without errors.

    Important:
    - This test does NOT touch LanguageDataset or TextDataset directly,
      those are already tested in test_data.py.
    - It builds a tiny DataFrame with columns 'Text' and 'Language'.
    - It monkeypatches pandas.read_pickle inside train_module so that
      train() reads our in-memory DataFrame instead of a real file.
    """
    import pandas as pd

    # Small in-memory dataset, similar to what train.py expects.
    df = pd.DataFrame(
    {
        "Text": [
            "hello one", "hello two", "hello three", "hello four", "hello five",
            "hola uno", "hola dos", "hola tres", "hola cuatro", "hola cinco",
        ],
        "Language": [
            "en", "en", "en", "en", "en",
            "es", "es", "es", "es", "es",
        ],
    }
)


    # Configure paths to point inside tmp_path.
    cfg = DummyCfg()
    cfg.data.path = str(tmp_path / "dummy_processed.pkl")
    cfg.data.splits_dir = str(tmp_path / "splits")
    cfg.training.save_best_to = str(tmp_path / "models" / "best_model.pt")
    cfg.training.save_history_to = str(tmp_path / "models" / "history.pt")

    # Save the DataFrame once, then intercept read_pickle.
    df.to_pickle(cfg.data.path)

    original_read_pickle = pd.read_pickle

    def fake_read_pickle(path, *args, **kwargs):
        path = str(path)
        if path == cfg.data.path:
            return df
        return original_read_pickle(path, *args, **kwargs)

    monkeypatch.setattr(train_module.pd, "read_pickle", fake_read_pickle)

    # Make sure wandb calls are no-ops even if something toggles it on.
    dummy_wandb = types.SimpleNamespace(
        init=lambda *a, **k: None,
        watch=lambda *a, **k: None,
        log=lambda *a, **k: None,
        finish=lambda *a, **k: None,
        Artifact=lambda *a, **k: types.SimpleNamespace(add_file=lambda *aa, **kk: None),
        log_artifact=lambda *a, **k: None,
    )
    monkeypatch.setattr(train_module, "wandb", dummy_wandb)

    # The test passes if no exception is raised during train().
    train_module.train(cfg)
