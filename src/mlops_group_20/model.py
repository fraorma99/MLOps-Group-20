from torch import nn
import torch


class LanguageClassifier(nn.Module):
    """LSTM-based language detection model."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_classes: int = 17, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        #Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  #(batch_size, hidden_dim*2)
        output = self.dropout(hidden_cat)
        output = self.fc(output)  #(batch_size, num_classes)
        return output


class Model(nn.Module):
    """Alias for backward compatibility"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


if __name__ == "__main__":
    #Test the language classifier
    model = LanguageClassifier(vocab_size=10000, num_classes=17)
    x = torch.randint(0, 1000, (32, 100))  #Batch of 32, sequence length 100
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
