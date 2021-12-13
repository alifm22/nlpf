import torch
from torch.nn import Embedding, Linear, LSTM, Module

class CharacterEmbedding(Module):
    """Custom network predicting the next character of a string.
    Parameters
    ----------
    vocab_size : int
        The number of charactesr in the vocabulary.
    embedding_dim : int
        Dimension of the character embedding vectors.
    dense_dim : int
        Number of neurons in the linear layer that follows the LSTM.
    hidden_dim : int
        Size of the LSTM hidden state.
    max_norm : int
        If any of the embedding vectors has a higher L2 norm than `max_norm`
        it is rescaled.
    n_layers : int
        Number of the layers of the LSTM.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=2,
        dense_dim=32,
        hidden_dim=8,
        max_norm=2,
        n_layers=1,
    ):
        super().__init__()

        self.embedding = Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=vocab_size - 1,
                norm_type=2,
                max_norm=max_norm,
        )
        self.lstm = LSTM(
                embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers
        )
        self.linear_1 = Linear(hidden_dim, dense_dim)
        self.linear_2 = Linear(dense_dim, vocab_size)


    def forward(self, x, h=None, c=None):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, window_size)` of dtype
            `torch.int64`.
        h, c : torch.Tensor or None
            Hidden states of the LSTM.
        Returns
        -------
        logits : torch.Tensor
            Tensor of shape `(n_samples, vocab_size)`.
        h, c : torch.Tensor or None
            Hidden states of the LSTM.
        """
        emb = self.embedding(x)  # (n_samples, window_size, embedding_dim)
        if h is not None and c is not None:
            _, (h, c) = self.lstm(emb, (h, c))
        else:
            _, (h, c) = self.lstm(emb)  # (n_layers, n_samples, hidden_dim)

        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)
        x = self.linear_1(h_mean)  # (n_samples, dense_dim)
        logits = self.linear_2(x)  # (n_samples, vocab_size)

        return logits, h, c