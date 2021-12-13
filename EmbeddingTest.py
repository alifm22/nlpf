from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn import Embedding, Linear, LSTM, Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

from atoms.CharacterDataset import CharacterDataset
from molecules.CharacterEmbedding import CharacterEmbedding
from atoms.LossFunction import compute_loss
from atoms.TextGenerator import generate_text

from nltk.corpus import brown

if __name__ == "__main__":
    # with open("text.txt", "r") as f:
    #     text = "\n".join(f.readlines())
    text = ""
    for i in brown.words():
        text += (i+" ")     

    # Hyperparameters model
    vocab_size = 70
    window_size = 10
    embedding_dim = 2
    hidden_dim = 16
    dense_dim = 32
    n_layers = 1
    max_norm = 2

    # Training config
    n_epochs = 25
    train_val_split = 0.8
    batch_size = 128
    random_state = 13

    torch.manual_seed(random_state)

    loss_f = torch.nn.CrossEntropyLoss()
    dataset = CharacterDataset(text, window_size=window_size, vocab_size=vocab_size)

    n_samples = len(dataset)
    split_ix = int(n_samples * train_val_split)

    train_indices, val_indices = np.arange(split_ix), np.arange(split_ix, n_samples)

    train_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
    )
    val_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
    )

    net = CharacterEmbedding(
            vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dense_dim=dense_dim,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
    )
    optimizer = torch.optim.Adam(
            net.parameters(),
            lr=1e-2,
    )

    emb_history = []

    for e in range(n_epochs + 1):
        net.train()
        for X_batch, y_batch in tqdm(train_dataloader):
            if e == 0:
                break
            optimizer.zero_grad()
            probs, _, _ = net(X_batch)
            loss = loss_f(probs, y_batch)
            loss.backward()

            optimizer.step()

        train_loss = compute_loss(loss_f, net, train_dataloader)
        val_loss = compute_loss(loss_f, net, val_dataloader)
        print(f"Epoch: {e}, {train_loss=:.3f}, {val_loss=:.3f}")

        # Generate one sentence
        initial_text = "I hope it works "
        generated_text = generate_text(
            100, net, dataset, initial_text=initial_text, random_state=random_state
        )
        print(generated_text)

        # Prepare DataFrame
        weights = net.embedding.weight.detach().clone().numpy()

        df = pd.DataFrame(weights, columns=[f"dim_{i}" for i in range(embedding_dim)])
        df["epoch"] = e
        df["character"] = dataset.vocabulary

        emb_history.append(df)

final_df = pd.concat(emb_history)
final_df.to_csv("res.csv", index=False)