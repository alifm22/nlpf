import numpy as np

# class LossFunction:
def compute_loss(cal, net, dataloader):
    """Computer average loss over a dataset."""
    net.eval()
    all_losses = []
    for X_batch, y_batch in dataloader:
        probs, _, _ = net(X_batch)

        all_losses.append(cal(probs, y_batch).item())

    return np.mean(all_losses)