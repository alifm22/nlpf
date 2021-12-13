import numpy as np
import torch
import torch.nn.functional as F

def generate_text(n_chars, net, dataset, initial_text="Hello", random_state=None):

    """Generate text with the character-level model.
    Parameters
    ----------
    n_chars : int
        Number of characters to generate.
    net : Module
        Character-level model.
    dataset : CharacterDataset
        Instance of the `CharacterDataset`.
    initial_text : str
        The starting text to be used as the initial condition for the model.
    random_state : None or int
        If not None, then the result is reproducible.
    Returns
    -------
    res : str
        Generated text.
    """
    if not initial_text:
        raise ValueError("You need to specify the initial text")

    res = initial_text
    net.eval()
    h, c = None, None

    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_chars):
        previous_chars = initial_text if res == initial_text else res[-1]
        features = torch.LongTensor([[dataset.ch2ix[c] for c in previous_chars]])
        logits, h, c = net(features, h, c)
        probs = F.softmax(logits[0], dim=0).detach().numpy()
        new_ch = np.random.choice(dataset.vocabulary, p=probs)
        res += new_ch

    return res