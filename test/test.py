from collections import Counter ,defaultdict
import torch
vocab_size = 5
text = 'This is a text file. I changed the vocabulary size from 50 to 5 This is a text file.'

most_common_ch2ix = {}
ch2ix = defaultdict(lambda: vocab_size - 1)
for i, x in enumerate(Counter(text).most_common()[: (vocab_size - 1)]):
    most_common_ch2ix.update({x[0]: i}) # x is a tuple (character, frequency)


ch2ix.update(most_common_ch2ix)
ch2ix["~"] = vocab_size - 1


ix2ch = {v: k for k, v in ch2ix.items()}  # ch2ix.items() -> character, index pair
vocabulary = [ix2ch[i] for i in range(vocab_size)]

window_size=1

# X = torch.LongTensor(
#     [ch2ix[c] for c in text[ix : ix + window_size]]
# )
# y = ch2ix[text[ix + window_size]]

ix = 0
for c in text[ix:ix+window_size]:
    print(c)
    print(ch2ix[c])
# def getItem(ix=3):
#     X = torch.LongTensor(
#         [ch2ix[c] for c in text[ix : ix + window_size]]
#     )
#     y = ch2ix[text[ix + window_size]]

#     return X, y

# print(getItem())