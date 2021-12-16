from numpy import character
from atoms.CharacterDataset import CharacterDataset
# from TempTest import MyTest
from torch.utils.data import Dataset

data = 'This is a text file. I changed the vocabulary size from 50 to 5 This is a text file.'
characterDataset = CharacterDataset(data)
# print(characterDataset.text)
print(f'character to index:\n {characterDataset.ch2ix}')
print(f'Index to character:\n {characterDataset.ix2ch}')
# print(characterDataset.__len__())

# characterDataset = MyTest(data)
# # print(f'character to index:\n {characterDataset.ch2ix}')
# print(f'Index to character:\n {characterDataset.ix2ch}')
