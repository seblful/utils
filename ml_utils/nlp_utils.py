from typing import List
from collections import Counter

import torch
import torch.nn.functional as F


def texts_to_one_hot(dict_size: int,
                     sent_size: int,
                     texts_corpus: List) -> List:
    """
    Takes corpus with texts and creates
    list with one hot encoded vectors with size - 
    (sent_size, dict_size)
    """
    # Create list of all words and count it
    words = [word for sentence in texts_corpus for word in sentence.split()]
    word_counts = Counter(words)

    # Create a vocabulary from the word tokens
    vocab = list(set(words))
    sorted_vocab = sorted(
        vocab, key=lambda x: word_counts[x], reverse=True)
    sorted_vocab = sorted_vocab[:dict_size - 2]

    # Create a dictionary to map words to indices
    word_to_idx = {word: i for i, word in enumerate(sorted_vocab, start=1)}

    # Convert the tokens to their corresponding indices
    indexed_data = [[word_to_idx.get(word, dict_size - 1)
                     for word in sentence.split()] for sentence in texts_corpus]

    one_hot_corpus = []

    for text in indexed_data:
        # Perform one-hot encoding using the one_hot function
        one_hot_encoded = torch.zeros(sent_size, dict_size)
        one_hot = F.one_hot(torch.tensor(
            text), num_classes=dict_size)

        slice_shape = min(one_hot.shape[0], sent_size)
        one_hot_encoded[:slice_shape] = one_hot[:slice_shape]

        one_hot_corpus.append(one_hot_encoded)

    return one_hot_corpus
