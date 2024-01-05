from typing import Tuple, List, Set
from collections import Counter
import numpy as np
import numpy.typing as npt

def bag_of_words_matrix(sentences: List[str],COUNT_THRESHOLD: int) -> npt.ArrayLike:
    """
    Convert the dataset into a V x M matrix.

    Parameters:
    - sentences (List[str]): List of sentences in the dataset.
    - COUNT_THRESHOLD (int): Threshold for filtering low-frequency words.

    Returns:
    - npt.ArrayLike: Bag-of-words matrix.

    Note:
    The function tokenizes the sentences, creates a vocabulary, and constructs a bag-of-words matrix.
    """
    token_list = [sentence.strip().split(" ") for sentence in sentences]
    token_list = [token.strip().lower() for token in sum(token_list,[])]

    word_dict = Counter(token_list)
    COUNT_THRESHOLD = 2
    word_dict_t = {"<UNK>":0}

    for word, count in word_dict.items():
        if count <= COUNT_THRESHOLD:
            word_dict_t["<UNK>"] += count
        else:
            word_dict_t[word] = count

    vocab = list(word_dict_t.keys())

    word_matrix = np.zeros(shape=(len(sentences),len(vocab)))

    for sent_index,sentence in enumerate(sentences):
        sent = [token if token in vocab else "<UNK>" for token in sentence.strip().split(" ")]
        sent_count_dict = Counter(sent)
        for vocab_index,vocab_word in enumerate(vocab):
            if vocab_word in sent_count_dict.keys():
                word_matrix[sent_index,vocab_index] = sent_count_dict[vocab_word]            
    word_matrix = word_matrix.T
    print(word_matrix.shape)
    return word_matrix

def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into a K x M matrix.

    Parameters:
    - data (Tuple[List[str], Set[str]]): Tuple containing a list of intent labels and a set of unique intent labels.

    Returns:
    - npt.ArrayLike: Labels matrix.

    Note:
    The function creates a labels matrix based on the provided intent labels and unique intent vocabulary.

    """
    intent = data[0]
    intent_vocab = data[1]

    label_matrix = np.zeros(shape=(len(intent),len(intent_vocab)))
    
    for index, label in enumerate(intent):
        for int_index, intent in enumerate(intent_vocab):
            if label == intent:
                label_matrix[index,int_index] = 1
    
    label_matrix = label_matrix.T
    print(label_matrix.shape)
    
    return label_matrix

def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.

    Parameters:
    - z (npt.ArrayLike): Input array.

    Returns:
    - npt.ArrayLike: Output array after applying the softmax function.

    """
    return np.divide(np.exp(z),np.sum(np.exp(z),axis=0))

def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.

    Parameters:
    - z (npt.ArrayLike): Input array.

    Returns:
    - npt.ArrayLike: Output array after applying the ReLU function.

    """
    return np.array([np.max([0,i]) for i in z])

def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.

    Parameters:
    - z (npt.ArrayLike): Input array.

    Returns:
    - npt.ArrayLike: Output array representing the first derivative of the ReLU function.

    """
    return np.array([1 if i > 0 else 0 for i in z])
