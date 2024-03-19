# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random


def sample_seqs(seqs: List[str], labels: List[bool], total_samples: int) -> Tuple[List[str], List[bool]]:
    """
    This function samples the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels
        total_samples: int
            Total number of samples to return

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    # Determine the class with fewer samples
    positive_seqs = [seq for seq, label in zip(seqs, labels) if label]
    negative_seqs = [seq for seq, label in zip(seqs, labels) if not label]
    
    num_positive = len(positive_seqs)
    num_negative = len(negative_seqs)

    if num_positive < num_negative:
        minority_seqs = positive_seqs
        majority_seqs = negative_seqs
    else:
        minority_seqs = negative_seqs
        majority_seqs = positive_seqs

    # Determine the length of the shortest sequence
    min_length = min(len(seq) for seq in seqs)

    # Calculate the number of samples from each class
    minority_samples = int(total_samples / 2)

    majority_samples = total_samples - minority_samples

    # Sample sequences from the minority class with replacement
    sampled_minority_seqs = [seq[start:start + min_length] for seq in random.choices(minority_seqs, k=minority_samples) for start in [random.randint(0, len(seq) - min_length)]]
    sampled_minority_labels = [1] * minority_samples if num_positive < num_negative else [0] * minority_samples

    # Sample sequences from the majority class with replacement
    sampled_majority_seqs = [seq[start:start + min_length] for seq in random.choices(majority_seqs, k=majority_samples) for start in [random.randint(0, len(seq) - min_length)]]
    sampled_majority_labels = [1] * majority_samples if num_positive >= num_negative else [0] * majority_samples

    # Combine sampled sequences and labels
    sampled_seqs = sampled_minority_seqs + sampled_majority_seqs
    sampled_labels = sampled_minority_labels + sampled_majority_labels

    # Shuffle the combined samples
    combined_samples = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined_samples)
    sampled_seqs, sampled_labels = zip(*combined_samples)

    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    return[encode_seq(seq) for seq in seq_arr]


    pass


def encode_base(base:str):
    """
    Encode a single DNA base into a one-hot encoded vector.

    Parameters:
        base (str): A single DNA base ('A', 'T', 'C', or 'G') to be encoded.

    Returns:
        List[int]: A one-hot encoded vector representing the input base.

    Raises:
        ValueError: If the input base is not one of the valid DNA bases ('A', 'T', 'C', or 'G').
    """
    # Establish encoding pattern
    base_dict={"A" : [1, 0, 0, 0],"T" : [0, 1, 0, 0],"C": [0, 0, 1, 0],"G": [0, 0, 0, 1]}
    #Raise error if unknown base
    if base not in base_dict.keys():
        raise ValueError("Invaild Base {}".format(base))

    # Reutn encoded base
    return base_dict[base]


def encode_seq(seq:List[str]):
    """
    Encode a sequence of DNA bases into a one-hot encoded vector.

    Parameters:
        seq (List[str]): A list of DNA bases ('A', 'T', 'C', or 'G') to be encoded.

    Returns:
        List[int]: A one-hot encoded vector representing the input sequence.

    """

    #Set up final array
    output=[]
    for base in seq:
        #Add encoded directly to end of array
        output=output+encode_base(base)
    
    return output
