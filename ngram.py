"""
N-gram Language Model

This module implements an n-gram language model for text generation and analysis.

Good reference:
Speech and Language Processing. Daniel Jurafsky & James H. Martin.
https://web.stanford.edu/~jurafsky/slp3/3.pdf

Example run:
python ngram.py
"""

import os
import itertools
from typing import List, Dict, Union, Iterator
import numpy as np

# -----------------------------------------------------------------------------
# Random number generation

class RNG:
    """
    A deterministic random number generator that mimics Python's random interface.
    This class provides a controlled, reproducible source of randomness.
    """

    def __init__(self, seed: int) -> None:
        """
        Initialize the random number generator.

        Args:
            seed (int): The seed for the random number generator.
        """
        self.state: int = seed

    def random_u32(self) -> int:
        """
        Generate a random 32-bit unsigned integer using the xorshift algorithm.

        Returns:
            int: A random 32-bit unsigned integer.
        """
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self) -> float:
        """
        Generate a random float in the range [0, 1).

        Returns:
            float: A random float between 0 (inclusive) and 1 (exclusive).
        """
        return (self.random_u32() >> 8) / 16777216.0

random = RNG(1337)

# -----------------------------------------------------------------------------
# Sampling from the model

def sample_discrete(probs: List[float], coinf: float) -> int:
    """
    Sample from a discrete probability distribution.

    Args:
        probs (List[float]): A list of probabilities representing the distribution.
        coinf (float): A random float in [0, 1) used for sampling.

    Returns:
        int: The index of the sampled item from the distribution.
    """
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors

# -----------------------------------------------------------------------------
# Models: n-gram model, and a fallback model that can use multiple n-gram models

class NgramModel:
    """
    An n-gram language model that predicts the probability of the next token
    given a context of n-1 previous tokens.
    """

    def __init__(self, vocab_size: int, seq_len: int, smoothing: float = 0.0) -> None:
        """
        Initialize the n-gram model.

        Args:
            vocab_size (int): The size of the vocabulary.
            seq_len (int): The length of the n-gram sequence.
            smoothing (float, optional): Smoothing parameter to handle unseen n-grams. Defaults to 0.0.
        """
        self.seq_len: int = seq_len
        self.vocab_size: int = vocab_size
        self.smoothing: float = smoothing
        # The parameters of this model: an n-dimensional array of counts
        self.counts: np.ndarray = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)
        # A buffer to store the uniform distribution, just to avoid creating it every time
        self.uniform: np.ndarray = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape: List[int]) -> None:
        """
        Update the model's counts based on an observed sequence of tokens.

        Args:
            tape (List[int]): A sequence of tokens to train on.
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        self.counts[tuple(tape)] += 1

    def get_counts(self, tape: List[int]) -> np.ndarray:
        """
        Get the counts for a given context.

        Args:
            tape (List[int]): A sequence of tokens representing the context.

        Returns:
            np.ndarray: The counts for the given context.
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        return self.counts[tuple(tape)]

    def __call__(self, tape: List[int]) -> np.ndarray:
        """
        Calculate the conditional probability distribution of the next token.

        Args:
            tape (List[int]): A sequence of tokens representing the context.

        Returns:
            np.ndarray: The probability distribution for the next token.
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # Get the counts, apply smoothing, and normalize to get the probabilities
        counts = self.counts[tuple(tape)].astype(np.float32)
        counts += self.smoothing  # Add smoothing ("fake counts") to all counts
        counts_sum = counts.sum()
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        return probs

class BackoffNgramModel:
    """
    A backoff model that combines multiple n-gram models of different orders.
    During training, it updates all the models with the same data.
    During inference, it uses the highest order model that has data for the current context.
    """

    def __init__(self, vocab_size: int, seq_len: int, smoothing: float = 0.0, counts_threshold: int = 0) -> None:
        """
        Initialize the backoff n-gram model.

        Args:
            vocab_size (int): The size of the vocabulary.
            seq_len (int): The maximum length of the n-gram sequence.
            smoothing (float, optional): Smoothing parameter for all sub-models. Defaults to 0.0.
            counts_threshold (int, optional): Minimum count to consider a context valid. Defaults to 0.
        """
        self.seq_len: int = seq_len
        self.vocab_size: int = vocab_size
        self.smoothing: float = smoothing
        self.counts_threshold: int = counts_threshold
        self.models: Dict[int, NgramModel] = {i: NgramModel(vocab_size, i, smoothing) for i in range(1, seq_len + 1)}

    def train(self, tape: List[int]) -> None:
        """
        Train all sub-models on the given sequence of tokens.

        Args:
            tape (List[int]): A sequence of tokens to train on.
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        for i in range(1, self.seq_len + 1):
            self.models[i].train(tape[-i:])

    def __call__(self, tape: List[int]) -> np.ndarray:
        """
        Calculate the conditional probability distribution of the next token,
        using the highest order model with sufficient data.

        Args:
            tape (List[int]): A sequence of tokens representing the context.

        Returns:
            np.ndarray: The probability distribution for the next token.

        Raises:
            ValueError: If no model is found for the current context.
        """
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # Find the highest order model that has data for the current context
        for i in reversed(range(1, self.seq_len + 1)):
            tape_i = tape[-i+1:] if i > 1 else []
            counts = self.models[i].get_counts(tape_i)
            if counts.sum() > self.counts_threshold:
                return self.models[i](tape_i)
        # We shouldn't get here because unigram model should always have data
        raise ValueError("No model found for the current context")

# -----------------------------------------------------------------------------
# Data iteration and evaluation utils

def dataloader(tokens: List[int], window_size: int) -> Iterator[List[int]]:
    """
    Generate fixed-size windows from a sequence of tokens.

    Args:
        tokens (List[int]): A sequence of tokens.
        window_size (int): The size of each window.

    Yields:
        List[int]: A window of tokens.
    """
    for i in range(len(tokens) - window_size + 1):
        yield tokens[i:i+window_size]

def eval_split(model: Union[NgramModel, BackoffNgramModel], tokens: List[int]) -> float:
    """
    Evaluate a given model on a given sequence of tokens.

    Args:
        model (Union[NgramModel, BackoffNgramModel]): The model to evaluate.
        tokens (List[int]): A sequence of tokens to evaluate on.

    Returns:
        float: The average negative log-likelihood (loss) of the model on the given tokens.
    """
    sum_loss = 0.0
    count = 0
    for tape in dataloader(tokens, model.seq_len):
        x = tape[:-1]  # The context
        y = tape[-1]   # The target
        probs = model(x)
        prob = probs[y]
        sum_loss += -np.log(prob)
        count += 1
    mean_loss = sum_loss / count if count > 0 else 0.0
    return mean_loss

# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main function to train and evaluate the n-gram language model.
    """
    # "Train" the Tokenizer, so we're able to map between characters and tokens
    train_text: str = open('data/train.txt', 'r').read()
    assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
    uchars: List[str] = sorted(list(set(train_text)))  # Unique characters we see in the input
    vocab_size: int = len(uchars)
    char_to_token: Dict[str, int] = {c: i for i, c in enumerate(uchars)}
    token_to_char: Dict[int, str] = {i: c for i, c in enumerate(uchars)}
    EOT_TOKEN: int = char_to_token['\n']  # Designate \n as the delimiting <|endoftext|> token
    
    # Pre-tokenize all the splits one time up here
    test_tokens: List[int] = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
    val_tokens: List[int] = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
    train_tokens: List[int] = [char_to_token[c] for c in open('data/train.txt', 'r').read()]

    # Hyperparameter search with grid search over the validation set
    seq_lens: List[int] = [3, 4, 5]
    smoothings: List[float] = [0.03, 0.1, 0.3, 1.0]
    best_loss: float = float('inf')
    best_kwargs: Dict[str, Union[int, float]] = {}
    for seq_len, smoothing in itertools.product(seq_lens, smoothings):
        # Train the n-gram model
        model = NgramModel(vocab_size, seq_len, smoothing)
        for tape in dataloader(train_tokens, seq_len):
            model.train(tape)
        # Evaluate the train/val loss
        train_loss = eval_split(model, train_tokens)
        val_loss = eval_split(model, val_tokens)
        print(f"seq_len {seq_len} | smoothing {smoothing:.2f} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
        # Update the best hyperparameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_kwargs = {'seq_len': seq_len, 'smoothing': smoothing}

    # Re-train the model with the best hyperparameters
    seq_len = best_kwargs['seq_len']
    print("best hyperparameters:", best_kwargs)
    model = NgramModel(vocab_size, **best_kwargs)
    for tape in dataloader(train_tokens, seq_len):
        model.train(tape)

    # Sample from the model
    tape: List[int] = [EOT_TOKEN] * (seq_len - 1)
    for _ in range(200):
        probs = model(tape)
        # Sample the next token
        coinf = random.random()
        probs_list = probs.tolist()
        next_token = sample_discrete(probs_list, coinf)
        # Update the token tape, print token and continue
        next_char = token_to_char[next_token]
        # Update the tape
        tape.append(next_token)
        if len(tape) > seq_len - 1:
            tape = tape[1:]
        print(next_char, end='')
    print()  # Newline

    # At the end, evaluate and report the test loss
    test_loss = eval_split(model, test_tokens)
    test_perplexity = np.exp(test_loss)
    print(f"test_loss {test_loss}, test_perplexity {test_perplexity}")

    # Get the final counts, normalize them to probs, and write to disk for visualization
    counts = model.counts + model.smoothing
    probs = counts / counts.sum(axis=-1, keepdims=True)
    vis_path = os.path.join("dev", "ngram_probs.npy")
    np.save(vis_path, probs)
    print(f"wrote {vis_path} to disk (for visualization)")

if __name__ == "__main__":
    main()