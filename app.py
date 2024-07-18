import ast
import itertools
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

from ngram import (
    NgramModel,
    dataloader,
    eval_split,
    RNG,
    sample_discrete,
)


def validate_input(
    seq_lens_text: str, smoothings_text: str
) -> Tuple[List[int], List[float]]:
    """
    Validate and parse user input for sequence lengths and smoothing values.

    Args:
        seq_lens_text (str): User input for sequence lengths.
        smoothings_text (str): User input for smoothing values.

    Returns:
        Tuple[List[int], List[float]]: Parsed sequence lengths and smoothing values.

    Raises:
        ValueError: If input is invalid or doesn't meet requirements.
    """
    try:
        seq_lens = ast.literal_eval(seq_lens_text)
        smoothings = ast.literal_eval(smoothings_text)
    except (SyntaxError, ValueError):
        raise ValueError("Sequence lengths and smoothings must be valid Python syntax.")

    if not isinstance(seq_lens, list) or not isinstance(smoothings, list):
        raise ValueError("Sequence lengths and smoothings must be Python lists.")

    if not all(isinstance(x, int) and x > 0 for x in seq_lens):
        raise ValueError("Sequence Lengths must be positive integers.")

    if not all(isinstance(x, (int, float)) and x > 0 for x in smoothings):
        raise ValueError("Smoothing values must be positive numbers.")

    return seq_lens, smoothings


def load_tokens(file_path: str) -> List[int]:
    """
    Load and tokenize text from a file.

    Args:
        file_path (str): Path to the file containing text.

    Returns:
        List[int]: List of tokenized characters.
    """
    with open(file_path, "r") as f:
        text = f.read()
    return [char_to_token[c] for c in text]


def evaluate_hyperparameters(
    seq_lens: List[int],
    smoothings: List[float],
    train_tokens: List[int],
    val_tokens: List[int],
) -> Dict:
    """
    Evaluate different combinations of hyperparameters.

    Args:
        seq_lens (List[int]): List of sequence lengths to evaluate.
        smoothings (List[float]): List of smoothing values to evaluate.
        train_tokens (List[int]): Tokenized training data.
        val_tokens (List[int]): Tokenized validation data.

    Returns:
        Dict: Best hyperparameters and their performance.
    """
    best_loss = float("inf")
    best_kwargs = {}
    iterations = len(seq_lens) * len(smoothings)

    st.write("## Hyperparameter Evaluation Results")
    progress_bar = st.progress(0.0, "")
    df_placeholder = st.empty()
    df = pd.DataFrame(
        columns=["Sequence Length", "Smoothing", "Training Loss", "Validation Loss"]
    )

    for i, (seq_len, smoothing) in enumerate(itertools.product(seq_lens, smoothings)):
        model = NgramModel(vocab_size, seq_len, smoothing)
        for tape in dataloader(train_tokens, seq_len):
            model.train(tape)

        train_loss = eval_split(model, train_tokens)
        val_loss = eval_split(model, val_tokens)

        progress_bar.progress(
            (i + 1) / iterations, f"Iteration {i + 1} of {iterations}"
        )
        df.loc[len(df)] = {
            "Sequence Length": seq_len,
            "Smoothing": smoothing,
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
        }
        df_placeholder.dataframe(df, height=int(35.2 * (iterations + 1)))

        if val_loss < best_loss:
            best_loss = val_loss
            best_kwargs = {"seq_len": seq_len, "smoothing": smoothing}

    return best_kwargs


def generate_sample(model: NgramModel, seq_len: int, num_chars: int = 200) -> str:
    """
    Generate a sample text from the trained model.

    Args:
        model (NgramModel): Trained N-gram model.
        seq_len (int): Sequence length used in the model.
        num_chars (int): Number of characters to generate.

    Returns:
        str: Generated text sample.
    """
    tape = [EOT_TOKEN] * (seq_len - 1)
    sample = ""
    for _ in range(num_chars):
        probs = model(tape)
        coinf = rng.random()
        next_token = sample_discrete(probs.tolist(), coinf)
        next_char = token_to_char[next_token]
        tape.append(next_token)
        if len(tape) > seq_len - 1:
            tape = tape[1:]
        sample += next_char
    return sample


def plot_heatmap(probs: np.ndarray) -> None:
    """
    Plot a heatmap of the model's conditional probabilities.

    Args:
        probs (np.ndarray): Probability matrix from the trained model.
    """
    expected_shape = (27, 27, 27, 27)
    if probs.shape != expected_shape:
        st.error(
            f"Unexpected probability matrix shape. Expected {expected_shape}, but got {probs.shape}."
        )
        st.stop()

    reshaped = probs.reshape(27**2, 27**2)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(reshaped, cmap="hot", interpolation="nearest")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Probability", rotation=270, labelpad=15)

    st.pyplot(fig)


def plot_probability_distribution(
    model: NgramModel, tape: List[int], name: str
) -> None:
    """
    Plot the probability distribution of the next token given the current state.

    Args:
        model (NgramModel): Trained N-gram model.
        tape (List[int]): Current state (sequence of tokens).
        name (str): Current generated name.
    """
    probs = model(tape)
    chars = [token_to_char[i] for i in range(len(probs))]
    df = pd.DataFrame(
        {
            "Token": chars,
            "Probability": probs,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"##### Step: {len(name) + 1}")
        st.write(f"Current state: '{name}'")
        st.write(f"Tape of input tokens to model: {tape}")
        st.write(f"Characters shown to model: {[token_to_char[i] for i in tape]}")
        df_display = df.copy()
        df_display["Probability"] = df_display["Probability"].apply(
            lambda x: f"{x:.4g}"
        )
        st.dataframe(df_display.T)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = plt.Normalize(df["Probability"].min(), df["Probability"].max())
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
        sm.set_array([])

        colors = sm.to_rgba(df["Probability"]).tolist()

        sns.barplot(x="Token", y="Probability", hue="Token", legend=False, data=df, palette=colors, ax=ax)

        ax.set_xlabel("Token")
        ax.set_ylabel("Probability")
        ax.set_title(f'Probability of Next Token (From State: "{name}")')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=90)

        st.pyplot(fig)


def generate_name_step_by_step(model: NgramModel, seq_len: int) -> str:
    """
    Generate a name step-by-step, visualizing each step.

    Args:
        model (NgramModel): Trained N-gram model.
        seq_len (int): Sequence length used in the model.

    Returns:
        str: Generated name.
    """
    tape = [EOT_TOKEN] * (seq_len - 1)
    name = ""

    st.write("### Step-by-step name generation")
    st.write(
        """
    At each step, the model looks at the last (N - 1) tokens and outputs
    a probability distribution for the next token. The model generates unique
    names by sampling from this distribution. Let's visualize this process step-by-step.
    """
    )

    while len(name) == 0 or name[-1] != "\n":
        plot_probability_distribution(model, tape, name)

        # Sample the next token
        probs = model(tape)
        coinf = rng.random()
        next_token = sample_discrete(probs.tolist(), coinf)
        next_char = token_to_char[next_token]

        # Update the tape and name
        tape.append(next_token)
        if len(tape) > seq_len - 1:
            tape = tape[1:]
        name += next_char

        st.write(f"Sampled next token: {next_token}, Next character: '{next_char}'")
        st.write("---")

    return name.strip()


def main():
    st.set_page_config("N-gram Language Model Name Generator", page_icon="🔤")

    st.write(
        """
    # N-gram Language Model Name Generator

    This app trains an N-gram language model on a dataset of names and generates new, unique names based on the learned patterns.
    Adjust the parameters to see how they affect the model's performance and the generated names.

    Every combination of `Sequence Lengths` and `Smoothings` set will
    be evaluated to determine the best combination. The model will then
    be retrained using the optimal hyperparameters and generate a heat
    map showing conditional probabilities of the N-gram model.
    """
    )

    st.sidebar.write("## Parameters")
    seq_lens_text = st.sidebar.text_input("Sequence Lengths", "[3, 4, 5]")
    smoothings_text = st.sidebar.text_input("Smoothings", "[0.03, 0.1, 0.3, 1.0]")

    try:
        seq_lens, smoothings = validate_input(seq_lens_text, smoothings_text)
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop()

    random_seed = st.sidebar.number_input("Random Seed", 1337)
    test_tokens_path = st.sidebar.text_input("Test Tokens Path", "data/test.txt")
    val_tokens_path = st.sidebar.text_input("Validation Tokens Path", "data/val.txt")
    train_tokens_path = st.sidebar.text_input("Training Tokens Path", "data/train.txt")

    iterations = len(seq_lens) * len(smoothings)
    st.sidebar.write(f"Total Iterations: {iterations}")

    global train_text, uchars, vocab_size, char_to_token, token_to_char, EOT_TOKEN, rng

    train_text = open(train_tokens_path, "r").read()
    assert all(
        c == "\n" or ("a" <= c <= "z") for c in train_text
    ), "Invalid characters in training data"
    uchars = sorted(list(set(train_text)))
    vocab_size = len(uchars)
    char_to_token = {c: i for i, c in enumerate(uchars)}
    token_to_char = {i: c for i, c in enumerate(uchars)}
    EOT_TOKEN = char_to_token["\n"]

    test_tokens = load_tokens(test_tokens_path)
    val_tokens = load_tokens(val_tokens_path)
    train_tokens = load_tokens(train_tokens_path)

    rng = RNG(random_seed)

    if st.button("Train Model and Generate Names"):
        best_kwargs = evaluate_hyperparameters(
            seq_lens, smoothings, train_tokens, val_tokens
        )

        st.write("## Best Hyperparameters")
        st.dataframe(
            {
                "Hyperparameter": ["Sequence Length", "Smoothing"],
                "Value": [best_kwargs["seq_len"], best_kwargs["smoothing"]],
            }
        )

        model = NgramModel(vocab_size, **best_kwargs)
        for tape in dataloader(train_tokens, best_kwargs["seq_len"]):
            model.train(tape)

        st.write("## Generated Names")
        sample = generate_sample(model, best_kwargs["seq_len"])
        st.text_area("", sample, height=300)

        st.write("## Model Evaluation Results")
        test_loss = eval_split(model, test_tokens)
        test_perplexity = np.exp(test_loss)
        df = pd.DataFrame(
            {
                "Metric": ["Test Loss", "Test Perplexity"],
                "Value": [test_loss, test_perplexity],
            }
        )
        st.dataframe(df)

        counts = model.counts + model.smoothing
        probs = counts / counts.sum(axis=-1, keepdims=True)
        plot_heatmap(probs)

        st.write(
            """
        ## Understanding the Heatmap

        The heatmap visualizes the conditional probabilities learned by the N-gram model:

        - Each pixel represents the probability of a specific sequence of characters.
        - Brighter colors indicate higher probabilities, darker colors show lower probabilities.
        - The x and y axes represent different character sequences.
        - Bright spots reveal common character patterns in names.
        - Dark areas indicate rare or unlikely character combinations.

        This visualization helps us understand how the model has captured the statistical patterns of name structures from the training data.
        """
        )

        st.write("## How the N-Gram Model Generates Names")
        st.write(
            f"### Using Best Model (N={best_kwargs['seq_len']}, smoothing={best_kwargs['smoothing']})"
        )
        st.write(
            """
        The N-gram model generates names by following these steps:

        1. Start with a sequence of (N-1) tokens, usually representing the start of a name.
        2. For each step:
           a. The model looks at the last (N-1) tokens.
           b. It outputs a probability distribution for the next token.
           c. A token is randomly sampled from this distribution.
           d. The sampled token is added to the sequence, and the process repeats.
        3. The generation stops when an end-of-text token is sampled.

        This process allows the model to create unique names that follow the patterns it learned from the training data.
        """
        )

        st.write("##### Special Note")
        st.write(
            "The model outputs tokens, which in this case, correspond to characters. In more complex models like GPT, tokens can represent more than one character. In our case, here is a reference for what value each token index maps to."
        )

        st.write("### Token to Character Mapping")
        df = pd.DataFrame(
            {
                "Token Index": [char_to_token[c] for c in uchars],
                "Character": [r"\n" if c == "\n" else c for c in uchars],
            }
        )
        
        st.dataframe(df.T)

        st.write("## Interactive Name Generation")
        st.write("Let's generate a name step-by-step to see how the model works.")
        generated_name = generate_name_step_by_step(model, best_kwargs["seq_len"])
        st.write(f"### Final generated name: {generated_name}")


if __name__ == "__main__":
    main()
