import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from ngram import NgramModel, BackoffNgramModel, dataloader, eval_split, sample_discrete, RNG

# Initialize random number generator
random = RNG(1337)

# Load data
@st.cache_data
def load_data():
    train_text = open('data/train.txt', 'r').read()
    uchars = sorted(list(set(train_text)))
    vocab_size = len(uchars)
    char_to_token = {c: i for i, c in enumerate(uchars)}
    token_to_char = {i: c for i, c in enumerate(uchars)}
    EOT_TOKEN = char_to_token['\n']
    
    train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]
    val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
    test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
    
    return vocab_size, char_to_token, token_to_char, EOT_TOKEN, train_tokens, val_tokens, test_tokens

vocab_size, char_to_token, token_to_char, EOT_TOKEN, train_tokens, val_tokens, test_tokens = load_data()

# Train model
@st.cache_resource
def train_model(seq_len: int, smoothing: float):
    model = NgramModel(vocab_size, seq_len, smoothing)
    for tape in dataloader(train_tokens, seq_len):
        model.train(tape)
    return model

# Streamlit app
st.title("N-gram Language Model Visualization")

st.markdown("""
This app demonstrates the workings of an N-gram language model. N-gram models are a type of probabilistic language model used to predict the next item in a sequence based on the (n-1) previous items.

### What is an N-gram model?
An N-gram model predicts the probability of a word based on the N-1 previous words. For example, a 3-gram (trigram) model would predict the next word based on the previous two words.

### How does it work?
1. The model is trained on a corpus of text.
2. It counts the occurrences of each N-gram in the training data.
3. These counts are used to estimate probabilities for predicting the next word (or character, in our case) in a sequence.

Let's explore the model by adjusting parameters and seeing the results!
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
seq_len = st.sidebar.slider("Sequence Length (N)", min_value=2, max_value=6, value=3, 
                            help="The 'N' in N-gram. Higher values consider more context but require more data.")
smoothing = st.sidebar.slider("Smoothing", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                              help="Helps handle unseen N-grams. Higher values give more probability to unseen events.")

st.sidebar.markdown("""
### Parameter Explanation

**Sequence Length (N)**: This determines how many previous characters the model considers when predicting the next character. A higher N allows the model to capture more complex patterns, but requires more data to train effectively.

**Smoothing**: This parameter helps the model handle N-grams it hasn't seen during training. Without smoothing, any unseen N-gram would have a probability of zero, which can be problematic. Smoothing assigns a small probability to these unseen events.
""")

# Train model
model = train_model(seq_len, smoothing)

# Evaluate model
train_loss = eval_split(model, train_tokens)
val_loss = eval_split(model, val_tokens)
test_loss = eval_split(model, test_tokens)

st.header("Model Evaluation")
st.markdown("""
The model's performance is evaluated using the average negative log-likelihood (loss) on different datasets. 
Lower values indicate better performance.
""")
col1, col2, col3 = st.columns(3)
col1.metric("Train Loss", f"{train_loss:.4f}")
col2.metric("Validation Loss", f"{val_loss:.4f}")
col3.metric("Test Loss", f"{test_loss:.4f}")

st.markdown("""
- **Train Loss**: How well the model performs on the data it was trained on.
- **Validation Loss**: Performance on unseen data, used to tune hyperparameters.
- **Test Loss**: Final evaluation on completely unseen data.

If train loss is much lower than validation/test loss, the model might be overfitting.
""")

# Generate text
st.header("Text Generation")
st.markdown("""
Now, let's use our trained model to generate some text! The model will predict each character based on the previous N-1 characters.
""")
num_chars = st.slider("Number of characters to generate", min_value=10, max_value=500, value=100)
if st.button("Generate Text"):
    tape = [EOT_TOKEN] * (seq_len - 1)
    generated_text = ""
    for _ in range(num_chars):
        probs = model(tape)
        coinf = random.random()
        probs_list = probs.tolist()
        next_token = sample_discrete(probs_list, coinf)
        next_char = token_to_char[next_token]
        generated_text += next_char
        tape.append(next_token)
        if len(tape) > seq_len - 1:
            tape = tape[1:]
    st.text_area("Generated Text", generated_text, height=200)
    st.markdown("""
    This text is generated character by character. For each prediction:
    1. The model looks at the previous N-1 characters.
    2. It calculates the probability distribution for the next character.
    3. It randomly selects a character based on these probabilities.
    4. The process repeats for the desired number of characters.
    """)

# Visualize n-gram probabilities
st.header("N-gram Probabilities Visualization")
st.markdown("""
This heatmap visualizes the probabilities of the next character given different contexts. 
Each row represents a different context (previous N-1 characters), and each column represents a possible next character.
Brighter colors indicate higher probabilities.
""")
if st.button("Visualize Probabilities"):
    counts = model.counts + model.smoothing
    probs = counts / counts.sum(axis=-1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(probs.reshape(-1, vocab_size), cmap='viridis', aspect='auto')
    ax.set_xlabel('Next Character')
    ax.set_ylabel('Context (Previous Characters)')
    ax.set_title(f'{seq_len}-gram Probabilities')
    plt.colorbar(im)
    st.pyplot(fig)
    st.markdown("""
    Interpreting the heatmap:
    - Each row represents a unique context (combination of N-1 previous characters).
    - Each column represents a possible next character.
    - Brighter colors (yellow) indicate higher probabilities, while darker colors (blue) indicate lower probabilities.
    - You can observe patterns in the language model's predictions based on different contexts.
    """)

# Display model details
st.header("Model Details")
st.json({
    "Vocabulary Size": vocab_size,
    "Sequence Length": seq_len,
    "Smoothing": smoothing,
    "Total Parameters": model.counts.size
})
st.markdown("""
- **Vocabulary Size**: The number of unique characters in our dataset.
- **Sequence Length**: The 'N' in N-gram, determining how much context we use for predictions.
- **Smoothing**: The value added to all counts to handle unseen N-grams.
- **Total Parameters**: The total number of probability values our model stores.

A larger vocabulary or sequence length dramatically increases the number of parameters, which can lead to better performance but requires more data and computation.
""")

st.markdown("""
### Experiment and Learn!

Try adjusting the Sequence Length and Smoothing parameters in the sidebar. Observe how they affect:
1. The model's performance (train, validation, and test loss)
2. The quality of generated text
3. The patterns in the probability heatmap

This hands-on experimentation will help you understand the trade-offs in N-gram language modeling!
""")