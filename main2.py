import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global Variables
model_file = "chat_model.keras"
tokenizer_file = "tokenizer.pkl"
content_file = "content.txt"  # Input text file
max_sequence_len = 100  # Set based on average length of conversational input
fixed_vocab_size = 10000  # Fixed vocabulary size

# Function to load content from the text file
def load_content(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

# Function to create or load the model
def create_or_load_model(vocab_size):
    if os.path.exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_len),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dense(128, activation='relu'),
            Dense(vocab_size, activation='softmax')  # Output layer for next word prediction
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

# Function to preprocess text for conversation generation
def preprocess_text(text, tokenizer):
    sequences = []
    token_list = tokenizer.texts_to_sequences([text])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        # Filter out any sequence with a word index greater than fixed_vocab_size
        if any(word >= fixed_vocab_size for word in n_gram_sequence):
            continue
        sequences.append(n_gram_sequence)

    # Pad sequences
    sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')
    return sequences

# Function to save the tokenizer
def save_tokenizer(tokenizer):
    import pickle
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)

# Function to load the tokenizer
def load_tokenizer():
    import pickle
    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, 'rb') as file:
            return pickle.load(file)
    return Tokenizer(num_words=fixed_vocab_size, oov_token='<OOV>')

# Function to generate a conversational response
def generate_response(seed_text, next_words, model, max_sequence_len, tokenizer, stop_phrases=None):
    stop_phrases = stop_phrases or ["چطوری؟", "<EOS>"]

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)
        predicted_word = tokenizer.index_word.get(predicted_word_index[0], '')

        if predicted_word in stop_phrases:
            break

        seed_text += " " + predicted_word
    return seed_text

def main():
    content = load_content(content_file)
    if not content:
        print(f"No content found in {content_file}. Please add text content to the file.")
        return

    tokenizer = load_tokenizer()
    tokenizer.fit_on_texts([content])

    sequences = preprocess_text(content, tokenizer)

    # Use the fixed vocabulary size for the model
    vocab_size = fixed_vocab_size

    # Ensure model is created with the correct vocab size
    model = create_or_load_model(vocab_size)

    # Create labels for the next word prediction
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    # Train the model
    model.fit(X, y, epochs=50, verbose=1)

    # Save model and tokenizer after processing
    model.save(model_file)
    save_tokenizer(tokenizer)

    # Generate a response after training
    seed_text = "سلام"
    generated_response = generate_response(seed_text, next_words=20, model=model, max_sequence_len=max_sequence_len, tokenizer=tokenizer, stop_phrases=["چطوری؟", "<EOS>"])
    print(f"\nGenerated Response:\n{generated_response}\n")

if __name__ == "__main__":
    main()
