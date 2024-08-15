import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import PyPDF2

# Global Variables
model_file = "chat_model.keras"
tokenizer_file = "tokenizer.pkl"
pdf_file = "book.pdf"  # Input PDF file
max_sequence_len = 100  # Set based on average length of conversational input
fixed_vocab_size = 10000  # Fixed vocabulary size

# Function to extract text from a PDF file, page by page
def load_content_from_pdf_by_page(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            for page_number, page in enumerate(reader.pages):
                yield page.extract_text(), page_number, total_pages
    else:
        yield "", 0, 0

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
    # Initialize the model and tokenizer
    tokenizer = load_tokenizer()
    model = create_or_load_model(fixed_vocab_size)

    # Process the PDF file page by page
    for page_content, page_number, total_pages in load_content_from_pdf_by_page(pdf_file):
        if not page_content.strip():
            print(f"Skipping empty page {page_number + 1} of {total_pages}")
            continue

        print(f"Processing page {page_number + 1} of {total_pages}...")

        # Update the tokenizer with the current page's content
        tokenizer.fit_on_texts([page_content])

        # Preprocess the text from the page
        sequences = preprocess_text(page_content, tokenizer)

        if not sequences.any():
            print(f"No valid sequences found on page {page_number + 1} of {total_pages}")
            continue

        # Create labels for the next word prediction
        sequences = np.array(sequences)
        X, y = sequences[:,:-1], sequences[:,-1]
        y = tf.keras.utils.to_categorical(y, num_classes=fixed_vocab_size)

        # Train the model on the current page's data
        model.fit(X, y, epochs=10, verbose=1)  # Adjust epochs as needed for each page

    # Save model and tokenizer after processing all pages
    model.save(model_file)
    save_tokenizer(tokenizer)

    # Generate a response after training
    seed_text = "سلام"
    generated_response = generate_response(seed_text, next_words=20, model=model, max_sequence_len=max_sequence_len, tokenizer=tokenizer, stop_phrases=["چطوری؟", "<EOS>"])
    print(f"\nGenerated Response:\n{generated_response}\n")

if __name__ == "__main__":
    main()
