import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = "Your data set text goes here. It can be a large corpus of text."

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

sequence_length = 40
X = []
y = []

for i in range(sequence_length, len(sequences)):
    X.append(sequences[i-sequence_length:i])
    y.append(sequences[i])

X = np.array(X)
y = np.array(y)
X = pad_sequences(X, maxlen=sequence_length)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y = np.expand_dims(y, axis=-1)
model.fit(X, y, epochs=10, batch_size=64)

def generate_text(seed_text, next_chars=100):
    result = seed_text
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([result[-sequence_length:]])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length)
        predicted = model.predict(token_list, verbose=0)
        predicted_char_index = np.argmax(predicted)
        for char, index in tokenizer.word_index.items():
            if index == predicted_char_index:
                result += char
                break
    return result
