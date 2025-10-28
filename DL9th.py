import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


tweets = [
    "This product is amazing! #LoveIt", 
    "I hate this product. Worst purchase ever!", 
    "The product is okay, but could be better.", 
    "Totally worth the price! Very satisfied.", 
    "Not what I expected, I'm disappointed."
]


sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']

G = nx.Graph()
G.add_nodes_from([(1, {'text': tweets[0]}), (2, {'text': tweets[1]}), (3, {'text': tweets[2]}), 
                  (4, {'text': tweets[3]}), (5, {'text': tweets[4]})])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sentiments)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
X = tokenizer.texts_to_sequences(tweets)
X = pad_sequences(X, padding='post')

y = np.array(y)


def build_model(input_length, vocab_size, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=input_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


input_length = X.shape[1]
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(np.unique(y))


model = build_model(input_length, vocab_size, num_classes)
model.summary()

history = model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=input_length, padding='post')
    prediction = model.predict(padded_sequence)
    sentiment = label_encoder.inverse_transform([np.argmax(prediction)])
    return sentiment

sample_text = "This product is so cool!"
print(f"Predicted Sentiment for '{sample_text}': {predict_sentiment(sample_text)[0]}")


for node in G.nodes:
    sentiment = predict_sentiment(G.nodes[node]['text'])[0]
    G.nodes[node]['sentiment'] = sentiment

pos = nx.spring_layout(G)
sentiment_labels = nx.get_node_attributes(G, 'sentiment')

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
        node_size=2000, font_size=10, font_weight='bold')
nx.draw_networkx_labels(G, pos, labels=sentiment_labels, font_color='red', font_size=12)

plt.title("Network Graph with Sentiment Labels (Twitter User Opinions)")
plt.show()
