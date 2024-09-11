import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.replace('[^a-zA-Z0-9\s]', '')
    return text

train_df['input_text'] = train_df['input_text'].apply(preprocess_text)
train_df['summary_text'] = train_df['summary_text'].apply(preprocess_text)
test_df['input_text'] = test_df['input_text'].apply(preprocess_text)
test_df['summary_text'] = test_df['summary_text'].apply(preprocess_text)

# Tokenization and padding
max_input_len = 100
max_summary_len = 20

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(train_df['input_text'])
input_sequences = input_tokenizer.texts_to_sequences(train_df['input_text'])
input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')

summary_tokenizer = Tokenizer()
summary_tokenizer.fit_on_texts(train_df['summary_text'])
summary_sequences = summary_tokenizer.texts_to_sequences(train_df['summary_text'])
summary_sequences = pad_sequences(summary_sequences, maxlen=max_summary_len, padding='post')

# Define the model
latent_dim = 300

# Encoder
encoder_inputs = tf.keras.Input(shape=(max_input_len,))
encoder_embedding = tf.keras.layers.Embedding(len(input_tokenizer.word_index) + 1, latent_dim, trainable=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

# Decoder
decoder_inputs = tf.keras.Input(shape=(max_summary_len,))
decoder_embedding = tf.keras.layers.Embedding(len(summary_tokenizer.word_index) + 1, latent_dim, trainable=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention Layer
attention = tf.keras.layers.AdditiveAttention()
attention_output = attention([decoder_outputs, encoder_outputs])
decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention_output])

# Dense layer
dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(summary_tokenizer.word_index) + 1, activation='softmax'))
decoder_outputs = dense(decoder_concat_input)

# Compile model
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare data for training
decoder_input_data = np.array(summary_sequences[:, :-1])
decoder_target_data = np.expand_dims(summary_sequences[:, 1:], -1)

# Train the model
model.fit([input_sequences, decoder_input_data], decoder_target_data, batch_size=64, epochs=20, validation_split=0.2)

# Save the model
model.save('hindi_text_summarization.h5')

print("Model trained and saved successfully.")
