# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset = pd.read_csv(train_file_path, sep='\t', header=None)
test_dataset = pd.read_csv(test_file_path, sep='\t', header=None)

train_data = train_dataset[1]
train_labels = train_dataset[0]
test_data = test_dataset[1]
test_labels = test_dataset[0]

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data)

# Chuyển đổi văn bản thành số liệu
X_train = tokenizer.texts_to_sequences(train_data)
X_test = tokenizer.texts_to_sequences(test_data)

# Padding
max_length = 100
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Xây dựng mô hình RNN
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    SimpleRNN(units=64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert string labels to numerical labels
train_labels_numerical = np.where(train_labels == 'ham', 0, 1)
test_labels_numerical = np.where(test_labels == 'ham', 0, 1)

# Huấn luyện mô hình using the numerical labels
model.fit(X_train, train_labels_numerical, epochs=5, batch_size=2, validation_split=0.2)

# Đánh giá mô hình using the numerical labels
loss, accuracy = model.evaluate(X_test, test_labels_numerical)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  x_pred = tokenizer.texts_to_sequences([pred_text])
  x_pred = pad_sequences(x_pred, maxlen=max_length)
  pred_proba = model.predict(x_pred)[0][0]
  label = 'spam' if pred_proba > 0.5 else 'ham'
  return [pred_proba, label]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
