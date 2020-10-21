from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# to Decode the sentence from encoded(list of Numbers) form to readable format
def decode_sentence(lst):
    text = " ".join([rev_word_dict.get(i, "?") for i in lst])
    return text


# Features and labels for training the model
features = [[3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 5, 6, 7, 13, 9, 14], [15, 16, 9, 17], [18, 19, 20, 9, 21],
            [22, 21, 9, 23], [22, 24, 9, 25], [6, 7, 20, 9, 16], [22, 21, 9, 20], [18, 7, 20, 9, 26],
            [27, 7, 28, 9, 29], [4, 5, 18, 19, 20, 9, 30], [31, 7, 32, 9, 30], [33, 5, 27, 7, 34, 9, 35],
            [11, 12, 5, 27, 7, 36, 9, 37], [18, 19, 38, 9, 39], [11, 12, 5, 18, 19, 40, 9, 41], [42, 7, 43, 9, 44]]
labels = [[0], [0], [0], [1], [1], [1], [0], [1], [1], [0], [1], [0], [0], [0], [1], [1], [0]]
labels = np.reshape(labels, (len(labels),))
word_dict = {'hey': 3, 'calculate': 4, 'the': 5, 'sum': 6, 'of': 7, '9': 8, 'and': 9, '6': 10, 'what': 11, 'is': 12,
             '5': 13, '13': 14, 'add': 15, '2': 16, '55': 17, 'difference': 18, 'between': 19, '10': 20, '3': 21,
             'subtract': 22, '7': 23, '22': 24, '4': 25, '90': 26, 'addition': 27, '14': 28, '42': 29, '34': 30,
             'sumation': 31, '15': 32, 'do': 33, '25': 34, '19': 35, '47': 36, '37': 37, '11': 38, '65': 39, '100': 40,
             '45': 41, 'summation': 42, '198': 43, '31': 44}
rev_word_dict = dict([(v, k) for k, v in word_dict.items()])


# Converting the raw list of numbers into a format that the model and the machone can understand using
# padding to max of 10 words
train_data = pad_sequences(features, maxlen=10, value=0, padding="post")

# Definig the model and adding Layers to it
model = Sequential()
model.add(Embedding(100, 16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Training the model with the formatted data(Training data) and the training labels(labels) for 150 times
model.fit(train_data, labels, epochs=150)


def review_encode(s):
    encoded = []

    for word in s:
        if word.lower() in word_dict:
            encoded.append(word_dict[word.lower()])

    return encoded


line = input()
nline = line.strip().split(" ")
encode = review_encode(nline)
encode = pad_sequences([encode], value=0, padding="post", maxlen=10)  # make the data 10 words long
predict = model.predict(encode)
pred = predict[0]
print(pred)
if round(pred[0]) == 0:
    print("Addition")
else:
    print("Subraction")

line_lst = line.split()
val_lst = []
for x in line_lst:
    if x.isdigit():
        val_lst.append(int(x))

val1, val2 = val_lst
if round(pred[0]) == 0:
    ans = val1 + val2
else:
    ans = val1 - val2

print(ans)


