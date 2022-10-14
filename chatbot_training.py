# Dependencies
import random
import json
import pickle as pkl
import os
# to ignore GPU warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import io
import numpy as np
from HanTa import HanoverTagger as ht
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import gradient_descent_v2



""" VORVERARBEITUNG TEIL 1: LEMMATIZATION """

def sent_tokenizer(sent):
    # Returns a tokenized list of words w/o punctuation symbols
    #\b = boundary \w = word
    wordlist = re.findall(r"\b\w.+?\b",sent)
    return wordlist

train_data = json.loads(io.open("trainingsdaten.json", "r", encoding="utf-8").read())
# print(train_data["intents"][1]["patterns"][0])

words = []
classes = []
assignments = [] # Zuordnungen

for intent in train_data["intents"]:
    for pattern in intent["patterns"]:
        word_list = sent_tokenizer(pattern)
        words.extend(word_list)
        assignments.append((word_list, intent["intent"]))
        if intent["intent"] not in classes:
            classes.append(intent["intent"])

# Create list of word lemmas
tagger = ht.HanoverTagger("morphmodel_ger.pgz")
words = [tagger.analyze(word.lower())[0] for word in words]
# Reduce lists to unique values
words = sorted(set(words))
classes = sorted(set(classes))
# Writes into a pickle file
pkl.dump(words, open("words.pkl", "wb"))
pkl.dump(classes, open("classes.pkl", "wb"))


""" VORVERARBEITUNG TEIL 2: BAG OF WORDS VECTORIZATION """
training = []
output_empty = [0] * len(classes)

for assignment in assignments:
    bag = []
    word_patterns = assignment[0]
    word_patterns = [tagger.analyze(word.lower())[0] for word in word_patterns]
    for word in words:
        # Evaluating if each word occurs in BOW
        # If occurs 0 else 1
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(assignment[1])] = 1

    training.append([bag, output_row])

# Shuffle data
random.shuffle(training)
# Convert into numpy array
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])


""" TEIL 3: TRAINING DES KI-MODELLS """
# Build network architecture
model = Sequential()
model.add(Dense(60, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# Define gradient descent
# weight_decay = "..While weight decay is an additional term in the weight update rule that causes the weights to exponentially decay to zero, if no other update is scheduled."
# nesterov = "Nesterov Accelerated Gradient is a momentum-based SGD optimizer that looks ahead"
sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Evaluate the complete model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Start training
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=10, verbose=1)
# Save model
model.save("chatbot_model.h5", hist)
model.summary()