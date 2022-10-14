# Dependencies
import random
import json
import pickle as pkl
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
from HanTa import HanoverTagger as ht
from tensorflow.python.keras.models import load_model
import numpy as np


""" LOAD MODEL FILES TAGGER """

model = load_model("chatbot_model.h5")
words = pkl.load(open("words.pkl","rb"))
classes = pkl.load(open("classes.pkl","rb"))
tagger = ht.HanoverTagger("morphmodel_ger.pgz")
answer_data = json.loads(io.open("antwortdaten.json", "r", encoding="utf-8").read())


""" FUNCTION FOR BOT INTERACTION """

def sent_tokenizer(sent):
    # Returns a tokenized list of words w/o punctuation symbols
    #\b = boundary \w = word
    wordlist = re.findall(r"\b\w.+?\b",sent)
    return wordlist

def tokenize_lemmatize(sent):
    # Tokenization
    sentence_words = sent_tokenizer(sent)
    # Lemmatization
    sentence_words = [tagger.analyze(word.lower())[0] for word in sentence_words]
    return sentence_words

def bag_of_words(sent):
    # Checks if sentence is in BOW by assigning 1 if yes else 0
    sentence_words = tokenize_lemmatize(sent)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sent):
    bow = bag_of_words(sent)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.10
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort pred results in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    # Create empty list
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]],"probabilty": float(r[1])})
    # check if first values threshold smaller than number X
    if return_list[0]['probabilty'] < float(0.8):
        return_list.insert(0,{"intent": "SONSTIGES","probabilty": float(1.0)})
    return return_list

def get_repsonse(intents_list, intents_json):
    # Takes output of predict_class function and returns respective value
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["response"]
    for i in list_of_intents:
        if i["intent"] == tag:
            result = random.choice(i["patterns"])
            break
    return result


""" CONVERSATION """

print(f"[BOT]:\t Herzlich willkommen. Ich bin dein Chatbot rund um Fragen zum Leibniz-Rechenzentrum. Los stell mir eine Frage!")
print("         👉 P.S. Gib 'EXIT' zum Beenden ein.")

# Endlosschleife für die Konversation
while True:
    message = input(f"[USER]:\t")
    if message == "EXIT":
        print(f"[BOT]:\tVielen Dank für das Gespräch.")
        break
    ints = predict_class(message)
    res = get_repsonse(ints, answer_data)
    # KI Funktionalität
    print(f"[BOT]:\t{res}")
