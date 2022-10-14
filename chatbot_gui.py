import random
import json
import pickle as pkl
import io
import os
from unittest.mock import Base
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
from HanTa import HanoverTagger as ht
from tensorflow.python.keras.models import load_model
import numpy as np
import tkinter
from tkinter import *


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
    # <----- OPTIONAL: If prob is smaller than THRESHOLD_OPT than print the standard response of "SONSTIGES" -----> #
    # THRESHOLD_OPT = float(0.3)
    # if return_list[0]['probabilty'] < THRESHOLD_OPT:
    #    return_list.insert(0,{"intent": "SONSTIGES","probabilty": float(1.0)})
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


""" CONVERSATION WRAPPED IN TKINTER GUI """
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        ints = predict_class(msg)
        res = get_repsonse(ints, answer_data)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("Chat")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
# Placeholder
# ChatLog.insert(END,'Bot: Herzlich willkommen. Ich bin dein Chatbot rund um Fragen zum Leibniz-Rechenzentrum. Los stell mir eine Frage!\n👉 P.S. Gib EXIT zum Beenden ein.\n')
# Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0 ,bg="#325dde", activebackground="#325dde",fg='#ffffff', command= send )
# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

#EntryBox.bind("<Return>", send)
# Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
# Mainloop
base.mainloop()