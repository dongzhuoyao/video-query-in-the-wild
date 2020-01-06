from data_generate.activitynet_label import (
    arv_train_label,
    arv_test_label,
    arv_val_label,
    activitynet_label_list,
)
import json

import numpy as np

import gensim.downloader as api

word_vectors = api.load("fasttext-wiki-news-subwords-300")


def wrap_sentence(tokens):
    for _, t in enumerate(tokens):
        tokens[_] = t.lower()
    tmp = 0
    effective_token = 0

    def change(tokens, old, new):
        idx = tokens.index(old)
        tokens[idx] = new
        return tokens

    if "mooping" in tokens:
        tokens = change(tokens, "mooping", "cleaning")
    if "slacklining" in tokens:
        tokens = ["slackline"]
    if "powerbocking" in tokens:
        tokens = ["spring", "stilts"]
    if "forniture" in tokens:
        tokens = change(tokens, "forniture", "furniture")
    if "jack-o-lanterns" in tokens:
        tokens = ["carving", "halloween", "pumpkin"]
    if "plataform" in tokens:
        tokens = change(tokens, "plataform", "platform")
    if "blow-drying" in tokens:
        tokens = ["blow", "drying", "hair"]
    if "rock-paper-scissors" in tokens:
        tokens = ["rock", "paper", "scissors"]

    for t in tokens:
        t = t.lower()
        try:
            _ = word_vectors[t.lower()]
            tmp += _
            effective_token += 1
        except:
            # tmp = np.ones((200,))
            print("pass word: {}".format(t.lower()))

    tmp = tmp / (effective_token + 1e-10)
    return tmp


with open(
    "wordembed_fasttext_d300.json", "w"
) as f:  # KeyError: "word 'mooping' not in vocabulary"
    _d = dict()
    for label in activitynet_label_list:
        tokens = label.split()
        vectors = wrap_sentence(tokens)
        # import ipdb;ipdb.set_trace()
        # [word_num, 1024]
        _d[label] = vectors.tolist()
        print(tokens)
        print(len(_d[label]))
    json.dump(_d, f)
