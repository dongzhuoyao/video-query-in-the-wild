import json
import numpy as np
from numpy import linalg as LA
from sklearn import preprocessing as sklearn_preprocessing

video_embed = json.load(open("wordembed_elmo.json"))
openset_embed = json.load(open("wordembed_kinetics700_elmo.json"))

for key, value in video_embed.items():
    value = sklearn_preprocessing.normalize(np.array(value).reshape(1, -1))
    result_list = []
    for c_key, c_value in openset_embed.items():
        c_value = sklearn_preprocessing.normalize(
            np.array(c_value).reshape(1, -1)
        )
        score = -LA.norm(c_value - value)
        result_list.append(dict(label=c_key, score=score))
    result_list.sort(key=lambda x: x["score"], reverse=True)
    print("top")
    for ele in result_list[:5]:
        print("{} ->{}: {}".format(key, ele["label"], ele["score"]))

    print("ss")
