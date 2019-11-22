
import json
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

activitynet_label_list = [line.strip() for line in open('kinetics700/kinetics700_label.txt',"r").readlines()]
elmo = ElmoEmbedder()
with open("wordembed_kinetics700_elmo.json","w") as f:
    _d = dict()
    for label in activitynet_label_list:
        label = label.lower()
        tokens = label.split()
        print(tokens)
        vectors = elmo.embed_sentence(tokens)
        #import ipdb;ipdb.set_trace()
        #[word_num, 1024]
        _d[label] = np.mean(vectors[-1],axis=0).tolist()
        if label == 'archery':
            print(np.mean(vectors[-1],axis=0))
    json.dump(_d, f)

