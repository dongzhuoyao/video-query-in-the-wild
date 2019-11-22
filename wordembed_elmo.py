from data_generate.activitynet_label import arv_train_label,arv_test_label,arv_val_label,activitynet_label_list
import json
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

elmo = ElmoEmbedder()
with open("wordembed_elmo.json","w") as f:
    _d = dict()
    for label in activitynet_label_list:
        #label = label.lower()
        tokens = label.split()
        print(tokens)
        vectors = elmo.embed_sentence(tokens)
        #import ipdb;ipdb.set_trace()
        #[word_num, 1024]
        _d[label] = np.mean(vectors[-1],axis=0).tolist()
        if label == 'archery':
            print(np.mean(vectors[-1],axis=0))
    json.dump(_d, f)

