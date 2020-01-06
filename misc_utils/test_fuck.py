from tqdm import tqdm
import numpy as np
from sklearn import preprocessing as sklearn_preprocessing
import math

Q_num = 5000
G_num = 5000
Moment_num = 50


def l2_distance(vector1, vector2):  # slow
    dist = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist


for q in tqdm(range(Q_num), total=Q_num):
    score_list = []
    for g in range(G_num):
        for m in range(Moment_num):
            a = np.random.rand(1, 2048)
            b = np.random.rand(1, 2048)
            score = -np.linalg.norm(a.reshape(-1) - b.reshape(-1))
            # score = -np.dot((a.reshape(-1)- b.reshape(-1)),(a.reshape(-1)- b.reshape(-1)))
            # score = - l2_distance(list(a.reshape(-1)),list(b.reshape(-1)), )
            score_list.append(score)
    sorted(score_list)
