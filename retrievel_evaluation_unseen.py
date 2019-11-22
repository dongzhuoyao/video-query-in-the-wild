
""" Video loader for the Charades dataset """
import torch,os,time
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from datasets.utils import default_loader
import datasets.video_transforms as videotransforms
from tqdm import tqdm
import random,json,sklearn
import torch.utils.data as data
from data_generate.activitynet_label import arv_train_label,arv_test_label,arv_val_label,activitynet_label_list
from pytorchgo.utils import logger
from sklearn import preprocessing as sklearn_preprocessing
from datasets.arv import  json_path
from sklearn.metrics import average_precision_score
import getpass
import faiss  # make faiss available
from scipy import stats

fps = 3
debug_iter = 30
R_at_N_tiou_thres = 0.5
noisy_label = "noisy_activity"
retrieval_type_noise = "noise"


from data_generate.activitynet_label_100_20_80 import json_path,longvideo_json_path

from datasets.dongzhuoyao_utils import read_video,read_activitynet

def _pre_process(cur_video_list, input_size, test_frame_num):#time-consuming
    torch_list = []
    for cur_video in cur_video_list:
        start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(cur_video)
        cur_video['frame_info']=dict(
            frame_path=frame_path,
            frame_num=frame_num,
            start_frame_idx=start_frame_idx
        )
        if frame_num == 0:raise#bad data in activitynet-validation
        images = read_video(frame_path=frame_path, start_frame_idx=start_frame_idx,
                            gt_frame_num=frame_num, train_frame_num=test_frame_num,
                            video_transform=transforms.Compose([
                            videotransforms.CenterCrop(input_size),
                            #videotransforms.RandomHorizontalFlip()
                                ]),
                            activitynet_frame_num=activitynet_frame_num)
        images = torch.from_numpy(images).float().unsqueeze(0)
        torch_list.append(images)

    return torch.cat(torch_list, 0)

def Average(lst):
    return sum(lst) / (len(lst)+1e-10)

def generate_multi_query(query_list):
    logger.warn("generate multi query")
    random.seed(620)
    cls_dict = {}
    for q in query_list:
        if q['label'] not in cls_dict.keys():
            cls_dict[q['label']]  = [q]
        else:
            cls_dict[q['label']].append(q)

    new_query_list = []
    for i, q in enumerate(query_list):
        same_type_videos = cls_dict[q['label']]
        same_type_videos = [_ for _ in same_type_videos if _['video_id']!= q['video_id']]
        extra = random.choices(same_type_videos,k=4)
        tmp = [q]
        tmp.extend(extra)
        new_query_list.append(tmp)

    logger.warning("randomness check: {}".format([q['video_id'] for q in new_query_list[0]]))
    return new_query_list

class class_map():
    def __init__(self,query_list):
        self.class_dict = dict()
        self.class_agnostic_ap = []
        self.base_classes = []
        self.novel_classes = []
        self.r_at_n = [30, 50, 100]
        self.full_retrieval_top = {
            str(self.r_at_n[0]): list(),
            str(self.r_at_n[1]): list(),
            str(self.r_at_n[2]): list(),
        }
        self.base_retrieval_top = {
            str(self.r_at_n[0]): list(),
            str(self.r_at_n[1]): list(),
            str(self.r_at_n[2]): list(),
        }
        self.novel_retrieval_top = {
            str(self.r_at_n[0]): list(),
            str(self.r_at_n[1]): list(),
            str(self.r_at_n[2]): list(),
        }

        self.avg2_full_retrieval_top = {
            str(self.r_at_n[0]): dict(),
            str(self.r_at_n[1]): dict(),
            str(self.r_at_n[2]): dict(),
        }

        #init

        self.cm_gt_labels = []
        self.cm_labels = []
        self.query_duration_map_dict = dict()
        self.top30_result_list =list()
        self.system_ap_dict = dict(y_true=list(),
                                   y_pred=list(),
                                   base_y_true=list(),
                                   base_y_pred=list(),
                                   novel_y_true=list(),
                                   novel_y_pred=list())

        self.set_class_info(query_list)

    def set_class_info(self, query_list):
        for q in query_list:
            if q['retrieval_type'] == 'base':
                self.base_classes.append(q['label'])
            elif  q['retrieval_type'] == 'novel':
                self.novel_classes.append(q['label'])
            else:raise

    def mp_add2dict(self, single_query_hit):
        single_query_hit = [d for d in single_query_hit if not d['ignore']]
        y_true = np.array([d['tp'] for d in single_query_hit])
        y_true[-1] = 1  # for robust mAP
        y_scores = np.array([d['score'] for d in single_query_hit])
        # y_true = np.array([d['tp'] for d in single_query_hit])
        # y_scores = np.array([d['score'] for d in single_query_hit])
        ap = average_precision_score(y_true, y_scores)
        #######
        recall_list = []
        for _thres in self.r_at_n:
            single_query_npos = sum([i['tp'] for i in single_query_hit])+1e-10
            cur_recall = sum([i['tp'] for i in single_query_hit[:_thres]]) * 1.0 / single_query_npos
            recall_list.append(cur_recall)
        return dict(ap=ap, recall_list=recall_list)

    def mp_add2dict_update(self,cls_name, retrieval_type, r_dict):
        ap = r_dict['ap']
        recall_list = r_dict['recall_list']


        self.class_agnostic_ap.append(ap)
        if cls_name in self.class_dict.keys():
            self.class_dict[cls_name].append(ap)
        else:
            self.class_dict[cls_name] = [ap]
        #######
        for _, _thres in enumerate(self.r_at_n):
            self.full_retrieval_top[str(_thres)].append(recall_list[_])

            if cls_name not in self.avg2_full_retrieval_top[str(_thres)]:
                self.avg2_full_retrieval_top[str(_thres)][cls_name] = []
            self.avg2_full_retrieval_top[str(_thres)][cls_name].append(recall_list[_])

            if retrieval_type == 'base':
                self.base_retrieval_top[str(_thres)].append(recall_list[_])
            elif retrieval_type == 'novel':
                self.novel_retrieval_top[str(_thres)].append(recall_list[_])
            else:
                raise





    def add2dict(self,cls_name, retrieval_type, single_query_hit):
        single_query_hit = [d for d in single_query_hit if not d['ignore']]
        y_true = np.array([d['tp'] for d in single_query_hit])
        y_true[-1] = 1#for robust mAP
        y_scores = np.array([d['score'] for d in single_query_hit])
        #y_true = np.array([d['tp'] for d in single_query_hit])
        #y_scores = np.array([d['score'] for d in single_query_hit])
        #confusion matrix
        if  'gt_label' in single_query_hit[0]:
            self.cm_gt_labels.extend([i['gt_label'] for i in single_query_hit[:100]])
        if 'label'  in single_query_hit[0]:
            self.cm_labels.extend([i['label'] for i in single_query_hit[:100]])
        if 'query_frame_info' in single_query_hit[0] and 'candidate_frame_info' in single_query_hit[0]:
            self.top30_result_list.append(single_query_hit[:30])


        self.system_ap_dict['y_true'].extend(y_true.tolist())
        self.system_ap_dict['y_pred'].extend(y_scores.tolist())
        if retrieval_type=="base":
            self.system_ap_dict['base_y_true'].extend(y_true.tolist())
            self.system_ap_dict['base_y_pred'].extend(y_scores.tolist())
        elif  retrieval_type=="novel":
            self.system_ap_dict['novel_y_true'].extend(y_true.tolist())
            self.system_ap_dict['novel_y_pred'].extend(y_scores.tolist())
        else:
            raise
        # confusion matrix

        ap = average_precision_score(y_true, y_scores)

        if 'query_duration_sec' in single_query_hit[0]:
            self.query_duration_map_dict[single_query_hit[0]['query_duration_sec']] = ap

        self.class_agnostic_ap.append(ap)

        if cls_name in self.class_dict.keys():
            self.class_dict[cls_name].append(ap)
        else:
            self.class_dict[cls_name] = [ap]
        #######
        for _thres in self.r_at_n:
            single_query_npos = sum([i['tp'] for i in single_query_hit])+1e-10
            cur_recall = sum([i['tp'] for i in single_query_hit[:_thres]]) * 1.0 / single_query_npos

            self.full_retrieval_top[str(_thres)].append(cur_recall)

            if cls_name not in self.avg2_full_retrieval_top[str(_thres)]:
                self.avg2_full_retrieval_top[str(_thres)][cls_name] = []
            self.avg2_full_retrieval_top[str(_thres)][cls_name].append(cur_recall)

            if retrieval_type == 'base':
                self.base_retrieval_top[str(_thres)].append(cur_recall)
            elif retrieval_type == 'novel':
                self.novel_retrieval_top[str(_thres)].append(cur_recall)
            else:
                raise


    def get_result(self, pass_content=[]):
        for _thres in self.r_at_n:
            self.full_retrieval_top[str(_thres)] = Average(self.full_retrieval_top[str(_thres)])
            self.base_retrieval_top[str(_thres)] = Average(self.base_retrieval_top[str(_thres)])
            self.novel_retrieval_top[str(_thres)] = Average(self.novel_retrieval_top[str(_thres)])
            logger.warning("avg1 R@{}={}".format(str(_thres), self.full_retrieval_top[str(_thres)]))
            logger.warning("avg1 base R@{}={}".format(str(_thres), self.base_retrieval_top[str(_thres)]))
            logger.warning("avg1 novel R@{}={}".format(str(_thres), self.novel_retrieval_top[str(_thres)]))

            for cls_name, ap_list in self.avg2_full_retrieval_top[str(_thres)].items():
                self.avg2_full_retrieval_top[str(_thres)][cls_name] = Average(ap_list)#list to value

            base_recall = Average([self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in self.base_classes])
            novel_recall = Average([self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in self.novel_classes])
            avg_recall = Average([self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in (self.novel_classes+self.base_classes)])

            logger.warning("avg2 R@{}={}".format(str(_thres), avg_recall))
            logger.warning("avg2 base R@{}={}".format(str(_thres), base_recall))
            logger.warning("avg2 novel R@{}={}".format(str(_thres), novel_recall))
            logger.warn("-"*30)


        ################
        base_ap_list = [];novel_ap_list = []
        for _cls_name in self.base_classes:
            base_ap_list.extend(self.class_dict[_cls_name])

        for _cls_name in self.novel_classes:
            novel_ap_list.extend(self.class_dict[_cls_name])

        for cls_name, ap_list  in self.class_dict.items():
            self.class_dict[cls_name] = Average(ap_list)


        class_specific_map =  Average(list(self.class_dict.values()))
        class_specific_base_map = Average([self.class_dict[_cls_name] for _cls_name in self.base_classes])
        class_specific_novel_map = Average([self.class_dict[_cls_name] for _cls_name in self.novel_classes])
        hmean = stats.hmean([class_specific_base_map+1e-10, class_specific_novel_map+1e-10])

        class_agnostic_map = Average(self.class_agnostic_ap)

        avg1_class_specific_map = Average(base_ap_list+novel_ap_list)
        avg1_class_specific_base_map = Average(base_ap_list)
        avg1_class_specific_novel_map = Average(novel_ap_list)
        avg1_hmean = stats.hmean([avg1_class_specific_base_map+1e-10, avg1_class_specific_novel_map+1e-10])

        logger.warn("*" * 30)
        logger.warning("avg2 harmonic map={}".format(hmean))
        logger.warning("avg2 class_specific_base_map={}".format(class_specific_base_map))
        logger.warning("avg2 class_specific_novel_map={}".format(class_specific_novel_map))
        logger.warning("avg2 class_specific_map={}".format(class_specific_map))
        logger.warning("avg1 class_agnostic_map(doubtful metric,contain val class?)={}".format(class_agnostic_map))

        logger.warning("(report metric)avg1 harmonic map={}".format(avg1_hmean))
        logger.warning("(report metric)avg1 class_specific_base_map={}".format(avg1_class_specific_base_map))
        logger.warning("(report metric)avg1 class_specific_novel_map={}".format(avg1_class_specific_novel_map))
        logger.warning("avg1 class_specific_map={}".format(avg1_class_specific_map))
        logger.warning(json_path)
        logger.warning(longvideo_json_path)

        #save confusion matrix related json.


        cm_dict = dict(
            gt_labels=self.cm_gt_labels,
            label = self.cm_labels,
            base_classes = self.base_classes,
            novel_classes = self.novel_classes,
            query_duration_map_dict = self.query_duration_map_dict,
            system_ap_dict = self.system_ap_dict,
            class_map_dict = self.class_dict,
            top30_result_list = self.top30_result_list,
            pass_content=pass_content,
            )


        return dict(
            hmean_ap = hmean,
            ap=hmean,
            class_specific_map= class_specific_map,
            base_map = class_specific_base_map,
            novel_map = class_specific_novel_map,
            old_ap = class_agnostic_map,
            recall = self.full_retrieval_top,
            base_recall=self.base_retrieval_top,
            novel_recall=self.novel_retrieval_top,
            cm_dict=cm_dict,
        )


class ARV_Retrieval():
    def __init__(self,args, feat_extract_func,metric_func=None,split="validation",threshold=0.5,drawRP=False,):

        self.feat_extract_func = feat_extract_func
        self.split = args.eval_split
        self.args = args
        self.test_batch_size = args.test_batch_size
        self.input_size = args.input_size
        if args.test_frame_num == None:
            self.test_frame_num = 128
            logger.warn("test_frame_num is None, set default 128")
        else:
            self.test_frame_num = args.test_frame_num
        self.feat_dim = args.metric_feat_dim
        self.use_faiss = args.use_faiss
        self.load_data()
        logger.info("loading {} data: {}".format(self.split, len(self.data_list[self.split])))
        logger.warn("memory_leak_debug={}".format(args.memory_leak_debug))
        query_num = 1
        if "query_num" in self.args:
            query_num = self.args.query_num
        logger.warning("query_num: {}".format(query_num))

    def load_data(self):
        if self.split == 'training':
            raise
            self.data_dict = json.load(open(json_path))
            self.data_list = dict(training=list(), testing=list(), validation=list())
            for k, v in self.data_dict[self.split].items():
                    if k==noisy_label:
                        self.data_list[self.split].extend(v[:600])#experience value
                    else:
                        self.data_list[self.split].extend(v[:10])  #for novel class, the real video number is less than 10.
            return

        self.data_dict = json.load(open(json_path))
        self.data_list = dict(training=list(), testing=list(), validation=list())
        for k, v in self.data_dict[self.split].items():
                self.data_list[self.split].extend(v)


    def extract_item_feature(self):
        import pickle
        cache_path = os.path.join(logger.get_logger_dir(),"feat_cache.pkl")
        if os.path.exists(cache_path) and  self.args.read_cache_feat:
            file = open(cache_path, 'rb')
            object_file = pickle.load(file)
            self.query_list = object_file['query_list']
            self.gallery_list = object_file['gallery_list']
            logger.warn("load cache_feat from {}".format(cache_path))

        else:
            cur_list = list()
            def chunks(l, n):
                # For item i in a range that is a length of l,
                for i in range(0, len(l), n):
                    # Create an index range for l of n items:
                    yield l[i:i + n]
            chunk_list = list(chunks(self.data_list[self.split],self.test_batch_size))
            for idxx, data_batch in tqdm(enumerate(chunk_list), total=len(chunk_list), desc="{}: extracting feat, batch size:{}x{}".format(self.split, self.test_batch_size,len(chunk_list))):
                if self.args.debug  and idxx > 5:break
                if not self.args.memory_leak_debug:
                    img = _pre_process(data_batch, self.input_size, self.test_frame_num)
                    feat = self.feat_extract_func(img)
                else:
                    feat = np.random.rand(len(data_batch), self.feat_dim).astype(np.float32)
                for i in range(len(data_batch)):
                    data_batch[i]['feat'] = sklearn_preprocessing.normalize(np.mean(feat[i].reshape(feat[i].shape[0],-1), axis=-1).reshape(1, -1))#set value
                cur_list.extend(data_batch)


            self.query_list = []
            for q in cur_list:
                if q['retrieval_type']!=retrieval_type_noise:
                    self.query_list.append(q)

            self.gallery_list = cur_list

            with open(cache_path, 'wb') as fp:
                logger.warn("dump cache_feat to {}".format(cache_path))
                pickle.dump(dict(query_list=self.query_list,gallery_list=self.gallery_list), fp)

        if self.use_faiss:
            #res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(self.feat_dim)  # build the index
            xb = []
            for _g in self.gallery_list:
                xb.append(_g['feat'].reshape(1,-1))
            xb = np.concatenate(xb, axis=0)
            index.add(xb)  # add vectors to the index
            logger.info("faiss index.ntotal: {}".format(index.ntotal))

            self.faiss_index = index
            self.faiss_xb = xb


    def ranking(self,):
        logger.warn(
            "start ranking, query size={}, gallery size={}".format(len(self.query_list), len(self.gallery_list)))

        self.class_map_evaluation = class_map(self.query_list)
        self.original_query_list = self.query_list
        self.query_list = generate_multi_query(self.query_list)

        for _, queries in tqdm(enumerate(self.query_list), total=len(self.query_list), desc="{}: ranking".format(self.split)):
                query = queries[0]
                ignore_videoid_list = [q['video_id'] for q in queries]
                #ignore_videoid_list = query['video_id']
                assert query['retrieval_type'] != retrieval_type_noise
                gt_label = query['label']
                single_query_hit = list()

                for idx, candidate in enumerate(self.gallery_list):
                    scored_dict = {}

                    scored_dict['label'] = candidate['label']
                    scored_dict['gt_label'] = gt_label
                    scored_dict['query_duration_sec'] = query['segment'][1] - query['segment'][0]

                    scored_dict['query_frame_info'] = query['frame_info']
                    scored_dict['candidate_frame_info'] = candidate['frame_info']
                    if candidate['video_id'] in ignore_videoid_list:
                        scored_dict['ignore'] = True
                    else:
                        scored_dict['ignore'] = False

                    if candidate['label'] == gt_label:
                        scored_dict['tp'] = 1
                    else:
                        scored_dict['tp'] = 0
                    single_query_hit.append(scored_dict)

                if "query_num" in self.args:
                    query_feat = 0
                    for _iiii in range(self.args.query_num):
                        query_feat += queries[_iiii]['feat']
                    query_feat /= self.args.query_num
                else:
                    query_feat = query['feat']
                D, I = self.faiss_index.search(query_feat, k=self.faiss_xb.shape[0])
                D = np.squeeze(D)
                I = np.squeeze(I)

                tmp_single_query_hit = []
                for j in range(I.shape[0]):
                    _j_th = I[j]
                    single_query_hit[_j_th]['score'] = -D[j]
                    tmp_single_query_hit.append(single_query_hit[_j_th])  # ranking
                single_query_hit = tmp_single_query_hit

                self.class_map_evaluation.add2dict(cls_name=gt_label,retrieval_type=query['retrieval_type'],single_query_hit=single_query_hit)

        _ = self.class_map_evaluation.get_result(self.original_query_list)
        return _



    def evaluation(self):
        self.extract_item_feature()
        d = self.ranking()
        return d


if __name__ == '__main__':
    def demo_feat_extract(input):
        return 1
    def demo_metric_cal(x, y):
        return 1
    arv_retrieval = ARV_Retrieval(feat_extract_func=demo_feat_extract, metric_func=demo_metric_cal)
    d = arv_retrieval.evaluation()
    print(d)
