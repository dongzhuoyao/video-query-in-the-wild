import torch,os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import datasets.video_transforms as videotransforms
from tqdm import tqdm
import random, json
import pytorchgo_logger as logger
from sklearn.metrics import average_precision_score
import faiss
from scipy import stats

fps = 3
debug_iter = 30
R_at_N_tiou_thres = 0.5
NOISE_LABEL = "distractor_activity"
RETRIEVAL_TYPE_NOISE = "noise"


from dongzhuoyao_utils import fps,noisy_label,activtynet_fps3_path,read_video,read_activitynet,dataset_config
class VRActivityNet(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.transform = transforms.Compose([
                videotransforms.RandomCrop(args.input_size),
            ])
        self.input_size = args.input_size
        self.fps = fps
        self.train_frame = args.train_frame
        self.split = "training"
        self.novel_num = args.novel_num

        self._data = self.load_data()
        self.sanity_check()
        l = 0
        for cls_name in self.data_dict[self.split]:
            if cls_name != noisy_label:
                l += len(self.data_dict[self.split][cls_name])
        self.length = l


    def sanity_check(self):
        for cls_name in self.data_dict[self.split]:
            _new_list = []
            _removed_dict = {}
            for d in self.data_dict[self.split][cls_name]:
                activitynet_subset = d['activitynet_subset']
                video_id = d['video_id']
                if os.path.isdir(os.path.join(activtynet_fps3_path,activitynet_subset,video_id)):
                    _new_list.append(d)
                    continue
                _removed_dict[video_id]="shit"
            self.data_dict[self.split][cls_name] = _new_list
        print("sanity check, removing {} items".format(len(list(_removed_dict.keys()))))

    def load_data(self):
        self.data_dict = json.load(open(dataset_config[self.args.meta_split]['json_path']))
        print("load data done.")
        new_dict = {}
        self.cur_label_list = []
        for cls_name, item_list in self.data_dict[self.split].items():
            if cls_name == noisy_label:continue
            if cls_name in dataset_config[self.args.meta_split]['arv_train_label']:
                new_dict[cls_name] = item_list
            else:#only keep minimal novel class
                new_dict[cls_name] = item_list[:self.novel_num]
            self.cur_label_list.append(cls_name)
            if "noisy" in cls_name:
                print("a")

        self.data_dict[self.split]=new_dict#remove novel and noisy label
        self.cls2int = {label: i for i, label in enumerate(self.cur_label_list)}
        assert len(list(self.cls2int.keys())) == self.args.nclass

        if "d300" in self.args.semantic_json:
            self.semantic_mem = np.zeros((self.args.nclass, 300), dtype=np.float32)
        elif "d200" in self.args.semantic_json:
            self.semantic_mem = np.zeros((self.args.nclass, 200), dtype=np.float32)
        elif "d1024" in self.args.semantic_json:
            self.semantic_mem = np.zeros((self.args.nclass, 1024), dtype=np.float32)
        else:
            raise
        self.label2word_embed = json.load(open(self.args.semantic_json))
        from sklearn import preprocessing as sklearn_preprocessing
        for label_name in self.label2word_embed.keys():
            id = self.cls2int[label_name]
            tmp = sklearn_preprocessing.normalize(
                np.array(self.label2word_embed[label_name]).reshape(1, -1))  # L2 Norm!!!!!!!!!!!
            self.semantic_mem[id, :] = tmp
            assert tmp.max() <= 1 and tmp.min() >= -1
        self.semantic_mem = torch.from_numpy(self.semantic_mem).float()


    def __getitem__(self, index):

        anchor_class_name = random.choice(list(set(self.cur_label_list)))
        negative_class_name = random.choice(list(set(self.cur_label_list)-set(list(anchor_class_name))))
        meta =  {}
        meta['do_not_collate'] = True

        def _read(video_dict):
            assert  video_dict['label'] != noisy_label
            start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(video_dict)
            images = read_video(frame_path=frame_path, start_frame_idx=start_frame_idx,
                                gt_frame_num=frame_num, train_frame_num=self.train_frame,
                                video_transform=self.transform,
                                activitynet_frame_num=activitynet_frame_num)
            return images
        if len(self.data_dict[self.split][anchor_class_name]) >=2:
            anchor, positive = random.sample(self.data_dict[self.split][anchor_class_name],2)
        else:#if only one instance per class
            anchor = positive = self.data_dict[self.split][anchor_class_name][0]
        negative = random.sample(self.data_dict[self.split][negative_class_name],1)[0]

        anchor = _read(anchor)[np.newaxis, :]
        positive = _read(positive)[np.newaxis, :]
        negative = _read(negative)[np.newaxis, :]

        triple = np.concatenate([anchor,positive,negative],axis=0)
        meta['labels']=[self.cls2int[anchor_class_name], self.cls2int[anchor_class_name], self.cls2int[negative_class_name]]
        meta['label_names'] = [anchor_class_name, anchor_class_name, negative_class_name]
        meta['semantic_mem'] = self.semantic_mem
        return triple, meta

    def __len__(self):

        return self.length//3


def get_my_dataset(args):
    from torch.utils.data.dataloader import default_collate
    import collections
    def my_collate(batch):
        if isinstance(batch[0], collections.Mapping) and 'do_not_collate' in batch[0]:
            return batch
        if isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [my_collate(samples) for samples in transposed]
        else:
            return default_collate(batch)

    train_loader = torch.utils.data.DataLoader(
        VRActivityNet(args=args), batch_size=args.batch_size, collate_fn=my_collate, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=False)
    return train_loader


#####################################evaluation data part#####################



def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]



def _pre_process(cur_video_list, input_size, test_frame_num):  # time-consuming
    torch_list = []
    for cur_video in cur_video_list:
        start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(cur_video)
        cur_video['frame_info'] = dict(
            frame_path=frame_path,
            frame_num=frame_num,
            start_frame_idx=start_frame_idx
        )
        if frame_num == 0: raise  # bad data in activitynet-validation
        images = read_video(frame_path=frame_path, start_frame_idx=start_frame_idx,
                            gt_frame_num=frame_num, train_frame_num=test_frame_num,
                            video_transform=transforms.Compose([
                                videotransforms.CenterCrop(input_size),
                                # videotransforms.RandomHorizontalFlip()
                            ]),
                            activitynet_frame_num=activitynet_frame_num)
        images = torch.from_numpy(images).float().unsqueeze(0)
        torch_list.append(images)

    return torch.cat(torch_list, 0)


def Average(lst):
    return sum(lst) / (len(lst) + 1e-10)


def generate_multi_query(query_list):
    logger.warning("generate multi query")
    random.seed(620)
    cls_dict = {}
    for q in query_list:
        if q['label'] not in cls_dict.keys():
            cls_dict[q['label']] = [q]
        else:
            cls_dict[q['label']].append(q)

    new_query_list = []
    for i, q in enumerate(query_list):
        same_type_videos = cls_dict[q['label']]
        same_type_videos = [_ for _ in same_type_videos if _['video_id'] != q['video_id']]
        extra = random.choices(same_type_videos, k=4)
        tmp = [q]
        tmp.extend(extra)
        new_query_list.append(tmp)

    logger.warning("randomness check: {}".format([q['video_id'] for q in new_query_list[0]]))
    return new_query_list


class evaluation_metric():
    def __init__(self, args, query_list):
        self.args = args
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

        # init

        self.cm_gt_labels = []
        self.cm_labels = []
        self.query_duration_map_dict = dict()
        self.top30_result_list = list()
        self.system_ap_dict = dict(y_true=list(),
                                   y_pred=list(),
                                   base_y_true=list(),
                                   base_y_pred=list(),
                                   novel_y_true=list(),
                                   novel_y_pred=list())
        self.set_class_info(query_list)

    def set_class_info(self, query_list):
        for q in query_list:
            q = q[0]
            assert q['label'] in dataset_config[self.args.meta_split]['arv_train_label']+dataset_config[self.args.meta_split]['arv_test_label']#only allow query video from train, test classes.
            if q['retrieval_type'] == 'base':
                self.base_classes.append(q['label'])
            elif q['retrieval_type'] == 'novel':
                self.novel_classes.append(q['label'])
            else:
                raise

    def multiprocess_calculate(self, single_query_hit):
        single_query_hit = [d for d in single_query_hit if not d['ignore']]
        y_true = np.array([d['tp'] for d in single_query_hit]);y_true[-1] = 1  # for robust mAP
        y_scores = np.array([d['score'] for d in single_query_hit])
        ap = average_precision_score(y_true, y_scores)
        #######
        recall_list = []
        for _thres in self.r_at_n:
            single_query_npos = sum([i['tp'] for i in single_query_hit]) + 1e-10
            cur_recall = sum([i['tp'] for i in single_query_hit[:_thres]]) * 1.0 / single_query_npos
            recall_list.append(cur_recall)
        return dict(ap=ap, recall_list=recall_list)

    def multiprocess_update(self, cls_name, retrieval_type, r_dict):
        ap = r_dict['ap']
        recall_list = r_dict['recall_list']

        #####update ap
        self.class_agnostic_ap.append(ap)
        if cls_name in self.class_dict.keys():
            self.class_dict[cls_name].append(ap)
        else:
            self.class_dict[cls_name] = [ap]
        #######update recall list
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

    def add2dict(self, cls_name, retrieval_type, single_query_hit):
        single_query_hit = [d for d in single_query_hit if not d['ignore']]
        y_true = np.array([d['tp'] for d in single_query_hit]);y_true[-1] = 1  # for robust mAP
        y_scores = np.array([d['score'] for d in single_query_hit])
        # confusion matrix
        if 'gt_label' in single_query_hit[0]:
            self.cm_gt_labels.extend([i['gt_label'] for i in single_query_hit[:100]])
        if 'label' in single_query_hit[0]:
            self.cm_labels.extend([i['label'] for i in single_query_hit[:100]])
        if 'query_frame_info' in single_query_hit[0] and 'candidate_frame_info' in single_query_hit[0]:
            self.top30_result_list.append(single_query_hit[:30])

        self.system_ap_dict['y_true'].extend(y_true.tolist())
        self.system_ap_dict['y_pred'].extend(y_scores.tolist())
        if retrieval_type == "base":
            self.system_ap_dict['base_y_true'].extend(y_true.tolist())
            self.system_ap_dict['base_y_pred'].extend(y_scores.tolist())
        elif retrieval_type == "novel":
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
            single_query_npos = sum([i['tp'] for i in single_query_hit]) + 1e-10
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
            logger.warning("1-order R@{}={}".format(str(_thres), self.full_retrieval_top[str(_thres)]))
            logger.warning("1-order base R@{}={}".format(str(_thres), self.base_retrieval_top[str(_thres)]))
            logger.warning("1-order novel R@{}={}".format(str(_thres), self.novel_retrieval_top[str(_thres)]))

            for cls_name, ap_list in self.avg2_full_retrieval_top[str(_thres)].items():
                self.avg2_full_retrieval_top[str(_thres)][cls_name] = Average(ap_list)  # list to value

            base_recall = Average(
                [self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in self.base_classes])
            novel_recall = Average(
                [self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in self.novel_classes])
            avg_recall = Average([self.avg2_full_retrieval_top[str(_thres)][_cls_name] for _cls_name in
                                  (self.novel_classes + self.base_classes)])

            logger.warning("2-order R@{}={}".format(str(_thres), avg_recall))
            logger.warning("2-order base R@{}={}".format(str(_thres), base_recall))
            logger.warning("2-order novel R@{}={}".format(str(_thres), novel_recall))
            logger.warning("-" * 30)

        ################
        base_ap_list = [];
        novel_ap_list = []
        for _cls_name in self.base_classes:
            base_ap_list.extend(self.class_dict[_cls_name])

        for _cls_name in self.novel_classes:
            novel_ap_list.extend(self.class_dict[_cls_name])

        for cls_name, ap_list in self.class_dict.items():
            self.class_dict[cls_name] = Average(ap_list)


        o1_class_agnostic_map = Average(self.class_agnostic_ap)
        o1_class_specific_map = Average(base_ap_list + novel_ap_list)
        o1_class_specific_base_map = Average(base_ap_list)
        o1_class_specific_novel_map = Average(novel_ap_list)
        o1_hmean = stats.hmean([o1_class_specific_base_map + 1e-10, o1_class_specific_novel_map + 1e-10])

        o2_class_specific_map = Average(list(self.class_dict.values()))
        o2_class_specific_base_map = Average([self.class_dict[_cls_name] for _cls_name in self.base_classes])
        o2_class_specific_novel_map = Average([self.class_dict[_cls_name] for _cls_name in self.novel_classes])
        o2_hmean = stats.hmean([o2_class_specific_base_map + 1e-10, o2_class_specific_novel_map + 1e-10])

        logger.warning("*" * 30)
        logger.warning("1-order harmonic map={}".format(o1_hmean))
        logger.warning("1-order class_specific_base_map={}".format(o1_class_specific_base_map))
        logger.warning("1-order class_specific_novel_map={}".format(o1_class_specific_novel_map))
        logger.warning("1-order class_specific_map={}".format(o1_class_specific_map))
        logger.warning("1-order class_agnostic_map={}".format(o1_class_agnostic_map))

        logger.warning("(report metric)2-order harmonic map={}".format(o2_hmean))
        logger.warning("(report metric)2-order class_specific_base_map={}".format(o2_class_specific_base_map))
        logger.warning("(report metric)2-order class_specific_novel_map={}".format(o2_class_specific_novel_map))
        logger.warning("2-order class_specific_map={}".format(o2_class_specific_map))
        logger.warning(dataset_config[self.args.meta_split]['json_path'])
        logger.warning(dataset_config[self.args.meta_split]['moment_eval_json_path'])

        # save confusion matrix related json.
        cm_dict = dict(
            gt_labels=self.cm_gt_labels,
            label=self.cm_labels,
            base_classes=self.base_classes,
            novel_classes=self.novel_classes,
            query_duration_map_dict=self.query_duration_map_dict,
            system_ap_dict=self.system_ap_dict,
            class_map_dict=self.class_dict,
            top30_result_list=self.top30_result_list,
            pass_content=pass_content,
        )

        return dict(
            ap=o2_hmean,
            base_map=o2_class_specific_base_map,
            novel_map=o2_class_specific_novel_map,
            recall=self.full_retrieval_top,
            base_recall=self.base_retrieval_top,
            novel_recall=self.novel_retrieval_top,
            cm_dict=cm_dict,
        )


class ARV_Retrieval_Clip():
    def __init__(self, args, feat_extract_func):
        self.args = args
        self.temporal_stride = args.temporal_stride
        self.feat_extract_func = feat_extract_func
        self.feat_dim = args.metric_feat_dim
        self.clip_sec = args.clip_sec
        self.test_frame_num = args.test_frame_num
        self.test_batch_size = args.test_batch_size
        self.input_size = args.input_size
        self.possible_classes = dataset_config[self.args.meta_split]['arv_train_label']+dataset_config[self.args.meta_split]['arv_test_label']
        self.load_data()
        logger.warning("memory_leak_debug={}".format(args.memory_leak_debug))
        logger.warning("query_num: {}".format(self.args.query_num))

    def load_data(self):
        data_dict = json.load(open(dataset_config[self.args.meta_split]['moment_eval_json_path']))
        self.query_list = []
        for q in data_dict['query']:
            if q['retrieval_type'] == RETRIEVAL_TYPE_NOISE:
                continue
            self.query_list.append(q)
        self.gallery_list = data_dict['gallery']

        logger.warning("query length={}, gallery size={}".format(len(self.query_list), len(self.gallery_list)))

    def extract_item_feature(self):
        import pickle
        cache_path = os.path.join(logger.get_logger_dir(), "clip_feat_cache.pkl")
        if os.path.exists(cache_path) and self.args.read_cache_feat:
            file = open(cache_path, 'rb')
            object_file = pickle.load(file)
            self.query_list = object_file['query_list']
            self.gallery_list = object_file['gallery_list']
            logger.warning("load cache_feat from {}".format(cache_path))
        else:
            cur_list = list()
            chunk_list = list(chunks(self.query_list, self.test_batch_size))
            for idxx, data_batch in tqdm(enumerate(chunk_list), total=len(chunk_list),
                                         desc="eval_clips, extracting query feat, batch size:{}x{}".format(
                                             self.test_batch_size, len(chunk_list))):
                if self.args.debug and idxx > debug_iter: break

                if self.args.memory_leak_debug:
                    feat = np.random.rand(len(data_batch), self.feat_dim, self.test_frame_num).astype(np.float32)
                else:
                    img = _pre_process(data_batch, self.input_size, self.test_frame_num)
                    feat = self.feat_extract_func(img)

                assert len(data_batch) == feat.shape[0]
                tpooled_feat = np.mean(feat, axis=-1)
                for i in range(len(data_batch)):
                    data_batch[i]['feat'] = tpooled_feat[i]
                cur_list.extend(data_batch)

            self.query_list = []
            for q in cur_list:
                if  q['label'] in self.possible_classes:
                    self.query_list.append(q)

            for proceeded_id, _g in tqdm(enumerate(self.gallery_list), total=len(self.gallery_list),
                                         desc="eval_clips, extracting gallery feat"):
                if self.args.debug and proceeded_id > debug_iter * 10: break
                start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(_g)
                chunk_list = list(chunks(list(range(activitynet_frame_num)), self.test_frame_num))
                feats_list = []
                for idxx, data_batch in enumerate(chunk_list):
                    if not self.args.memory_leak_debug:
                        images = read_video(frame_path=frame_path, start_frame_idx=data_batch[0],
                                            gt_frame_num=len(data_batch), train_frame_num=self.test_frame_num,
                                            video_transform=transforms.Compose([
                                                videotransforms.CenterCrop(self.input_size),
                                            ]),
                                            activitynet_frame_num=activitynet_frame_num)
                        images = torch.from_numpy(images).float().unsqueeze(0)
                        assert images.shape[1] == self.test_frame_num
                        _feats = self.feat_extract_func(images)  # [B,C,T]
                    else:
                        _feats = np.random.rand(1, self.feat_dim, self.test_frame_num).astype(np.float32)
                    _feats = _feats[0][:, :len(data_batch)]  # truncate when meeting last clip of the long video
                    feats_list.append(_feats)
                _feats = np.concatenate(feats_list, axis=1)
                assert _feats.shape[1] == activitynet_frame_num // self.temporal_stride, \
                    "{} not equal to {}".format(_feats.shape[1], activitynet_frame_num // self.temporal_stride)
                self.gallery_list[proceeded_id]['feat'] = _feats#[C,T]

            self.gallery_list = [g for g in self.gallery_list if 'feat' in g]  # useful when debugging

            def garner_feat(_g, clip_sec=self.clip_sec):
                length = _g['feat'].shape[1]
                r_list = []
                annotations = _g['annotations']

                def cal_label(loc_sec):
                    target_label = "unknown"
                    for ann in annotations:
                        seg = ann['segment']
                        label = ann['label']
                        if seg[0] <= loc_sec[0] and seg[1] >= loc_sec[
                            1] and label in self.possible_classes:  # if inside
                            target_label = label
                            break
                    return target_label

                for clip_start in range(0, int(_g['activitynet_duration']), clip_sec):
                    loc_feat = [clip_start * fps // self.temporal_stride,
                                (clip_start + clip_sec) * fps // self.temporal_stride]
                    loc_sec = [clip_start, clip_start + clip_sec]
                    if loc_feat[1] > length:  # out of boundary, skip it
                        continue
                    r_list.append(dict(
                        feat_indice=loc_feat,
                        feat=
                        np.mean(_g['feat'][:,loc_feat[0]:loc_feat[1]], axis=1),
                        duration_sec=clip_sec,
                        loc_sec=loc_sec,
                        clip_label=cal_label(loc_sec),
                        video_id=_g['video_id'],
                        activitynet_duration=_g['activitynet_duration'],
                        border=_g['border'],
                        segment=_g['segment'],
                    )
                    )
                return r_list

            self.tmp_gallery = list()
            self.old_gallery = self.gallery_list
            for i, _g in tqdm(enumerate(self.gallery_list), desc="compute potential tIoU"):
                if self.args.debug and i > debug_iter * 10: break
                self.tmp_gallery.extend(garner_feat(_g))
            self.gallery_list = self.tmp_gallery

            with open(cache_path, 'wb') as fp:
                logger.warning("dump cache_feat to {}".format(cache_path))
                pickle.dump(dict(query_list=self.query_list, gallery_list=self.gallery_list), fp)

            logger.warning("average #segment={} per candidate video.".format(len(self.gallery_list) / proceeded_id))

        if not self.args.debug and not self.args.memory_leak_debug:
            logger.warning("check class completeness.")
            complete_dict = dict.fromkeys(self.possible_classes, 0)
            for can in self.gallery_list:
                if can['clip_label'] == 'unknown':
                    continue
                complete_dict[can['clip_label']] += 1
            for key, value in complete_dict.items():
                assert value > 0, "{} doesn't exist in gallery!".format(key)

        index = faiss.IndexFlatL2(self.feat_dim)  # build the index
        xb = []
        for _g in self.gallery_list:
            xb.append(_g['feat'].reshape(1, -1))
        xb = np.concatenate(xb, axis=0)
        index.add(xb)  # add vectors to the index
        logger.info("faiss index.ntotal: {}".format(index.ntotal))

        self.faiss_index = index
        self.faiss_xb = xb

    def ranking(self, ):
        logger.warning(
            "start ranking, query size={}, gallery clips size={}".format(len(self.query_list), len(self.gallery_list)))

        if self.args.debug:
            self.query_list = [[i] for i in self.query_list]
        else:
            self.query_list = generate_multi_query(self.query_list)
        self.query_list = [q for q in self.query_list if q[0]['is_query'] == 1]
        self.class_map_evaluation = evaluation_metric(self.args, self.query_list)

        for _ in tqdm(range(len(self.query_list)), total=len(self.query_list), desc="eval_clips, ranking"):
            queries = self.query_list[_]
            query = queries[0]
            ignore_videoid_list = [q['video_id'] for q in queries]

            assert query['retrieval_type'] != RETRIEVAL_TYPE_NOISE
            gt_label = query['label']
            single_query_hit = list()
            for idx, candidate in enumerate(self.gallery_list):
                scored_dict = {}
                scored_dict['gt_label'] = gt_label

                if candidate['clip_label'] == gt_label:  # mainly evaluate R@N at tIoU=0.5
                    scored_dict['tp'] = 1
                else:
                    scored_dict['tp'] = 0

                if candidate['video_id'] in ignore_videoid_list:
                    scored_dict['ignore'] = True
                else:
                    scored_dict['ignore'] = False

                single_query_hit.append(scored_dict)
            if "query_num" in self.args:
                query_feat = 0
                for _iiii in range(self.args.query_num):
                    query_feat += queries[_iiii]['feat']
                query_feat /= self.args.query_num
            else:
                query_feat = query['feat']
            D, I = self.faiss_index.search(query_feat.reshape(1, query_feat.shape[0]), k=self.faiss_xb.shape[0])
            D = np.squeeze(D)
            I = np.squeeze(I)

            tmp_single_query_hit = []
            for j in range(I.shape[0]):
                _j_th = I[j]
                single_query_hit[_j_th]['score'] = -D[j]
                tmp_single_query_hit.append(single_query_hit[_j_th])  # ranking
            single_query_hit = tmp_single_query_hit

            self.class_map_evaluation.add2dict(cls_name=gt_label, retrieval_type=query['retrieval_type'],
                                               single_query_hit=single_query_hit)

        return self.class_map_evaluation.get_result()

    def evaluation(self):
        if "evaluate" in self.args and self.args.evaluate:
            self.extract_item_feature()
            d = self.ranking()
            return d
        else:
            return None


class ARV_Retrieval_Moment():
    def __init__(self, args, feat_extract_func):
        self.args = args
        self.temporal_stride = args.temporal_stride
        self.feat_extract_func = feat_extract_func
        self.test_batch_size = args.test_batch_size
        self.test_frame_num = args.test_frame_num
        self.input_size = args.input_size
        self.possible_classes = dataset_config[self.args.meta_split]['arv_train_label']+dataset_config[self.args.meta_split]['arv_test_label']
        self.feat_dim = args.metric_feat_dim
        self.load_data()
        logger.warning("memory_leak_debug={}".format(args.memory_leak_debug))
        logger.warning("query_num: {}".format(self.args.query_num))

    def load_data(self):
        data_dict = json.load(open(dataset_config[self.args.meta_split]['moment_eval_json_path']))
        query = data_dict['query']
        self.query_list = []
        for q in query:
            if q['retrieval_type'] != RETRIEVAL_TYPE_NOISE:
                self.query_list.append(q)

        self.gallery_list = data_dict['gallery']
        logger.warning("query length={}, gallery size={}".format(len(self.query_list), len(self.gallery_list)))

    def extract_item_feature(self):
        import pickle
        cache_path = os.path.join(logger.get_logger_dir(), "moment_feat_cache.pkl")
        if os.path.exists(cache_path) and self.args.read_cache_feat:
            file = open(cache_path, 'rb')
            object_file = pickle.load(file)
            self.query_list = object_file['query_list']
            self.gallery_list = object_file['gallery_list']
            logger.warning("load cache_feat from {}".format(cache_path))
        else:
            ### extract feature for query video #####
            cur_list = list()
            chunk_list = list(chunks(self.query_list, self.test_batch_size))
            for idxx, data_batch in tqdm(enumerate(chunk_list), total=len(chunk_list),
                                         desc="extracting query feat, batch size:{}x{}".format(
                                             self.test_batch_size,
                                             len(chunk_list))):
                if self.args.debug and idxx > debug_iter: break
                if  self.args.memory_leak_debug:
                    feat = np.random.rand(len(data_batch), self.feat_dim, self.test_frame_num).astype(np.float32)
                else:
                    img = _pre_process(data_batch, self.input_size, self.test_frame_num)
                    feat = self.feat_extract_func(img)
                assert len(data_batch) == feat.shape[0]
                tpooled_feat = np.mean(feat, axis=-1)
                for i in range(len(data_batch)):
                    data_batch[i]['feat'] = tpooled_feat[i]
                cur_list.extend(data_batch)

            self.query_list = []
            for q in cur_list:
                if q['label'] in self.possible_classes:
                    self.query_list.append(q)

            ### extract feature for gallery video #####
            for proceeded_id, _g in tqdm(enumerate(self.gallery_list), total=len(self.gallery_list),
                                         desc="eval_moment, extracting gallery feat"):
                if self.args.debug and proceeded_id > debug_iter * 10: break
                start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(_g)
                chunk_list = list(chunks(list(range(activitynet_frame_num)), self.test_frame_num))
                feats_list = []
                for idxx, data_batch in enumerate(chunk_list):
                    if  self.args.memory_leak_debug:
                        _feats = np.random.rand(1, self.feat_dim, self.test_frame_num).astype(np.float32)
                    else:
                        images = read_video(frame_path=frame_path, start_frame_idx=data_batch[0],
                                            gt_frame_num=len(data_batch), train_frame_num=self.test_frame_num,
                                            video_transform=transforms.Compose([
                                                videotransforms.CenterCrop(self.input_size),
                                            ]),
                                            activitynet_frame_num=activitynet_frame_num)
                        images = torch.from_numpy(images).float().unsqueeze(0)
                        _feats = self.feat_extract_func(images)
                    _feats = _feats[0][:, :len(data_batch)]  # truncate when meeting last clip of the long video
                    feats_list.append(_feats)
                _feats = np.concatenate(feats_list, axis=1)
                assert _feats.shape[1] == activitynet_frame_num // self.temporal_stride
                self.gallery_list[proceeded_id]['feat'] = _feats#[T,C]

            def garner_feat(_g, clip_length_sec=5, max_clip_per_moment=26):
                feat_length = _g['feat'].shape[1]
                r_list = []
                annotations = _g['annotations']

                def cal_iou(min1, max1, min2, max2):
                    overlap = max(0, min(max1, max2) - max(min1, min2))
                    return overlap * 1.0 / (max(max2, max1) - min(min1, min2))
                def cal_hit(loc_sec):
                    best_iou = -1
                    best_result = None
                    for ann in annotations:
                        seg = ann['segment']
                        label = ann['label']
                        iou = cal_iou(seg[0], seg[1], loc_sec[0], loc_sec[1])
                        if iou > best_iou and label in self.possible_classes:
                            best_result = dict(
                                iou=iou,
                                label=label,
                                gt=seg,
                                pred=loc_sec,
                            )
                            best_iou = iou
                    if best_iou == -1:
                        return []
                    else:
                        return [best_result]

                for clips_per_moment in range(1, max_clip_per_moment + 1):
                    for moment_start_sec in range(0,
                                                  int(_g['activitynet_duration']) - clip_length_sec * clips_per_moment,
                                                  clip_length_sec):
                        moment_duration_sec = clip_length_sec * clips_per_moment

                        loc_feat = [moment_start_sec * fps // self.temporal_stride,
                                    (moment_start_sec + moment_duration_sec) * fps // self.temporal_stride]
                        loc_sec = [moment_start_sec, moment_start_sec + moment_duration_sec]
                        # print(loc_sec)
                        if loc_feat[1] > feat_length:
                            continue

                        r_list.append(dict(
                            feat=np.mean(_g['feat'][:,loc_feat[0]:loc_feat[1]], axis=1),
                            video_id=_g['video_id'],
                            hit_list=cal_hit(loc_sec)
                        )
                        )
                return r_list

            self.tmp_gallery = list()
            self.old_gallery = self.gallery_list
            moment_total = []
            for i, _g in tqdm(enumerate(self.gallery_list), desc="compute potential tIoU"):  # time-consuming
                if self.args.debug and i > debug_iter * 10: break
                moments = garner_feat(_g)
                moment_total.append(len(moments))
                self.tmp_gallery.extend(moments)
            self.gallery_list = self.tmp_gallery
            logger.warning("#moment/video = {}".format(sum(moment_total) / len(moment_total)))
            logger.warning("average #segment={} per candidate video.".format(len(self.gallery_list) / proceeded_id))

            with open(cache_path, 'wb') as fp:
                logger.warning("dump cache_feat to {}".format(cache_path))
                pickle.dump(dict(query_list=self.query_list, gallery_list=self.gallery_list), fp)

        logger.warning("check class completeness.")
        complete_dict = dict.fromkeys(self.possible_classes, 0)
        for can in self.gallery_list:
            if len(can['hit_list']) == 0:
                continue
            for hit_list in can['hit_list']:
                complete_dict[hit_list['label']] += 1
        if not self.args.debug:
            for key, value in complete_dict.items():
                assert value > 0, "{} doesn't exist in gallery!".format(key)

        index = faiss.IndexFlatL2(self.feat_dim)  # build the index
        xb = []
        for _g in self.gallery_list:
            xb.append(_g['feat'].reshape(1, -1))
        xb = np.concatenate(xb, axis=0)
        index.add(xb)  # add vectors to the index
        logger.info("faiss index.ntotal: {}".format(index.ntotal))
        self.faiss_index = index
        self.faiss_xb = xb

    def ranking(self, ):
        logger.warning("start ranking, query size={}, gallery potential moments size={}".format(len(self.query_list),
                                                                                             len(self.gallery_list)))

        if self.args.debug:
            self.query_list = [[i] for i in self.query_list]
        else:
            self.query_list = generate_multi_query(self.query_list)

        self.query_list = [q for q in self.query_list if q[0]['is_query']==1]
        self.class_map_evaluation05 = evaluation_metric(self.args, self.query_list)

        from multiprocessing import Process, Queue, JoinableQueue, cpu_count
        def work(id, jobs, result):  # https://gist.github.com/brantfaircloth/1255715/5ce00c58ae8775c7d75a7bc08ab75d5c7f665bca
            while True:
                queries = jobs.get()
                if queries is None:
                    break
                query = queries[0]
                ignore_videoid_list = [q['video_id'] for q in queries]#ignore those long videos, which current short query video comes from..
                assert query['retrieval_type'] != RETRIEVAL_TYPE_NOISE
                gt_label = query['label']
                single_query_hit = list()
                for idx, candidate in enumerate(self.gallery_list):
                    hit_list = candidate['hit_list']
                    scored_dict = {}
                    scored_dict['gt_label'] = gt_label

                    iou = 0
                    assert len(hit_list) <= 1
                    if len(hit_list) == 1 and hit_list[0]['label'] == gt_label:
                        iou = hit_list[0]['iou']
                    scored_dict['iou'] = iou

                    if candidate['video_id'] in ignore_videoid_list:
                        scored_dict['ignore'] = True
                    else:
                        scored_dict['ignore'] = False

                    single_query_hit.append(scored_dict)

                query_feat = 0
                for _iiii in range(self.args.query_num):
                    query_feat += queries[_iiii]['feat']
                query_feat /= self.args.query_num

                D, I = self.faiss_index.search(query_feat.reshape(1, query_feat.shape[0]), k=self.faiss_xb.shape[0])
                D = np.squeeze(D)
                I = np.squeeze(I)
                tmp_single_query_hit = []
                for j in range(I.shape[0]):
                    _j_th = I[j]
                    single_query_hit[_j_th]['score'] = -D[j]
                    tmp_single_query_hit.append(single_query_hit[_j_th])  # ranking
                single_query_hit = tmp_single_query_hit

                for s in single_query_hit:
                    if s['iou'] >= 0.5:
                        s['tp'] = 1
                    else:
                        s['tp'] = 0
                #dongzhuoyao,TODO, can do NMS after thresholding.
                d05 = self.class_map_evaluation05.multiprocess_calculate(
                    single_query_hit=single_query_hit)
                result.put(dict(cls_name=gt_label, retrieval_type=query['retrieval_type'],
                                dict05=d05))

        jobs = Queue()
        result = JoinableQueue()
        NUMBER_OF_PROCESSES = cpu_count() * 1 // 4
        logger.warning("multi processing evaluation: #Process={}".format(NUMBER_OF_PROCESSES))
        for w in self.query_list:
            jobs.put(w)
        [Process(target=work, args=(i, jobs, result)).start() for i in range(NUMBER_OF_PROCESSES)]#start up!

        #get multi-processing result, and send to evaluation
        for _ in tqdm(range(len(self.query_list)), desc="multi-process ranking,#cpu={}".format(NUMBER_OF_PROCESSES),
                      total=len(self.query_list)):#fetch data for len(self.query_list) times
            r = result.get()
            self.class_map_evaluation05.multiprocess_update(cls_name=r['cls_name'],
                                                            retrieval_type=r['retrieval_type'],
                                                            r_dict=r['dict05'])
            result.task_done()

        for w in range(NUMBER_OF_PROCESSES):
            jobs.put(None)

        result.join()
        jobs.close()
        result.close()

        logger.info("mAP05 result:")
        map05 = self.class_map_evaluation05.get_result()
        score_dict = dict(map05=map05)
        return score_dict

    def evaluation(self):
        if "evaluate" in self.args and self.args.evaluate:
            self.extract_item_feature()
            d = self.ranking()
            return d
        else:
            return None


class ARV_Retrieval():
    def __init__(self, args, feat_extract_func):
        self.args = args
        self.feat_extract_func = feat_extract_func
        self.eval_split = args.eval_split# no eval_split for clip, moment retrieval, because they are and only are used for testing.
        self.test_batch_size = args.test_batch_size
        self.input_size = args.input_size
        self.test_frame_num = args.test_frame_num
        self.feat_dim = args.metric_feat_dim
        self.load_data()
        if self.eval_split == "validation":
            self.possible_classes = dataset_config[self.args.meta_split]['arv_train_label'] + \
                                    dataset_config[self.args.meta_split]['arv_val_label']
        elif self.eval_split == "testing":
            self.possible_classes = dataset_config[self.args.meta_split]['arv_train_label'] + \
                                    dataset_config[self.args.meta_split]['arv_test_label']
        else:
            logger.warning("evaluation on training set!")

        logger.info("loading {} data: {}".format(self.eval_split, len(self.data_list[self.eval_split])))
        logger.warning("memory_leak_debug={}".format(args.memory_leak_debug))
        logger.warning("query_num: {}".format(self.args.query_num))

    def load_data(self):
        if self.eval_split == 'training':#evaluate on training data for checking
            raise
            self.data_dict = json.load(open(dataset_config[self.args.meta_split]['json_path']))
            self.data_list = dict(training=list(), testing=list(), validation=list())
            for k, v in self.data_dict[self.split].items():
                if k == NOISE_LABEL:
                    self.data_list[self.split].extend(v[:600])  # experience value
                else:
                    self.data_list[self.split].extend(v[:10])  # for novel class, the real video number is less than 10.
            return

        self.data_dict = json.load(open(dataset_config[self.args.meta_split]['json_path']))
        self.data_list = dict(training=list(), testing=list(), validation=list())
        for k, v in self.data_dict[self.eval_split].items():
            self.data_list[self.eval_split].extend(v)

    def extract_item_feature(self):
        import pickle
        cache_path = os.path.join(logger.get_logger_dir(), "feat_cache.pkl")
        if os.path.exists(cache_path) and self.args.read_cache_feat:
            file = open(cache_path, 'rb')
            object_file = pickle.load(file)
            self.query_list = object_file['query_list']
            self.gallery_list = object_file['gallery_list']
            logger.warning("load cache_feat from {}".format(cache_path))
        else:
            cur_list = list()
            chunk_list = list(chunks(self.data_list[self.eval_split], self.test_batch_size))
            for idxx, data_batch in tqdm(enumerate(chunk_list), total=len(chunk_list),
                                         desc="{}: extracting feat, batch size:{}x{}".format(self.eval_split,
                                                                                             self.test_batch_size,
                                                                                             len(chunk_list))):
                if self.args.debug and idxx > 5: break
                if self.args.memory_leak_debug:
                    feat = np.random.rand(len(data_batch), self.feat_dim, self.test_frame_num).astype(np.float32)
                else:
                    img = _pre_process(data_batch, self.input_size, self.test_frame_num)
                    feat = self.feat_extract_func(img)

                tpooled_feat = np.mean(feat, axis=-1)
                for i in range(len(data_batch)):
                    data_batch[i]['feat'] = tpooled_feat[i]
                cur_list.extend(data_batch)

            self.query_list = []
            for q in cur_list:
                if  q['label'] in self.possible_classes:
                    self.query_list.append(q)

            self.gallery_list = cur_list#for gallery, use all videos
            with open(cache_path, 'wb') as fp:
                logger.warning("dump cache_feat to {}".format(cache_path))
                pickle.dump(dict(query_list=self.query_list, gallery_list=self.gallery_list), fp)

        index = faiss.IndexFlatL2(self.feat_dim)  # build the index
        xb = []
        for _g in self.gallery_list:
            xb.append(_g['feat'].reshape(1, -1))
        xb = np.concatenate(xb, axis=0)
        index.add(xb)  # add vectors to the index
        logger.info("faiss index.ntotal: {}".format(index.ntotal))
        self.faiss_index = index
        self.faiss_xb = xb

    def ranking(self, ):
        logger.warning(
            "start ranking, query size={}, gallery size={}".format(len(self.query_list), len(self.gallery_list)))


        self.original_query_list = self.query_list
        self.query_list = generate_multi_query(self.query_list)
        self.query_list = [q for q in self.query_list if q[0]['is_query'] == 1]
        self.class_map_evaluation = evaluation_metric(self.args, self.query_list)

        for _, queries in tqdm(enumerate(self.query_list), total=len(self.query_list),
                               desc="{}: ranking".format(self.eval_split)):
            query = queries[0]
            ignore_videoid_list = [q['video_id'] for q in queries]
            assert query['retrieval_type'] != RETRIEVAL_TYPE_NOISE
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

                if "subclass_eval" in self.args and self.args.subclass_eval == "ggfather":  # evaluation based on class
                    from hierachy_class_util import is_same_father, is_same_grandfather, is_same_grandgrandfather
                    if is_same_grandgrandfather(candidate['label'], gt_label):
                        scored_dict['tp'] = 1
                    else:
                        scored_dict['tp'] = 0

                if "subclass_eval" in self.args and self.args.subclass_eval == "father":  # evaluation based on class
                    from hierachy_class_util import is_same_father, is_same_grandfather, is_same_grandgrandfather
                    if is_same_father(candidate['label'], gt_label):
                        scored_dict['tp'] = 1
                    else:
                        scored_dict['tp'] = 0

                single_query_hit.append(scored_dict)


            query_feat = 0
            for _iiii in range(self.args.query_num):
                query_feat += queries[_iiii]['feat']
            query_feat /= self.args.query_num
            D, I = self.faiss_index.search(query_feat.reshape(1, query_feat.shape[0]), k=self.faiss_xb.shape[0])
            D = np.squeeze(D)
            I = np.squeeze(I)

            tmp_single_query_hit = []
            for j in range(I.shape[0]):
                _j_th = I[j]
                single_query_hit[_j_th]['score'] = -D[j]
                tmp_single_query_hit.append(single_query_hit[_j_th])  # ranking
            single_query_hit = tmp_single_query_hit

            self.class_map_evaluation.add2dict(cls_name=gt_label, retrieval_type=query['retrieval_type'],
                                               single_query_hit=single_query_hit)
        return self.class_map_evaluation.get_result(self.original_query_list)

    def evaluation(self):
        self.extract_item_feature()
        d = self.ranking()
        return d


if __name__ == '__main__':
    def demo_feat_extract(input):
        return 1
    arv_retrieval = ARV_Retrieval(feat_extract_func=demo_feat_extract)
    d = arv_retrieval.evaluation()
    print(d)


