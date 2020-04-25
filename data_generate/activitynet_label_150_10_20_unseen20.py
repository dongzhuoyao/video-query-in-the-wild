from data_generate.activitynet_label import activitynet_label_list
import random,os


random.seed(620)
random.shuffle(activitynet_label_list)
arv_train_label = activitynet_label_list[:150]
arv_val_label = activitynet_label_list[150:160]
arv_test_label = activitynet_label_list[160:180]
arv_unseen_label = activitynet_label_list[180:]

short_name = "150_10_20_unseen20"
json_name = "arv_db_{}.json".format(short_name)
moment_json_name  = "arv_db_{}_untrimmed.json".format(short_name)

json_path = os.path.join("data_generate",json_name)
moment_eval_json_path = os.path.join("data_generate",moment_json_name)


