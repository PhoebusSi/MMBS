import sys
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from collections import defaultdict, Counter
from dataset_vqacp_MMBS import Dictionary, VQAFeatureDataset

from model_MMBS import Model
import utils
import opts_MMBS as opts
from train_MMBS import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)

def entropy(c):
    result = -1
    if(len(c) > 0):
        result = 0
    for x in c:
        if x == 0:
            result += 0
        else:
            result += (-x)*math.log(x, 2)
    return result

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == '__main__':

    opt = opts.parse_opt()
    beta = opt.beta
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    model = Model(opt)
    model = model.cuda()
    model.apply(weights_init_kn)
    model = nn.DataParallel(model)#.cuda()

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)


    answer_voc_size = train_dset.num_ans_candidates

    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score


    question_type_to_prob_array = {}
    question_type_rank_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)

        for label, total_score in question_type_to_probs[q_type].items():
            if q_type in question_type_rank_array.keys():
                if label.item() in question_type_rank_array[q_type].keys():
                    question_type_rank_array[q_type][label.item()] += total_score.item() 
                else:
                    question_type_rank_array[q_type][label.item()] = total_score.item() 

            else:
                question_type_rank_array[q_type] = {}
                question_type_rank_array[q_type][label.item()] = total_score.item() 
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    sorted_question_type_ans_dict = {}
    unbias_ans_4_question_type = {}
    all_score_4_question_type = {}
    all_ans_4_question_type = {}
    for q_type in question_type_rank_array.keys():
        sorted_question_type_ans_dict[q_type] = sorted(question_type_rank_array[q_type].items(), key=lambda item:item[1])
        all_ans_4_question_type[q_type] = [x for x,_ in sorted_question_type_ans_dict[q_type]]
        all_score_4_question_type[q_type] = [x for _,x in sorted_question_type_ans_dict[q_type]]

    norm_score_4_question_type = {}
    entropy4q_type = {}
    all_entropy = 0
    for q_type in question_type_rank_array.keys():
        sum_score = np.array(all_score_4_question_type[q_type]).sum()
        norm_score_4_question_type[q_type] = list(np.array(all_score_4_question_type[q_type])/sum_score)
        entropy4q_type[q_type] = entropy(norm_score_4_question_type[q_type] )
        all_entropy = all_entropy + entropy4q_type[q_type]

    norm_entropy4qtype = {}
    mean_entropy = all_entropy / 65
    for q_type in question_type_rank_array.keys():
        norm_entropy4qtype[q_type] = sigmoid(entropy4q_type[q_type] - mean_entropy)
        unbias_len = (1 - norm_entropy4qtype[q_type]) * beta * len(sorted_question_type_ans_dict[q_type])
        unbias_len = int(unbias_len)
        all_len = len(sorted_question_type_ans_dict[q_type])
        unbias_ans_4_question_type[q_type] = [x for x,_ in sorted_question_type_ans_dict[q_type][:unbias_len]]

    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]
            ex['unbias_ans'] = unbias_ans_4_question_type[q_type]



    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    opt.use_all = 1
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    train(model, train_loader, eval_loader, opt)
