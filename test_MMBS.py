import argparse
from collections import defaultdict, Counter
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


from dataset_vqacp_MMBS import Dictionary, VQAFeatureDataset
from model_MMBS import Model
import utils
import opts_MMBS as opts



def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    pred_Shuffling = torch.FloatTensor(N, M).zero_()
    pred_Removal = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i, _, Shuffling_q, Removal_q , positive_q, bias in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        Shuffling_q = Shuffling_q.cuda()
        Removal_q = Removal_q.cuda()
        positive_q = positive_q.cuda()
        logits,  logits_Shuffling, logits_Removal, _, _ = model(q, Shuffling_q, Removal_q, positive_q, v, temperature=0.5, estimator='easy', tau_plus=0.1, beta=1)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        pred_Shuffling[idx:idx+batch_size,:].copy_(logits_Shuffling.data)
        pred_Removal[idx:idx+batch_size,:].copy_(logits_Removal.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, pred_Shuffling, pred_Removal, qIds


def make_json(logits, logits_Shuffling, logits_Removal, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    results_Shuffling = []
    results_Removal = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
        result_Shuffling = {}
        result_Shuffling['question_id'] = qIds[i].item()
        result_Shuffling['answer'] = get_answer(logits_Shuffling[i], dataloader)
        results_Shuffling.append(result_Shuffling)
        result_Removal = {}
        result_Removal['question_id'] = qIds[i].item()
        result_Removal['answer'] = get_answer(logits_Removal[i], dataloader)
        results_Removal.append(result_Removal)
    return results, results_Shuffling, results_Removal

def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    
    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)
    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device

    model = Model(opt)
    model = model.cuda()

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
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

    eval_loader = DataLoader(eval_dset, 128, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    def process(args, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

        model.train(False)

        logits, logits_Shuffling, logits_Removal, qIds = get_logits(model, eval_loader)
        results, results_Shuffling, results_Removal = make_json(logits, logits_Shuffling, logits_Removal, qIds, eval_loader)
        model_label = opt.label 
        
        if opt.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % opt.s_epoch)
        
        utils.create_dir(opt.output)
        assert len(model_label) > 4
        if 0 <= opt.s_epoch:
            model_label += '_epoch%d' % opt.s_epoch
        with open(opt.output+'/test_%s_orig.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)
        with open(opt.output+'/test_%s_Shuffling.json' \
            % (model_label), 'w') as f:
            json.dump(results_Shuffling, f)
        with open(opt.output+'/test_%s_Removal.json' \
            % (model_label), 'w') as f:
            json.dump(results_Removal, f)

    process(opt, model, eval_loader)
