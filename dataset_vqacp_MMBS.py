from __future__ import print_function
import os
from torch.nn import functional as F
import json
import _pickle as cPickle
import numpy as np
import pickle
import utils
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import zarr
import random
COUNTING_ONLY = False


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
                    'amount of' in q.lower() or \
                    'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'question_type': answer['question_type'],
        'answer_type': answer['answer_type'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'test'
    """
    question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path)),
                           key=lambda x: x['question_id'])

    # train, val
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]

    utils.assert_eq(len(questions), len(answers))

    if ratio < 1.0:
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0, len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        answers_new = [answers[i] for i in index]
    else:
        questions_new = questions
        answers_new = answers

    entries = []
    for question, answer in zip(questions_new, answers_new):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(img_id, question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot, image_dataroot, ratio, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        print('loading image features and bounding boxes')
        # Load image features and bounding boxes
        self.features = zarr.open(os.path.join(image_dataroot, 'trainval.zarr'), mode='r')
        self.s_dim = self.spatials[list(self.spatials.keys())[0]].shape[1]
        print('loading image features and bounding boxes done!')

        self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
        self.tokenize()
        self.tensorize(name)

    def _unbias_or_not(self, labels, unbias_ans):
        prediction_ans_k, top_1ans_ind = torch.topk(F.softmax(labels, dim=0), k=1, dim=0, sorted=False)
        single_label = top_1ans_ind
        if single_label.item() in unbias_ans:
            return 1
        return 0


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            Removal_question_text = entry['question'].lower().replace(entry['question_type'], "")
            Removal_tokens = self.dictionary.tokenize(Removal_question_text, False)

            length = len(tokens)
            Removal_length = len(Removal_tokens)
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            else:
                tokens = tokens[:max_length]
                length = max_length
            if len(Removal_tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(Removal_tokens))
                Removal_tokens = Removal_tokens + padding
            else:
                Removal_tokens = Removal_tokens[:max_length]
                Removal_length = max_length
                
            utils.assert_eq(len(tokens), max_length)
            utils.assert_eq(len(Removal_tokens), max_length)
            entry['q_token'] = tokens
            entry['length'] = length
            entry['Removal_token'] = Removal_tokens
            entry['Removal_length'] = Removal_length

    def tensorize(self, name):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            Removal_question = torch.from_numpy(np.array(entry['Removal_token']))
            entry['Removal_token'] = Removal_question
            length = torch.from_numpy(np.array(entry['length']))
            entry['length'] = length
            Removal_length = torch.from_numpy(np.array(entry['Removal_length']))
            entry['Removal_length'] = Removal_length

              
            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features[entry['image']]))
            spatials = torch.from_numpy(np.array(self.spatials[entry['image']]))

        question = entry['q_token']
        length = entry['length']
        Removal_question = entry['Removal_token']
        Shuffling_q_id = list(range(0,length)) 
        random.shuffle(Shuffling_q_id)
        
        Shuffling_id = question[:length][Shuffling_q_id]
        mask_id = question[length :]
        Shuffling_question = torch.cat([Shuffling_id, mask_id], 0)
        question_id = entry['question_id']
        image_id = entry['image_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            if self._unbias_or_not(target, entry['unbias_ans']):  
                positive_question = question 
            else:
                if entry['answer_type'] in ['number', 'other']:
                    positive_question = Shuffling_question
                else:
                    positive_question = Removal_question 

            return features, spatials, question, target, question_id, image_id, Shuffling_question, Removal_question, positive_question, entry['bias']
        else:
            return features, spatials, question, question_id, image_id, Shuffling_question, Removal_question, positive_question, entry['bias']

    def __len__(self):
        return len(self.entries)


