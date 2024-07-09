import pandas as pd
import random
from transformers import pipeline

def merge_hashtags(words_list):
    merged_list = []
    for entry in words_list:
        word = entry['word']
        if word.startswith('##'):
            merged_list[-1]['word'] += word[2:]
        else:
            merged_list.append(entry)
    return merged_list


def parse_result(result):
    res = [], []
    for entry in result:
        res[0].append(entry['word'])
        res[1].append(entry['entity'])
    return res


class BertTagger:

    def __init__(self, path):
        self.path = path
        self.model = None
        self._load_trained_model(path)

    def check_tok_for_object_type(self, split, pred):
        new_pred = []
        i = 1
        for tok, lab in zip(split, pred):
            if len(split) > i and tok == 'BO' and split[i] != 'BO':
                new_pred.append('X')
            else:
                new_pred.append(lab)
            i += 1
        return new_pred

    def _load_trained_model(self, path):
        self.model = pipeline("ner", model=path)

    def get_tags_for_list(self, li: list) -> dict:
        tagged = {}
        for unique in li:
            unique = str(unique)
            if unique not in tagged:
                tagged[unique] = self.predict_single_label(unique)[1]
        return tagged

    def predict_single_label(self, label):
        merged = merge_hashtags(self.model(label))
        split, pred = parse_result(merged)
        pred = self.check_tok_for_object_type(split, pred)
        return split, pred

    def predict_batch_at_once(self, labels):
        res = self.model(labels)
        merged = [merge_hashtags(x) for x in res]
        return [parse_result(x) for x in merged]

    @staticmethod
    def _fill_all(x, seen_tagged):
        uniquely_tagged = []
        tagging = str()
        if x not in seen_tagged.keys():
            return
        for i in range(len(seen_tagged[x][0])):
            tagging = tagging + str(seen_tagged[x][0][i]) + '<>' + str(seen_tagged[x][1][i]) + ', '
        uniquely_tagged.append(tagging)
        return uniquely_tagged

