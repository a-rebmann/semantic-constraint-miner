import pandas as pd
from semconstmining.parsing.label_parser import BertTagger
from semconstmining.parsing.label_parser.label_utils import sanitize_label


class LogInfo:

    def __init__(self, bert_parser: BertTagger, labels: list = None, names: list = None, resources: list = None):
        self.labels = [] if labels is None else labels
        self.labels = [sanitize_label(label) for label in self.labels]
        self.names = [] if names is None else names
        self.bert_parser = bert_parser
        self.activities_to_parsed = {label: self.bert_parser.parse_label(label) for label in self.labels}
        self.objects = list(
            set([x.main_object for x in self.activities_to_parsed.values() if not pd.isna(x.main_object) and
                 x.main_object not in self.bert_parser.config.TERMS_FOR_MISSING]))
        self.actions = list(
            set([x.main_action for x in self.activities_to_parsed.values() if not pd.isna(x.main_action) and
                 x.main_action not in self.bert_parser.config.TERMS_FOR_MISSING]))
        self.resources = [] if resources is None else resources
