import pandas as pd
from semconstmining.parsing.label_parser.nlp_helper import sanitize_label, NlpHelper


class LogInfo:

    def __init__(self, nlp_helper: NlpHelper, labels: list = None, names: list = None,
                 resources_to_tasks: dict = None, log_id=None):
        self.labels = [] if labels is None else labels
        self.original_labels = self.labels
        self.labels = [sanitize_label(label) for label in self.labels]
        self.label_to_original_label = {label: original_label for label, original_label in
                                        zip(self.labels, self.original_labels)}

        self.names = [] if names is None else names
        self.names = [sanitize_label(name) for name in self.names]
        self.nlp_helper = nlp_helper
        self.activities_to_parsed = {label: self.nlp_helper.parse_label(label) for label in self.labels}
        self.objects = list(
            set([x.main_object for x in self.activities_to_parsed.values() if not pd.isna(x.main_object) and
                 x.main_object not in self.nlp_helper.config.TERMS_FOR_MISSING]))
        self.actions = list(
            set([x.main_action for x in self.activities_to_parsed.values() if not pd.isna(x.main_action) and
                 x.main_action not in self.nlp_helper.config.TERMS_FOR_MISSING]))
        self.original_label_to_object = {
            self.label_to_original_label[label]: self.activities_to_parsed[label].main_object
            for label in self.labels if self.activities_to_parsed[label].main_object in self.objects}
        self.original_label_to_action = {
            self.label_to_original_label[label]: self.activities_to_parsed[label].main_action
            for label in self.labels if self.activities_to_parsed[label].main_action in self.actions}
        self.object_to_original_labels = {
            obj: [self.label_to_original_label[label] for label in self.labels if
                  self.activities_to_parsed[label].main_object == obj]
            for obj in self.objects}
        self.action_to_original_labels = {
            act: [self.label_to_original_label[label] for label in self.labels if
                  self.activities_to_parsed[label].main_action == act]
            for act in self.actions}
        self.resources_to_tasks = {} if resources_to_tasks is None else \
            {sanitize_label(res): {sanitize_label(task) for task in tasks}
             for res, tasks in resources_to_tasks.items()}
        self.log_id = log_id
