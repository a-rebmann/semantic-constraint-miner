import os

import pandas as pd
from pm4py.read import read_xes


class LogHandler:

    def __init__(self, config):
        self.config = config
        self.log = None

    def handle_non_standard_att_names(self, pd_log, att_names):
        if len(att_names) == 3:
            pd_log.rename(columns={att_names[self.config.XES_CASE]: self.config.XES_CASE,
                                   att_names[self.config.XES_TIME]: self.config.XES_TIME,
                                   att_names[self.config.XES_NAME]: self.config.XES_NAME},
                          inplace=True)

    def read_xes_log(self, path, name):
        pd_log = read_xes(os.path.join(path, name))
        return pd_log

    def read_csv_log(self, path, name):
        pd_log = pd.read_csv(os.path.join(path, name), sep=";", engine="python")
        """check if given column names exist in log and rename them"""
        must_have = [self.config.XES_CASE, self.config.XES_NAME, self.config.XES_TIME]
        not_exist_str = " does not exist as column name"
        for e in must_have:
            if e not in pd_log.columns:
                raise ValueError(e + not_exist_str)
        # THIS IS OLD:
        # pd_log = pm4py.format_dataframe(pd_log, case_id=XES_CASE, activity_key=XES_NAME, timestamp_key=XES_TIME)
        return pd_log

    def read_log(self, path, name):
        log = None
        if ".csv" in name:
            log = self.read_csv_log(path, name)
        elif ".xes" in name:
            log = self.read_xes_log(path, name)
        self.log = log
        return self.log

    def get_resources_to_tasks(self):
        if self.log is None:
            raise RuntimeError("You must load a log before.")
        res_to_tasks = {}
        if self.config.XES_ROLE in self.log.columns:
            for role, group in self.log.groupby(self.config.XES_ROLE):
                role = role.replace(" and ", " & ")
                res_to_tasks[role] = set(group[self.config.XES_NAME].unique())
            if "unknown" not in res_to_tasks:
                res_to_tasks["unknown"] = set(self.log[pd.isna(self.log[self.config.XES_ROLE])]
                                                   [self.config.XES_NAME].unique())
            else:
                res_to_tasks["unknown"].update(set(self.log[pd.isna(self.log[self.config.XES_ROLE])]
                                                   [self.config.XES_NAME].unique()))
        return res_to_tasks
