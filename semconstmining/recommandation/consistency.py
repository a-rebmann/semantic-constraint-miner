import json

import requests


class ConsistencyChecker:

    def __init__(self, config):
        self.config = config

    def check_consistency(self, constraint_selection):
        """
        Check the consistency of the selected constraints
        :param constraint_selection: the selected constraints
        :return: the inconsistent constraint sets
        """
        mqi_sets = []
        const_str = set(constraint_selection.apply(
            lambda row: self.preprocess_str(row), axis=1).values.tolist())
        print(const_str)
        constraints = [c.replace(" ", "") for c in const_str if c is not None
                       and c not in self.config.TERMS_FOR_MISSING]
        # TODO which templates are supported?
        # TODO enclose operands in quotes
        constraints_str = " ".join(constraints).replace("|", "").replace("[", "(").replace("]", ")")
        print(constraints_str)
        x = requests.post(self.config.MQI_SERVER, data=constraints_str)
        if not x.ok:
            print("Error:" + str(x))
            return []
        res = json.loads(x.text)
        if "message" in res and res["message"] == "No JSON body provided":
            print("Error:" + str(x))
            return []
        print(res)
        for mqi_id, mqi_set in res["qmis"].items():
            mqi_sets.append(mqi_set)
        constraint_ids = []
        for mqi_set in mqi_sets:
            constraint_ids.append([constraint_selection[self.config.RECORD_ID].values.tolist()[i] for i in mqi_set])
        return constraint_ids

    def preprocess_str(self, row):
        res = row[self.config.CONSTRAINT_STR]
        if row[self.config.TEMPLATE] in self.config.MQI_CONSTRAINTS:
            res = res.replace(row[self.config.LEFT_OPERAND], "'" + row[self.config.LEFT_OPERAND] + "'")
            res = res.replace(row[self.config.RIGHT_OPERAND], "'" + row[self.config.RIGHT_OPERAND] + "'")
            return res
        else:
            return None
