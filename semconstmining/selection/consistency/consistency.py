import json
import logging

import requests

from semconstmining.declare.enums import Template

_logger = logging.getLogger(__name__)


class ConsistencyChecker:

    def __init__(self, config):
        self.config = config

    def check_trivial_consistency(self, constraint_selection):
        constraint_selection["inconsistent"] = False
        mask = (constraint_selection[self.config.CONSTRAINT_STR].str.contains("Responded"))
        relevant_for_check = constraint_selection[~mask]
        # go over all pairwise combinations of left operands
        for idx, constraint_row in relevant_for_check.iterrows():
            if not constraint_row["inconsistent"]:
                to_mark = constraint_selection[(constraint_selection[self.config.LEVEL] ==
                                               constraint_row[self.config.LEVEL]) & (
                                               constraint_selection[self.config.OBJECT] ==
                                               constraint_row[self.config.OBJECT])
                                               & (constraint_selection[self.config.TEMPLATE] ==
                                                constraint_row[self.config.TEMPLATE]) & (
                                                constraint_selection[self.config.LEFT_OPERAND] ==
                                                constraint_row[self.config.RIGHT_OPERAND]) & (
                                                constraint_selection[self.config.RIGHT_OPERAND] ==
                                                constraint_row[self.config.LEFT_OPERAND]
                )]
                for i, row in to_mark.iterrows():
                    if not row["inconsistent"]:
                        if row[self.config.SEMANTIC_BASED_RELEVANCE] > constraint_row[self.config.SEMANTIC_BASED_RELEVANCE]:
                            constraint_selection.loc[i, "inconsistent"] = True
                        else:
                            constraint_selection.loc[idx, "inconsistent"] = True
        return constraint_selection[~constraint_selection["inconsistent"]]

    def check_consistency(self, constraint_selection):
        """
        Check the consistency of the selected constraints
        :param constraint_selection: the selected constraints
        :return: the inconsistent constraint sets
        """
        try:
            mqi_sets = []
            const_str = set(constraint_selection.apply(
                lambda row: self.preprocess_str(row), axis=1).values.tolist())
            #print(const_str)
            constraints = [c.replace(" ", "") for c in const_str if c is not None
                           and c not in self.config.TERMS_FOR_MISSING]
            # TODO which templates are supported?
            # TODO enclose operands in quotes
            constraints_str = " ".join(constraints).replace("|", "").replace("[", "(").replace("]", ")")
            #print(constraints_str)
            x = requests.post(self.config.MQI_SERVER, data=constraints_str)
            if not x.ok:
                print("Error:" + str(x))
                return []
            res = json.loads(x.text)
            if "message" in res and res["message"] == "No JSON body provided":
                print("Error:" + str(x))
                return []
            #print(res)
            for mqi_id, mqi_set in res["qmis"].items():
                mqi_sets.append(mqi_set)
            constraint_ids = []
            for mqi_set in mqi_sets:
                constraint_ids.append([constraint_selection[self.config.RECORD_ID].values.tolist()[i] for i in mqi_set])
            _logger.info("Consistency check returned " + str(len(mqi_sets)) + " inconsistent subsets")
            return constraint_ids
        except Exception as e:
            _logger.warning("Error while checking consistency: " + str(e))
            return []

    def preprocess_str(self, row):
        res = row[self.config.CONSTRAINT_STR]
        if row[self.config.TEMPLATE] in self.config.MQI_CONSTRAINTS:
            res = res.replace(row[self.config.LEFT_OPERAND], "'" + row[self.config.LEFT_OPERAND] + "'")
            res = res.replace(row[self.config.RIGHT_OPERAND], "'" + row[self.config.RIGHT_OPERAND] + "'")
            return res
        else:
            return None

    def make_set_consistent_max_relevance(self, recommended_constraints, inconsistent_subsets):
        pass
