"""
Fits selected constraints to the specific process that is being analyzed.
"""
import pandas as pd
import logging

_logger = logging.getLogger(__name__)


class ConstraintFitter:

    def __init__(self, config, log_info, constraints):
        self.config = config
        self.log_info = log_info
        self.constraints = constraints

    def fit_constraints(self, sim_threshold=0.0):
        const_dfs = [self.fit_constraint(t, sim_threshold) for _, t in self.constraints.reset_index().iterrows()]
        if len(const_dfs) == 0:
            return pd.DataFrame()
        return (
            pd.concat(const_dfs).set_index(self.config.RECORD_ID)
        )

    #  TODO: This is a very ugly method. Refactor it.
    #  TODO: This only considers object-based log-constraint similarity. Add support for other types of similarity.
    def fit_constraint(self, row, sim_threshold):
        fitted_constraints = []
        if len(row[self.config.OBJECT_BASED_SIM_EXTERNAL]) == 1:
            obj_sim = row[self.config.OBJECT_BASED_SIM_EXTERNAL][self.config.OBJECT]
            for obj, sim in obj_sim.items():
                if sim >= sim_threshold:
                    record = {}
                    for index in row.index:
                        record[index] = row[index]
                    record[self.config.OBJECT] = obj
                    record[self.config.CONSTRAINT_STR] = row[self.config.CONSTRAINT_STR].replace(
                        row[self.config.OBJECT], obj)
                    record[self.config.RECORD_ID] = row[self.config.RECORD_ID] + "_" + self.config.OBJECT + "_" + obj
                    record[self.config.FITTED_RECORD_ID] = row[self.config.RECORD_ID]
                    fitted_constraints.append(record)
        elif len(row[self.config.OBJECT_BASED_SIM_EXTERNAL]) == 2:
            obj_sim_l = row[self.config.OBJECT_BASED_SIM_EXTERNAL][self.config.LEFT_OPERAND]
            obj_sim_r = row[self.config.OBJECT_BASED_SIM_EXTERNAL][self.config.RIGHT_OPERAND]
            for obj, sim in obj_sim_l.items():
                if sim >= sim_threshold:
                    record = {}
                    for index in row.index:
                        record[index] = row[index]
                    record[self.config.OBJECT] = obj
                    record[self.config.CONSTRAINT_STR] = row[self.config.CONSTRAINT_STR].replace(
                        row[self.config.LEFT_OPERAND], obj)
                    record[self.config.RECORD_ID] = row[
                                                        self.config.RECORD_ID] + "_" + self.config.LEFT_OPERAND + "_" + obj
                    if row[self.config.RIGHT_OPERAND] in obj_sim_r and obj_sim_r[
                        row[self.config.RIGHT_OPERAND]] >= sim_threshold:
                        record[self.config.CONSTRAINT_STR] = record[self.config.CONSTRAINT_STR].replace(
                            row[self.config.RIGHT_OPERAND], obj)
                        _logger.info("double replacement" + record[self.config.CONSTRAINT_STR])
                    record[self.config.FITTED_RECORD_ID] = row[self.config.RECORD_ID]
                    fitted_constraints.append(record)
            for obj, sim in obj_sim_r.items():
                if sim >= sim_threshold:
                    record = {}
                    for index in row.index:
                        record[index] = row[index]
                    record[self.config.OBJECT] = obj
                    record[self.config.CONSTRAINT_STR] = row[self.config.CONSTRAINT_STR].replace(
                        row[self.config.RIGHT_OPERAND], obj)
                    record[self.config.RECORD_ID] = row[
                                                        self.config.RECORD_ID] + "_" + self.config.RIGHT_OPERAND + "_" + obj
                    record[self.config.FITTED_RECORD_ID] = row[self.config.RECORD_ID]
                    fitted_constraints.append(record)
        return (
            pd.DataFrame.from_records(fitted_constraints)
        )
