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

    def fit_constraints(self, sim_threshold=0.5):
        const_dfs = [self.fit_constraint(t, sim_threshold) for _, t in self.constraints.reset_index().iterrows()]
        const_dfs = [t for t in const_dfs if not t.empty]
        if len(const_dfs) == 0:
            return pd.DataFrame()
        temp = (
            pd.concat(const_dfs).set_index(self.config.RECORD_ID)
        )
        temp = temp.drop_duplicates(subset=[self.config.CONSTRAINT_STR, self.config.LEVEL,
                                            self.config.MODEL_NAME, self.config.OBJECT])
        return temp

    def fit_constraint(self, row, sim_threshold):
        fitted_constraints = []
        if row[self.config.LEVEL] == self.config.OBJECT:
            fitted_constraints.extend(self.fit_object_constraint(row, sim_threshold))
        elif row[self.config.LEVEL] == self.config.MULTI_OBJECT:
            fitted_constraints.extend(self.fit_multi_object_or_activity_constraint(row, sim_threshold))
        elif row[self.config.LEVEL] == self.config.ACTIVITY:
            fitted_constraints.extend(self.fit_multi_object_or_activity_constraint(row, sim_threshold))
        elif row[self.config.LEVEL] == self.config.RESOURCE:
            fitted_constraints.extend(self.fit_resource_constraint(row, sim_threshold))
        return (
            pd.DataFrame.from_records(fitted_constraints)
        )

    def instantiate_obj_const(self, row, obj, old_act, new_act=None):
        record = {}
        for index in row.index:
            record[index] = row[index]
        record[self.config.OBJECT] = obj
        record[self.config.CONSTRAINT_STR] = row[self.config.CONSTRAINT_STR].replace(
            row[self.config.OBJECT], obj)
        record[self.config.RECORD_ID] = row[self.config.RECORD_ID] + "_" + self.config.OBJECT + "_" + obj
        record[self.config.FITTED_RECORD_ID] = record[self.config.RECORD_ID]
        record[self.config.LEFT_OPERAND] = row[self.config.LEFT_OPERAND]
        record[self.config.RIGHT_OPERAND] = row[self.config.RIGHT_OPERAND]
        if old_act is not None and new_act is not None:
            if old_act == record[self.config.LEFT_OPERAND]:
                record[self.config.LEFT_OPERAND] = new_act
            if old_act == record[self.config.RIGHT_OPERAND]:
                record[self.config.RIGHT_OPERAND] = new_act
        return record

    def fit_object_constraint(self, row, sim_threshold):
        sim_dict = row[self.config.INDIVIDUAL_RELEVANCE_SCORES]
        fitted_constraints = []
        if self.config.OBJECT in sim_dict:
            obj_sim = sim_dict[self.config.OBJECT]
            for obj, sim in obj_sim.items():
                if sim >= sim_threshold:
                    act = None
                    a = None
                    for a in sim_dict[self.config.ACTION].keys():
                        act = sim_dict[self.config.ACTION][a]
                    record = self.instantiate_obj_const(row, obj, old_act=a, new_act=act)
                    fitted_constraints.append(record)
        return fitted_constraints

    def instantiate_multi_obj_or_act_constraint(self, row, l_obj=None, r_obj=None):
        record = {}
        for index in row.index:
            record[index] = row[index]
        if l_obj is not None:
            record[self.config.LEFT_OPERAND] = l_obj
            record[self.config.CONSTRAINT_STR] = row[self.config.CONSTRAINT_STR].replace(
                row[self.config.LEFT_OPERAND], l_obj)
            record[self.config.RECORD_ID] = row[self.config.RECORD_ID] + "_" + self.config.OBJECT + "_" + l_obj
        if r_obj is not None:
            record[self.config.RIGHT_OPERAND] = r_obj
            record[self.config.CONSTRAINT_STR] = record[self.config.CONSTRAINT_STR].replace(
                row[self.config.RIGHT_OPERAND], r_obj)
            record[self.config.RECORD_ID] = record[self.config.RECORD_ID] + "_" + self.config.OBJECT + "_" + r_obj
        if record[self.config.LEFT_OPERAND] == record[self.config.RIGHT_OPERAND]:
            return None
        record[self.config.FITTED_RECORD_ID] = record[self.config.RECORD_ID]
        return record

    def fit_multi_object_or_activity_constraint(self, row, sim_threshold):
        sim_dict = row[self.config.INDIVIDUAL_RELEVANCE_SCORES]
        fitted_constraints = []
        if self.config.LEFT_OPERAND in sim_dict:
            obj_sim = sim_dict[self.config.LEFT_OPERAND]
            for obj, sim in obj_sim.items():
                if sim >= sim_threshold:
                    record = self.instantiate_multi_obj_or_act_constraint(row, obj, None)
                    if record is not None:
                        fitted_constraints.append(record)
        if self.config.RIGHT_OPERAND in sim_dict:
            obj_sim_r = sim_dict[self.config.RIGHT_OPERAND]
            for obj, sim in obj_sim_r.items():
                if sim >= sim_threshold:
                    record = self.instantiate_multi_obj_or_act_constraint(row, None, obj)
                    if record is not None:
                        fitted_constraints.append(record)
        if self.config.LEFT_OPERAND in sim_dict and self.config.RIGHT_OPERAND in sim_dict:
            obj_sim_l = sim_dict[self.config.LEFT_OPERAND]
            obj_sim_r = sim_dict[self.config.RIGHT_OPERAND]
            for obj_l, sim_l in obj_sim_l.items():
                if sim_l >= sim_threshold:
                    for obj_r, sim_r in obj_sim_r.items():
                        if sim_r >= sim_threshold and obj_l != obj_r:
                            record = self.instantiate_multi_obj_or_act_constraint(row, obj_l, obj_r)
                            if record is not None:
                                fitted_constraints.append(record)
        return fitted_constraints

    def instantiate_resource_constraint(self, row, act, res):
        record = self.instantiate_multi_obj_or_act_constraint(row, act, None)
        record[self.config.CONSTRAINT_STR] = record[self.config.CONSTRAINT_STR].replace(
            record[self.config.OBJECT], res)
        record[self.config.RECORD_ID] = row[self.config.RECORD_ID] + "_" + self.config.RESOURCE + act + "_" + res
        record[self.config.FITTED_RECORD_ID] = record[self.config.RECORD_ID]
        return record

    def fit_resource_constraint(self, row, sim_threshold):
        sim_dict = row[self.config.INDIVIDUAL_RELEVANCE_SCORES]
        fitted_constraints = []
        if self.config.LEFT_OPERAND in sim_dict:
            act_sim = sim_dict[self.config.LEFT_OPERAND]
            for act, sim in act_sim.items():
                if sim >= sim_threshold and self.config.RESOURCE in sim_dict:
                    for res, sim_res in sim_dict[self.config.RESOURCE].items():
                        if sim_res >= sim_threshold:
                            record = self.instantiate_resource_constraint(row, act, res)
                            fitted_constraints.append(record)
        return fitted_constraints
