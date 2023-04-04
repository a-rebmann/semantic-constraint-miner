import pandas as pd


class ConstraintFilter:

    def __init__(self, config, log_info, constraints):
        self.config = config
        self.log_info = log_info
        self.constraints = constraints

    def filter_based_on_data_object(self, data_objects):
        filtered_constraints = []
        for _, row in self.constraints.iterrows():
            if any(item in data_objects for item in row[self.config.DATA_OBJECT]):
                filtered_constraints.append(row)
        return pd.DataFrame(filtered_constraints)

    def filter_based_on_dict_entries(self, dictionary_items):
        filtered_constraints = []
        for _, row in self.constraints.iterrows():
            if any(item in dictionary_items for item in row[self.config.DICTIONARY]):
                filtered_constraints.append(row)
        return pd.DataFrame(filtered_constraints)
