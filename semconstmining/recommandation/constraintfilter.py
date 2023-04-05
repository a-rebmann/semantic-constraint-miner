import pandas as pd


class ConstraintFilter:

    def __init__(self, config, filter_config):
        self.config = config
        self.filter_config = filter_config

    def filter_constraints(self, constraints):
        filtered_constraints = constraints
        if self.filter_config.actions:
            filtered_constraints = self.filter_based_on_actions(filtered_constraints)
        if self.filter_config.action_categories:
            filtered_constraints = self.filter_based_on_action_categories(filtered_constraints)
        if self.filter_config.objects:
            filtered_constraints = self.filter_based_on_objects(filtered_constraints)
        if self.filter_config.data_objects:
            filtered_constraints = self.filter_based_on_data_object(filtered_constraints)
        if self.filter_config.dict_entries:
            filtered_constraints = self.filter_based_on_dict_entries(filtered_constraints)
        if self.filter_config.resources:
            filtered_constraints = self.filter_based_on_resources(filtered_constraints)
        if self.filter_config.names:
            filtered_constraints = self.filter_based_on_names(filtered_constraints)
        if self.filter_config.labels:
            filtered_constraints = self.filter_based_on_labels(filtered_constraints)
        if self.filter_config.arities:
            filtered_constraints = self.filter_based_on_arities(filtered_constraints)
        if self.filter_config.levels:
            filtered_constraints = self.filter_based_on_levels(filtered_constraints)
        return filtered_constraints

    def filter_based_on_data_object(self, constraints):
        filtered_constraints = []
        for _, row in constraints.iterrows():
            if any(item in self.filter_config.data_objects for item in row[self.config.DATA_OBJECT]):
                filtered_constraints.append(row)
        return pd.DataFrame(filtered_constraints)

    def filter_based_on_dict_entries(self, constraints):
        filtered_constraints = []
        for _, row in constraints.iterrows():
            if any(item in self.filter_config.dict_entries for item in row[self.config.DICTIONARY]):
                filtered_constraints.append(row)
        return pd.DataFrame(filtered_constraints)

    def filter_based_on_actions(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.LEFT_OPERAND].isin(self.filter_config.actions) |
                filtered_constraints[self.config.RIGHT_OPERAND].isin(self.filter_config.actions))
        ]

    def filter_based_on_action_categories(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.ACTION_CATEGORY].isin(self.filter_config.action_categories) |
                (filtered_constraints[self.config.ACTION_CATEGORY].isnull()))
        ]

    def filter_based_on_objects(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.DATA_OBJECT].isin(self.filter_config.objects) |
                (filtered_constraints[self.config.DATA_OBJECT].isnull()))
        ]

    def filter_based_on_resources(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.RESOURCE].isin(self.filter_config.resources) |
                (filtered_constraints[self.config.RESOURCE].isnull()))
        ]

    def filter_based_on_names(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.NAME].isin(self.filter_config.names) |
                (filtered_constraints[self.config.NAME].isnull()))
        ]

    def filter_based_on_labels(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.LABEL].isin(self.filter_config.labels) |
                (filtered_constraints[self.config.LABEL].isnull()))
        ]

    def filter_based_on_arities(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.OPERATOR_TYPE].isin(self.filter_config.arities) |
                (filtered_constraints[self.config.OPERATOR_TYPE].isnull()))
        ]

    def filter_based_on_levels(self, filtered_constraints):
        return filtered_constraints[
            (filtered_constraints[self.config.LEVEL].isin(self.filter_config.levels) |
                (filtered_constraints[self.config.LEVEL].isnull()))
        ]