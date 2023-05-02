import logging

import pandas as pd

from semconstmining.declare.enums import Template
from semconstmining.log.loginfo import LogInfo
from semconstmining.selection.instantiation.recommendation_config import RecommendationConfig

_logger = logging.getLogger(__name__)


class ConstraintRecommender:

    def __init__(self, config, recommender_config: RecommendationConfig, log_info: LogInfo):
        self.config = config
        self.recommender_config = recommender_config
        self.log_info = log_info
        self.activation_based_on = {
            Template.ABSENCE.templ_str: [],
            Template.EXISTENCE.templ_str: [],
            Template.EXACTLY.templ_str: [],
            Template.INIT.templ_str: [],
            Template.END.templ_str: [],
            Template.CHOICE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.EXCLUSIVE_CHOICE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.RESPONDED_EXISTENCE.templ_str: [self.config.LEFT_OPERAND],
            Template.RESPONSE.templ_str: [self.config.LEFT_OPERAND],
            Template.ALTERNATE_RESPONSE.templ_str: [self.config.LEFT_OPERAND],
            Template.CHAIN_RESPONSE.templ_str: [self.config.LEFT_OPERAND],
            Template.PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.ALTERNATE_PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.CHAIN_PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.ALTERNATE_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.CHAIN_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.CO_EXISTENCE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],

            Template.NOT_RESPONDED_EXISTENCE: [self.config.LEFT_OPERAND],
            Template.NOT_RESPONSE.templ_str: [self.config.LEFT_OPERAND],
            Template.NOT_CHAIN_RESPONSE.templ_str: [self.config.LEFT_OPERAND],
            Template.NOT_PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.NOT_CHAIN_PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.NOT_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.NOT_ALTERNATE_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.NOT_CHAIN_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
        }

    def recommend_by_activation(self, constraints):
        """
        Recommends constraints based on the activation of the constraint.
        :return: A dataframe with the recommended constraints.
        """
        if len(constraints) == 0:
            return constraints
        constraints[self.config.ACTIVATION] = constraints.apply(
            lambda row: self._compute_activation(row), axis=1)
        return constraints[constraints[self.config.ACTIVATION] > 0]

    def _compute_activation(self, row):
        """
        Computes the activation of a constraint.
        :param row: The row of a constraint.
        :return: True if the constraint should be activated, False otherwise.
        """
        if row[self.config.TEMPLATE] not in self.activation_based_on:
            return 0
        if len(self.activation_based_on[row[self.config.TEMPLATE]]) == 0:
            return 1
        res = []
        for column in self.activation_based_on[row[self.config.TEMPLATE]]:
            if row[column] is None or row[column] == "":
                res.append(0)
            elif row[self.config.LEVEL] == self.config.MULTI_OBJECT and row[column] in self.log_info.objects:
                res.append(1)
            elif row[self.config.LEVEL] == self.config.OBJECT and row[column] in self.log_info.actions:
                res.append(1)
            else:
                res.append(0)
        return 1 if (len(res) > 0 and any(res)) or (len(res) == 0) else 0

    def recommend(self, constraints):
        if len(constraints) == 0:
            return constraints
        constraints = constraints.copy(deep=True)
        constraints[self.config.RELEVANCE_SCORE] = 0.0
        relevance_func = self.recommender_config.get_lambda_function(constraints)
        constraints[self.config.RELEVANCE_SCORE] = constraints.apply(relevance_func, axis=1)
        constraints = pd.concat([constraints[constraints[self.config.OPERATOR_TYPE] == self.config.UNARY].nlargest(self.recommender_config.top_k, [self.config.RELEVANCE_SCORE]),
                                constraints[constraints[self.config.OPERATOR_TYPE] == self.config.BINARY].nlargest(self.recommender_config.top_k, [self.config.RELEVANCE_SCORE])])
        return constraints

    def non_conflicting_max_relevance(self, constraints, recommender_config: RecommendationConfig):
        return constraints
