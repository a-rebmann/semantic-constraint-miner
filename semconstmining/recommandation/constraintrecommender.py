import logging

from pandas import DataFrame
from semconstmining.constraintmining.aggregation.contextualsimcomputer import ContextualSimilarityComputer
from semconstmining.declare.enums import Template
from semconstmining.log.loginfo import LogInfo
from semconstmining.recommandation.recommendation_config import RecommendationConfig

_logger = logging.getLogger(__name__)


class ConstraintRecommender:

    def __init__(self, config, context_sim_computer: ContextualSimilarityComputer, log_info: LogInfo):
        self.config = config
        self.context_sim_computer = context_sim_computer
        self.log_info = log_info
        self.activation_based_on = {
            Template.ABSENCE.templ_str: [],
            Template.EXISTENCE.templ_str: [],
            Template.EXACTLY.templ_str:  [],
            Template.INIT.templ_str:  [],
            Template.END.templ_str:  [],
            Template.CHOICE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.EXCLUSIVE_CHOICE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.RESPONDED_EXISTENCE.templ_str:  [self.config.LEFT_OPERAND],
            Template.RESPONSE.templ_str:  [self.config.LEFT_OPERAND],
            Template.ALTERNATE_RESPONSE.templ_str:  [self.config.LEFT_OPERAND],
            Template.CHAIN_RESPONSE.templ_str:  [self.config.LEFT_OPERAND],
            Template.PRECEDENCE.templ_str: [self.config.RIGHT_OPERAND],
            Template.ALTERNATE_PRECEDENCE.templ_str:  [self.config.RIGHT_OPERAND],
            Template.CHAIN_PRECEDENCE.templ_str:  [self.config.RIGHT_OPERAND],
            Template.SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.ALTERNATE_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.CHAIN_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.CO_EXISTENCE.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],

            Template.NOT_RESPONDED_EXISTENCE: [self.config.LEFT_OPERAND],
            Template.NOT_RESPONSE.templ_str:  [self.config.LEFT_OPERAND],
            Template.NOT_CHAIN_RESPONSE.templ_str:  [self.config.LEFT_OPERAND],
            Template.NOT_PRECEDENCE.templ_str:  [self.config.RIGHT_OPERAND],
            Template.NOT_CHAIN_PRECEDENCE.templ_str:  [self.config.RIGHT_OPERAND],
            Template.NOT_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.NOT_ALTERNATE_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
            Template.NOT_CHAIN_SUCCESSION.templ_str: [self.config.LEFT_OPERAND, self.config.RIGHT_OPERAND],
        }

    def recommend_by_activation(self, constraints, recommender_config: RecommendationConfig):
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

    def recommend_general_by_objects(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        return constraints[constraints[self.config.OBJECT_BASED_SIM] >= sim_thresh]

    def recommend_general_by_names(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        return constraints[constraints[self.config.NAME_BASED_SIM] >= sim_thresh]

    def recommend_general_by_labels(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        return constraints[constraints[self.config.LABEL_BASED_SIM] >= sim_thresh]

    def recommend_by_objects(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        if len(self.log_info.objects) == 0:
            raise RuntimeError("No objects available")
        if self.config.OBJECT_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.objects)
            constraints = self.context_sim_computer.compute_object_based_contextual_similarity_external(
                constraints, self.log_info.objects)
        return constraints[constraints[self.config.MAX_OBJECT_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_names(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        if len(self.log_info.names) == 0:
            raise RuntimeError("No names available")
        if self.config.NAME_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.names)
            constraints = self.context_sim_computer.compute_name_based_contextual_similarity_external(
                constraints, self.log_info.names)
        return constraints[constraints[self.config.NAME_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_labels(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        if len(self.log_info.labels) == 0:
            raise RuntimeError("No labels available")
        if self.config.LABEL_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.labels)
            constraints = self.context_sim_computer.compute_label_based_contextual_similarity_external(
                constraints, self.log_info.labels)
        return constraints[constraints[self.config.LABEL_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_actions(self, constraints, sim_thresh) -> DataFrame:
        if len(constraints) == 0:
            return constraints
        if len(self.log_info.actions) == 0:
            raise RuntimeError("No actions available")
        if self.config.ACTION_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.actions)
            constraints = self.context_sim_computer.compute_action_based_contextual_similarity_external(
                constraints, self.log_info.actions)
        return constraints[constraints[self.config.ACTION_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend(self, constraints, recommender_config: RecommendationConfig):
        if len(constraints) == 0:
            return constraints
        constraints = constraints.copy(deep=True)
        constraints = self.recommend_general_by_objects(constraints, sim_thresh=recommender_config.object_thresh)
        constraints = self.recommend_general_by_names(constraints, sim_thresh=recommender_config.name_thresh)
        constraints = self.recommend_general_by_labels(constraints, sim_thresh=recommender_config.label_thresh)
        if recommender_config.object_external_thresh > 0:
            constraints = self.recommend_by_objects(constraints,
                                                    sim_thresh=recommender_config.object_external_thresh)
        if recommender_config.name_external_thresh > 0:
            constraints = self.recommend_by_names(constraints,
                                                  sim_thresh=recommender_config.name_external_thresh)
        if recommender_config.label_external_thresh > 0:
            constraints = self.recommend_by_labels(constraints,
                                                   sim_thresh=recommender_config.label_external_thresh)
        return constraints




    def non_conflicting_max_relevance(self, constraints, recommender_config: RecommendationConfig):
        return constraints

