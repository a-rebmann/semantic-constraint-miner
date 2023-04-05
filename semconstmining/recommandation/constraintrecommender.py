import logging

from pandas import DataFrame
from semconstmining.constraintmining.aggregation.contextualsimcomputer import ContextualSimilarityComputer
from semconstmining.log.loginfo import LogInfo
from semconstmining.recommandation.recommendation_config import RecommendationConfig

_logger = logging.getLogger(__name__)


class ConstraintRecommender:

    def __init__(self, config, context_sim_computer: ContextualSimilarityComputer, log_info: LogInfo):
        self.config = config
        self.context_sim_computer = context_sim_computer
        self.log_info = log_info

    def recommend_general_by_objects(self, constraints, sim_thresh) -> DataFrame:
        return constraints[constraints[self.config.OBJECT_BASED_SIM] >= sim_thresh]

    def recommend_general_by_names(self, constraints, sim_thresh) -> DataFrame:
        return constraints[constraints[self.config.NAME_BASED_SIM] >= sim_thresh]

    def recommend_general_by_labels(self, constraints, sim_thresh) -> DataFrame:
        return constraints[constraints[self.config.LABEL_BASED_SIM] >= sim_thresh]

    def recommend_by_objects(self, constraints, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0:
            raise RuntimeError("No objects available")
        if self.config.OBJECT_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.objects)
            constraints = self.context_sim_computer.compute_object_based_contextual_similarity_external(
                constraints, self.log_info.objects)
        return constraints[constraints[self.config.MAX_OBJECT_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_names(self, constraints, sim_thresh) -> DataFrame:
        if len(self.log_info.names) == 0:
            raise RuntimeError("No names available")
        if self.config.NAME_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.names)
            constraints = self.context_sim_computer.compute_name_based_contextual_similarity_external(
                constraints, self.log_info.names)
        return constraints[constraints[self.config.NAME_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_labels(self, constraints, sim_thresh) -> DataFrame:
        if len(self.log_info.labels) == 0:
            raise RuntimeError("No labels available")
        if self.config.LABEL_BASED_SIM_EXTERNAL not in constraints.columns:
            self.context_sim_computer.pre_compute_embeddings(self.log_info.labels)
            constraints = self.context_sim_computer.compute_label_based_contextual_similarity_external(
                constraints, self.log_info.labels)
        return constraints[constraints[self.config.LABEL_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend(self, constraints, recommender_config: RecommendationConfig):
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
