import logging

from pandas import DataFrame
from semconstmining.constraintmining.aggregation.contextualsimcomputer import ContextualSimilarityComputer
from semconstmining.log.loginfo import LogInfo

_logger = logging.getLogger(__name__)


def recommend_const_for_log(log_info, context_sim_computer: ContextualSimilarityComputer,
                            sim_thresh=.25):
    """
    Recommend constraints for a log based on the given similarity computer.
    Computes all possible combinations of similarity measures.
    :param context_sim_computer:
    :param sim_thresh:
    :return:

    Parameters
    ----------
    log_info
    """
    # TODO how to combine, weigh, rank?
    const_recommender = ConstraintRecommender(context_sim_computer, log_info)

    const_recommender.recommend_general_by_objects(sim_thresh)
    _logger.info("Done with recommending based on object only")
    const_recommender.recommend_general_by_labels(sim_thresh)
    _logger.info("Done with recommending based on labels only")
    const_recommender.recommend_general_by_names(sim_thresh)
    _logger.info("Done with recommending based on names only")
    const_recommender.recommend_general_or(sim_thresh)
    _logger.info("Done with recommending based on all aspects; either aspects above thresh suffices")
    const_recommender.recommend_general_and(sim_thresh)
    _logger.info("Done with recommending based on all aspects; all aspects must be above thresh")
    const_recommender.recommend_by_objects(sim_thresh)
    _logger.info("Done with recommending based on object only")
    const_recommender.recommend_by_labels(sim_thresh)
    _logger.info("Done with recommending based on labels only")
    const_recommender.recommend_by_names(sim_thresh)
    _logger.info("Done with recommending based on names only")
    const_recommender.recommend_or(sim_thresh)
    _logger.info("Done with recommending based on all aspects; either aspects above thresh suffices")
    const_recommender.recommend_and(sim_thresh)
    _logger.info("Done with recommending based on all aspects; all aspects must be above thresh")
    return const_recommender


class ConstraintRecommender:

    def __init__(self, config, context_sim_computer: ContextualSimilarityComputer, log_info: LogInfo):
        self.config=config
        self.context_sim_computer = context_sim_computer
        self.log_info = log_info
        # initialize embeddings for log
        self.context_sim_computer.pre_compute_embeddings(
            sentences=self.log_info.names + self.log_info.labels + self.log_info.objects)
        self.constraints_context_based = None

    def recommend_general_by_objects(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0:
            raise RuntimeError("No objects available")
        if self.config.OBJECT_BASED_SIM not in self.context_sim_computer.constraints.columns:
            self.context_sim_computer.compute_object_based_contextual_dissimilarity()
        return self.context_sim_computer.constraints[
            self.context_sim_computer.constraints[self.config.OBJECT_BASED_SIM] <= sim_thresh]

    def recommend_general_by_names(self, sim_thresh) -> DataFrame:
        if len(self.log_info.names) == 0:
            raise RuntimeError("No names available")
        if self.config.NAME_BASED_SIM not in self.context_sim_computer.constraints.columns:
            self.context_sim_computer.compute_name_based_contextual_dissimilarity()
        return self.context_sim_computer.constraints[
            self.context_sim_computer.constraints[self.config.NAME_BASED_SIM] <= sim_thresh]

    def recommend_general_by_labels(self, sim_thresh) -> DataFrame:
        if len(self.log_info.labels) == 0:
            raise RuntimeError("No labels available")
        if self.config.LABEL_BASED_SIM not in self.context_sim_computer.constraints.columns:
            self.context_sim_computer.compute_label_based_contextual_dissimilarity()
        return self.context_sim_computer.constraints[
            self.context_sim_computer.constraints[self.config.LABEL_BASED_SIM] <= sim_thresh]

    def recommend_general_and(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0 and len(self.log_info.names) == 0 and len(self.log_info.labels) == 0:
            raise RuntimeError("No info available")
        pattern = True
        if len(self.log_info.objects) != 0:
            pattern = pattern & (self.context_sim_computer.constraints[self.config.OBJECT_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_objects(sim_thresh)
        if len(self.log_info.names) != 0:
            pattern = pattern & (self.context_sim_computer.constraints[self.config.NAME_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_names(sim_thresh)
        if len(self.log_info.labels) != 0:
            pattern = pattern & (self.context_sim_computer.constraints[self.config.LABEL_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_labels(sim_thresh)
        return self.context_sim_computer.constraints[pattern]

    def recommend_general_or(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0 and len(self.log_info.names) == 0 and len(self.log_info.labels) == 0:
            raise RuntimeError("No info available")
        pattern = False
        if len(self.log_info.objects) != 0:
            pattern = pattern & (self.context_sim_computer.constraints[self.config.OBJECT_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_objects(sim_thresh)
        if len(self.log_info.names) != 0:
            pattern = pattern | (self.context_sim_computer.constraints[self.config.NAME_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_names(sim_thresh)
        if len(self.log_info.labels) != 0:
            pattern = pattern | (self.context_sim_computer.constraints[self.config.LABEL_BASED_SIM] <= sim_thresh)
            self.recommend_general_by_labels(sim_thresh)
        return self.context_sim_computer.constraints[pattern]

    def recommend_by_objects(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0:
            raise RuntimeError("No objects available")
        if self.constraints_context_based is None:
            self.constraints_context_based = self.context_sim_computer.constraints.copy(deep=True)
        if self.config.OBJECT_BASED_SIM_EXTERNAL not in self.constraints_context_based.columns:
            self.constraints_context_based = \
                self.context_sim_computer.compute_object_based_contextual_similarity_external(
                    self.constraints_context_based, self.log_info.objects)
        return self.constraints_context_based[self.constraints_context_based[self.config.MAX_OBJECT_BASED_SIM_EXTERNAL]
                                              >= sim_thresh]

    def recommend_by_names(self, sim_thresh) -> DataFrame:
        if len(self.log_info.names) == 0:
            raise RuntimeError("No names available")
        if self.constraints_context_based is None:
            self.constraints_context_based = self.context_sim_computer.constraints.copy(deep=True)
        if self.config.NAME_BASED_SIM_EXTERNAL not in self.constraints_context_based.columns:
            self.constraints_context_based = \
                self.context_sim_computer.compute_name_based_contextual_similarity_external(
                    self.constraints_context_based, self.log_info.names)
        return self.constraints_context_based[self.constraints_context_based[self.config.NAME_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_by_labels(self, sim_thresh) -> DataFrame:
        if len(self.log_info.labels) == 0:
            raise RuntimeError("No labels available")
        if self.constraints_context_based is None:
            self.constraints_context_based = self.context_sim_computer.constraints.copy(deep=True)
        if self.config.LABEL_BASED_SIM_EXTERNAL not in self.constraints_context_based.columns:
            self.constraints_context_based = \
                self.context_sim_computer.compute_label_based_contextual_similarity_external(
                    self.constraints_context_based, self.log_info.labels)
        return self.constraints_context_based[self.constraints_context_based[self.config.LABEL_BASED_SIM_EXTERNAL] >= sim_thresh]

    def recommend_and(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0 and len(self.log_info.names) == 0 and len(self.log_info.labels) == 0:
            raise RuntimeError("No info available")
        if not (self.config.OBJECT_BASED_SIM_EXTERNAL in self.constraints_context_based.columns or
                self.config.NAME_BASED_SIM_EXTERNAL in self.constraints_context_based.columns or
                self.config.LABEL_BASED_SIM_EXTERNAL in self.constraints_context_based.columns):
            self.recommend_by_objects(sim_thresh)
            self.recommend_by_names(sim_thresh)
            self.recommend_by_labels(sim_thresh)
        return self.constraints_context_based[
            (self.constraints_context_based[self.config.OBJECT_BASED_SIM_EXTERNAL] >= sim_thresh) &
            (self.constraints_context_based[self.config.NAME_BASED_SIM_EXTERNAL] >= sim_thresh) &
            (self.constraints_context_based[self.config.LABEL_BASED_SIM_EXTERNAL] >= sim_thresh)
            ]

    def recommend_or(self, sim_thresh) -> DataFrame:
        if len(self.log_info.objects) == 0 and len(self.log_info.names) == 0 and len(self.log_info.labels) == 0:
            raise RuntimeError("No info available")
        if not (self.config.OBJECT_BASED_SIM_EXTERNAL in self.constraints_context_based.columns or
                self.config.NAME_BASED_SIM_EXTERNAL in self.constraints_context_based.columns or
                self.config.LABEL_BASED_SIM_EXTERNAL in self.constraints_context_based.columns):
            self.recommend_by_objects(sim_thresh)
            self.recommend_by_names(sim_thresh)
            self.recommend_by_labels(sim_thresh)
        return self.constraints_context_based[
            (self.constraints_context_based[self.config.OBJECT_BASED_SIM_EXTERNAL] >= sim_thresh) |
            (self.constraints_context_based[self.config.NAME_BASED_SIM_EXTERNAL] >= sim_thresh) |
            (self.constraints_context_based[self.config.LABEL_BASED_SIM_EXTERNAL] >= sim_thresh)
            ]