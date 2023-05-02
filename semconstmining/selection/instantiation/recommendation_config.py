class RecommendationConfig:
    """
    Configuration for the recommendation module.
    """

    def __init__(self, config, frequency_weight=0.5, semantic_weight=0.5, object_thresh=0.0, name_thresh=0.5,
                 label_thresh=0.5, object_external_thresh=0.5,
                 name_external_thresh=0.0, label_external_thresh=0.0, top_k=100):
        self.config = config
        self.frequency_weight = frequency_weight
        self.semantic_weight = semantic_weight
        self.top_k = top_k
        self.object_thresh = object_thresh
        self.name_thresh = name_thresh
        self.label_thresh = label_thresh
        self.object_external_thresh = object_external_thresh
        self.name_external_thresh = name_external_thresh
        self.label_external_thresh = label_external_thresh
        self.generality_pattern = []
        self.relevance_pattern = []
        if object_thresh > 0:
            self.generality_pattern.append(self.config.OBJECT_BASED_SIM)
        if name_thresh > 0:
            self.generality_pattern.append(self.config.NAME_BASED_GENERALITY)
        if label_thresh > 0:
            self.generality_pattern.append(self.config.LABEL_BASED_GENERALITY)
        if object_external_thresh > 0:
            self.relevance_pattern.append(self.config.SEMANTIC_BASED_RELEVANCE)

    def get_lambda_function(self, constraints):
        return lambda x: self.frequency_weight * (x[self.config.SUPPORT] / constraints[self.config.SUPPORT].max()) + \
                         self.semantic_weight * (x[[patt for patt in self.generality_pattern]].max() +
                                                 x[[patt for patt in self.relevance_pattern]].max()) / 2 \
            if x[self.config.SUPPORT] > 0 else 0
