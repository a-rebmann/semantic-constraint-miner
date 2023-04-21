class RecommendationConfig:
    """
    Configuration for the recommendation module.
    """
    def __init__(self, config, object_thresh=0.5, name_thresh=0.5, label_thresh=0.5, object_external_thresh=0.5,
                 name_external_thresh=0.0, label_external_thresh=0.0):
        self.config = config
        self.object_thresh = object_thresh
        self.name_thresh = name_thresh
        self.label_thresh = label_thresh
        self.object_external_thresh = object_external_thresh
        self.name_external_thresh = name_external_thresh
        self.label_external_thresh = label_external_thresh
