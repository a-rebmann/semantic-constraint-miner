from typing import List


class FilterConfig:
    """
    A configuration for a filter
    """
    def __init__(self, config, actions=None, action_categories=None, objects=None,
                 data_objects=None,
                 dict_entries=None, resources=None, names=None, labels=None, arities=None,
                 levels=None):
        self.config = config
        self.actions: List[str] = [] if actions is None else actions
        self.action_categories:  List[str] = [] if action_categories is None else action_categories
        self.objects: List[str] = [] if objects is None else objects
        self.data_objects: List[str] = [] if data_objects is None else data_objects
        self.dict_entries: List[str] = [] if dict_entries is None else dict_entries
        self.resources: List[str] = [] if resources is None else resources
        self.names: List[str] = [] if names is None else names
        self.labels: List[str] = [] if labels is None else labels
        self.arities: List[str] = [] if arities is None else arities
        self.levels: List[str] = [] if levels is None else levels


