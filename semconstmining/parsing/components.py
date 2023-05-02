from itertools import chain


class Components:

    def __init__(self, config):
        self.config = config
        self.referenced_dict_entries = set()
        self.referenced_data_objects = set()
        self.parsed_tasks = {}
        self.all_actions_per_model = dict()
        self.all_objects_per_model = dict()
        self.action_to_category = {}
        self.all_resources_per_model = dict()
        self.task_per_resource_per_model = dict()

    @property
    def all_actions(self):
        return set(chain.from_iterable(self.all_actions_per_model.values()))

    @property
    def all_objects(self):
        return set(chain.from_iterable(self.all_objects_per_model.values()))

    @property
    def all_resources(self):
        return set(chain.from_iterable(self.all_resources_per_model.values()))

    def add_action(self, model_id, action):
        if model_id not in self.all_actions_per_model:
            self.all_actions_per_model[model_id] = set()
        self.all_actions_per_model[model_id].add(action)

    def add_object(self, model_id, obj):
        if model_id not in self.all_objects_per_model:
            self.all_objects_per_model[model_id] = set()
        self.all_objects_per_model[model_id].add(obj)

    def add_resource(self, model_id, resource):
        if model_id not in self.all_resources_per_model:
            self.all_resources_per_model[model_id] = set()
        self.all_resources_per_model[model_id].add(resource)

    def add_task_to_resource(self, model_id, task, resource):
        if model_id not in self.task_per_resource_per_model:
            self.task_per_resource_per_model[model_id] = dict()
        if resource not in self.task_per_resource_per_model[model_id]:
            self.task_per_resource_per_model[model_id][resource] = set()
        self.task_per_resource_per_model[model_id][resource].add(task)

