
class Components:

    def __init__(self, config):
        self.config = config
        self.referenced_dict_entries = set()
        self.referenced_data_objects = set()
        self.parsed_tasks = {}
        self.all_actions = set()
        self.all_objects = set()

