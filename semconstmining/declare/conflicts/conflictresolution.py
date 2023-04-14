class ConflictResolver:

    def __init__(self, constraints):
        self.constraints = constraints

    def resolve(self):
        raise NotImplementedError
