

class ResultAggregator:

    def __init__(self, config, violated_constraints):
        self.config = config
        self.violated_constraints = violated_constraints

    def simplify_output(self, simplify=True):
        """
        Simplify violated constraints by aggregating them
        TODO: assumes pairs og labels, change for all constraints
        """
        simplified_output = {"PRE": {}, "SUCC": {}}
        delete_from_pre = set()
        if not simplify:
            for pair in self.violated_constraints:
                if pair[0] not in simplified_output["PRE"]:
                    simplified_output["PRE"][pair[0]] = {pair[1]}
        else:
            for pair in  self.violated_constraints:
                if pair[0] not in simplified_output["PRE"]:
                    simplified_output["PRE"][pair[0]] = {pair[1]}
                else:
                    simplified_output["PRE"][pair[0]].add(pair[1])
            for pair in  self.violated_constraints:
                if pair[1] not in simplified_output["SUCC"]:
                    simplified_output["SUCC"][pair[1]] = {pair[0]}
                else:
                    simplified_output["SUCC"][pair[1]].add(pair[0])
            for label in simplified_output["PRE"]:
                if label in simplified_output["SUCC"] and len(simplified_output["SUCC"][label]) < len(
                        simplified_output["PRE"][label]):
                    del simplified_output["SUCC"][label]
                elif label in simplified_output["SUCC"] and len(simplified_output["PRE"][label]) <= len(
                        simplified_output["SUCC"][label]):
                    delete_from_pre.add(label)
            for label in delete_from_pre:
                del simplified_output["PRE"][label]
        return simplified_output
