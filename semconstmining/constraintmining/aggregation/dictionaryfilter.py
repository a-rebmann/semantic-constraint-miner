from nltk.corpus import words
from nltk.corpus import wordnet
from pandas import DataFrame

class DictionaryFilter:

    def __init__(self, config, constraints: DataFrame):
        self.config = config
        self.constraints = constraints
        self.words = words.words()
        self.nouns = {noun for x in wordnet.all_synsets('n') for noun in x.name().split('.', 1)}

    def _check_proper_object(self, tok: str):
        is_obj = len(tok) > 2 and tok in self.words and (
                    tok in self.nouns or tok in self.config.SCHEMA_ORG_OBJ + self.config.SCHEMA_ORG_OBJ_PROP)
        return is_obj

    def mark_natural_language_objects(self, exceptions: list[str] = None):
        exceptions = [] if exceptions is None else exceptions
        self.constraints[self.config.NAT_LANG_OBJ] = self.constraints[self.config.OBJECT].apply(
            lambda x: x == "" or x in exceptions or self._check_proper_object(x))
        return self.constraints

    def filter_with_proprietary_dict(self, prop_dict: dict, keep: bool = True):
        """
        :param prop_dict: the dictionary that contains the proprietary terms in its key set
        :param keep: wether the specified proprietary objects shall be kept or omitted
        :return: {self}.constraints!
        """
        if len(prop_dict) == 0:
            return self.constraints
        pattern = '|'.join(prop_dict.keys())
        if keep:
            self.constraints = self.constraints[(self.constraints[self.config.OBJECT] == "") |
                                                (self.constraints[self.config.OBJECT].str.contains(pattern))]
        else:
            self.constraints = self.constraints[(self.constraints[self.config.OBJECT] == "") |
                                                (~(self.constraints[self.config.OBJECT].str.contains(pattern)))]
        return self.constraints
