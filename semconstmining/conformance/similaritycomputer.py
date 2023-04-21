from enum import Enum
from nltk.corpus import wordnet
from itertools import chain
from semconstmining.parsing.label_parser import label_utils


class SimMode(Enum):
    EQUAL = 1,
    SYNONYM = 2,
    SEMANTIC_SIM = 3


class SimilarityComputer:

    def __init__(self, config, sim_mode=SimMode.EQUAL):
        self.config = config
        self.sim_mode = sim_mode
        self.label_util = label_utils.LabelUtil(config)


class SemanticSimilarityComputer(SimilarityComputer):

    def __init__(self, config, sim_threshold=1.0, compute_sim_per_log=True, match_one=False):
        super().__init__(config=config, sim_mode=SimMode.SEMANTIC_SIM)
        self.sim_threshold = sim_threshold
        self.compute_sim_per_log = compute_sim_per_log
        self.match_one = match_one
        self.glove = None
        self.dictionary = None
        self.docsim_index = None
        self.verb_list = []
        self.sim_verbs_map = {}
        self.synonym_map = {}

    def __repr__(self):
        return "SEMANTIC_SIM (" + str(self.sim_threshold) + " matchone_" + str(self.match_one) + ")"

    def initialize_similarities(self, kb_verbs, log_verbs):
        print("computing semantic similarity matrix")

        return kb_verbs, self.dictionary, self.docsim_index

    def set_loaded_similarities(self, kb_verbs, dictionary, docsim_index):
        self.verb_list = kb_verbs
        self.dictionary = dictionary
        self.docsim_index = docsim_index

    def compute_semantic_sim_verbs(self, verb, relevant_verbs):
        if verb in self.sim_verbs_map:
            return self.sim_verbs_map[verb]
        sims = self.docsim_index[self.dictionary.doc2bow(simple_preprocess(verb))]
        sim_verbs = [self.verb_list[index] for (index, sim) in sims if sim >= self.sim_threshold]
        # and self.verb_list[index] in relevant_verbs]
        if [verb] not in sim_verbs:
            sim_verbs.append([verb])
        sim_verbs = [sim_verb[0] for sim_verb in sim_verbs]
        return sim_verbs


class SynonymSimilarityComputer(SimilarityComputer):
    def __init__(self, config, match_one=False):
        super().__init__(config=config, sim_mode=SimMode.SYNONYM)
        self.match_one = match_one
        self.synonym_map = {}

    def __repr__(self):
        return "SYNONYMOUS similarity (matchone_" + str(self.match_one) + ")"

    def get_synonyms(self, verb):
        lemma = self.label_util.lemmatize_word(verb)
        if lemma in self.synonym_map:
            return self.synonym_map[lemma]
        synsets = wordnet.synsets(lemma)
        synonyms = set(chain.from_iterable([word.lemma_names() for word in synsets]))
        synonyms.add(lemma)
        self.synonym_map[lemma] = synonyms
        return synonyms
