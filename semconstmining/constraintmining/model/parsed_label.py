import logging

from nltk.corpus import stopwords

_logger = logging.getLogger(__name__)


def get_dummy(conf, label, lang):
    return ParsedLabel(conf, label, [label], ['X'], ['none'], ['none'], lang)


class ParsedLabel:

    def __init__(self, config, label, split, tags, bos, actions, lang, dictionary_entries=None):
        self.config = config
        self._stopwords = stopwords.words(lang)
        self.lang = lang
        self.label = label
        self.split_label = split
        self.tags = tags
        self.bos = [bo for bo in bos if bo not in config.TERMS_FOR_MISSING]
        self.bos_plain = " ".join(bo.strip() for bo in self.bos if bo not in self._stopwords).strip()
        self.actions = actions
        self._act_bo_label = None
        self.main_action = self.actions[0].strip() if len(self.actions) > 0 else ""
        self.main_object = self._get_main_object()
        self.dictionary_entries = [] if dictionary_entries is None else dictionary_entries

    def get_act_bo_label(self):
        if self._act_bo_label is None:
            self._act_bo_label = self.main_action + " " + self.main_object
        return self._act_bo_label

    def _get_main_object(self) -> str:
        main_object = self.bos[0].strip() if len(self.bos) > 0 else ""
        if len(main_object.split(" ")) > 1:
            if main_object.split(" ")[-1] in self._stopwords:
                main_object = " ".join(main_object.split(" ")[:-1])
            if main_object.split(" ")[0] in self._stopwords:
                main_object = " ".join(main_object.split(" ")[1:])
        if len(main_object.split(" ")) == 1 and (
                main_object.endswith("able") or main_object in self._stopwords):
            main_object = ""
        return main_object
