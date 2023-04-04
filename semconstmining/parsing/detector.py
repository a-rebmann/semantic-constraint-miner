import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from tqdm import tqdm
from langcodes import Language as LangCode


class ModelLanguageDetector:
    def __init__(self, config, threshold):
        self.config = config
        self.threshold = threshold
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        Language.factory("language_detector", func=self.get_lang_detector)
        self.nlp.add_pipe('language_detector', last=True)

    def _get_text_language(self, text):
        doc = self.nlp(str(self.clean(text)))
        detection = doc._.language
        return detection['language']

    def add_detected_natural_language_from_meta(self, df_meta):
        df_meta[self.config.DETECTED_NAT_LANG] = [self.get_language_from_code(self._get_text_language(name)) for name in tqdm(df_meta.name)]

    def get_detected_natural_language_from_bpmn_model(self, df):
        df_labels = self.get_df_models_and_labels(df, " ")
        df_labels[self.config.DETECTED_NAT_LANG] = [self.get_language_from_code(self._get_text_language(label)) for label in tqdm(df_labels.label_list)]
        df_labels = df_labels.drop_duplicates(ignore_index=True)
        return df_labels

    def get_df_models_and_labels(self, df, sep_str=" "):
        df_labels = df.drop(columns=[self.config.ELEMENT_CATEGORY, self.config.GLOSSARY, self.config.DATA_OBJECT,
                                     self.config.ELEMENT_ID_BACKUP, self.config.CLEANED_LABEL, self.config.DICTIONARY],
                            axis=1)
        df_labels.label = df_labels.label.apply(lambda x: str(x or ''))
        df_labels = df_labels.drop_duplicates(ignore_index=True)
        df_labels[self.config.LABEL_LIST] = df_labels.groupby([self.config.MODEL_ID])[self.config.LABEL].transform(lambda x: sep_str.join(x))
        df_labels = df_labels.drop(columns=[self.config.LABEL], axis=1)
        df_labels = df_labels.drop_duplicates(ignore_index=True)
        return df_labels

    def clean(self, label):
        # handle some special cases
        label = label.replace("\n", " ").replace("\r", "")
        # delete unnecessary whitespaces
        label = label.strip()
        return label

    def get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def get_language_from_code(self, code):
        return LangCode.make(language=code).display_name().lower()



