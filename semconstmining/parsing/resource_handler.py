import logging
import os
import warnings
from os.path import exists
import pandas as pd
from tqdm import tqdm
import json
from semconstmining.constraintmining.bert_parser import BertTagger, label_utils
from semconstmining.constraintmining.model.parsed_label import ParsedLabel, get_dummy
from semconstmining.parsing import parser
from semconstmining.parsing import detector

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

_logger = logging.getLogger(__name__)


class ResourceHandler:

    def __init__(self, config, model_collection_id):
        self.config = config
        self.model_collection_id = model_collection_id
        self.data_parser = parser.BpmnModelParser(config)
        self.bert_parser = BertTagger(config)
        self.elements_ser_file = config.DATA_INTERIM / (self.model_collection_id + "_" + config.ELEMENTS_SER_FILE)
        self.models_ser_file = config.DATA_INTERIM / (self.model_collection_id + "_" + config.MODELS_SER_FILE)
        self.languages_ser_file = config.DATA_INTERIM / (self.model_collection_id + "_" + config.LANG_SER_FILE)
        self.logs_ser_file = config.DATA_INTERIM / (self.model_collection_id + "_" + config.LOGS_SER_FILE)
        self.tagged_ser_file = config.DATA_INTERIM / (self.model_collection_id + "_" + config.TAGGED_SER_FILE)
        # TODO this is not generic
        self.dictionary_ser_file = config.DATA_DATASET_DICT if self.model_collection_id == "SAP-SAM" \
            else config.DATA_OPAL_DATASET_DICT if self.model_collection_id == "OPAL" else None
        self.bpmn_model_elements = None
        self.bpmn_models = None
        self.model_languages = None
        self.bpmn_logs = None
        self.bpmn_task_labels = None
        self.dictionary = None

    def get_parsed_task(self, t1):
        t1_parse = self.bpmn_model_elements[
            self.bpmn_model_elements[self.config.CLEANED_LABEL] == t1].reset_index()
        if len(t1_parse) == 0:
            return get_dummy(self.config, t1, self.config.EN)
        label = t1_parse[self.config.CLEANED_LABEL].values[0]
        split = t1_parse[self.config.SPLIT_LABEL].values[0]
        tags = t1_parse["tags"].values[0]
        dicts = []
        if t1_parse['glossary'].values[0] != '{}':
            dicts = [entry.replace("/glossary/", "") for entry in json.loads(t1_parse['glossary'].values[0])['name']]
            dicts = [entry for entry in dicts if entry not in self.config.TERMS_FOR_MISSING]
        if "lang" not in t1_parse.columns:
            lang = self.config.EN
        else:
            lang = t1_parse["lang"].values[0]
        return ParsedLabel(self.config, label, split, tags, self.bert_parser.find_objects(split, tags),
                           self.bert_parser.find_actions(split, tags, lemmatize=True), lang, dicts)

    def get_dictionary_entry(self, entry):
        if self.dictionary is None:
            self.load_dictionary_if_exists()
        if self.dictionary is None:
            _logger.warning("No dictionary found. Please make sure the dictionary data is available.")
        return self.dictionary.loc[entry]['name']

    def load_bpmn_model_elements(self):
        if exists(self.config.DATA_INTERIM / self.elements_ser_file):
            _logger.info("Loading elements from " + str(self.config.DATA_INTERIM / self.elements_ser_file) + ".")
            self.bpmn_model_elements = pd.read_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        else:
            self.bpmn_model_elements = self.data_parser.parse_model_elements()
            self.bpmn_model_elements[self.config.ORIGINAL_LABEL] = self.bpmn_model_elements[self.config.LABEL]
            self.bpmn_model_elements[self.config.LABEL] = self.bpmn_model_elements[self.config.LABEL].apply(
                lambda x: label_utils.sanitize_label(str(x or '')))
            self.bpmn_model_elements[self.config.CLEANED_LABEL] = self.bpmn_model_elements[self.config.LABEL]
            self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)

    def load_bpmn_models(self):
        if exists(self.config.DATA_INTERIM / self.models_ser_file):
            _logger.info("Loading models from " + str(self.config.DATA_INTERIM / self.models_ser_file) + ".")
            self.bpmn_models = pd.read_pickle(self.config.DATA_INTERIM / self.models_ser_file)
        else:
            self.bpmn_models = self.data_parser.parse_models(filter_df=self.bpmn_model_elements)
            self.bpmn_models = self.bpmn_models.loc[self.bpmn_models.index.isin(self.bpmn_model_elements.index.values)]
            self.bpmn_models[self.config.NAME] = self.bpmn_models[self.config.NAME].astype(str)
            _logger.info("Remove example models from " + str(len(self.bpmn_models)))
            pattern = '|'.join(self.config.EXAMPLE_MODEL_NAMES)
            df_no_example = self.bpmn_models[~self.bpmn_models[self.config.NAME].str.contains(pattern)]
            df_example = self.bpmn_models[self.bpmn_models[self.config.NAME].str.contains(pattern)].drop_duplicates(subset=[self.config.NAME])
            self.bpmn_models = pd.concat([df_example, df_no_example])
            _logger.info(str(len(self.bpmn_models)) + " models remaining")
            self.bpmn_models.to_pickle(self.config.DATA_INTERIM / self.models_ser_file)

    def determine_model_languages(self):
        if exists(self.config.DATA_INTERIM / self.languages_ser_file):
            _logger.info("Loading detected languages.")
            self.model_languages = pd.read_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
            if self.config.LABEL in self.model_languages.columns:
                self.model_languages.rename(columns={self.config.LABEL: self.config.LABEL_LIST}, inplace=True)
            if self.config.DETECTED_NAT_LANG not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.model_languages, how='left',
                                                    on=self.config.MODEL_ID)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
        else:
            _logger.info("Detect languages.")
            ld = detector.ModelLanguageDetector(self.config, 0.8)
            self.model_languages = ld.get_detected_natural_language_from_bpmn_model(self.bpmn_model_elements)
            if self.config.LABEL in self.model_languages.columns:
                self.model_languages.rename(columns={self.config.LABEL: self.config.LABEL_LIST}, inplace=True)
            self.model_languages.to_pickle(self.config.DATA_INTERIM / self.languages_ser_file)
            if self.config.DETECTED_NAT_LANG not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.model_languages, how='left',
                                                    on=self.config.MODEL_ID)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)

    def get_logs_for_sound_models(self):
        # Parse and convert the JSON-BPMNs to Petri nets
        if exists(self.config.DATA_INTERIM / self.logs_ser_file):
            self.bpmn_logs = pd.read_pickle(self.config.DATA_INTERIM / self.logs_ser_file)
        else:
            df_petri = self.data_parser.convert_models_to_pn_df(self.bpmn_models)
            self.bpmn_logs = self.data_parser.generate_logs_lambda(df_petri)
            self.bpmn_logs.to_pickle(self.config.DATA_INTERIM / self.logs_ser_file)
            for (dir_path, dir_names, filenames) in os.walk(self.config.PETRI_LOGS_DIR):
                for filename in filenames:
                    os.remove(dir_path + "/" + filename)

    def tag_task_labels(self):
        if exists(self.config.DATA_INTERIM / self.tagged_ser_file):
            _logger.info("Loading tagged labels.")
            self.bpmn_task_labels = pd.read_pickle(self.config.DATA_INTERIM / self.tagged_ser_file)
            if self.config.SPLIT_LABEL not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.bpmn_task_labels, how='left',
                                                    on=self.config.CLEANED_LABEL)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        else:
            _logger.info("Start tagging labels.")
            all_labs = list(
                self.bpmn_model_elements[self.bpmn_model_elements[self.config.ELEMENT_CATEGORY] == "Task"][self.config.LABEL].unique())
            _logger.info(str(len(all_labs)) + " labels cleaned. " + "Start parsing.")
            all_labs_split = [label_utils.split_label(lab) for lab in tqdm(all_labs)]
            tagged = self.bert_parser.parse_labels(all_labs_split)
            self.bpmn_task_labels = pd.DataFrame(
                {self.config.CLEANED_LABEL: all_labs, self.config.SPLIT_LABEL: all_labs_split, "tags": tagged,
                 "lang": ["english" for _ in range(len(all_labs))]})
            self.bpmn_task_labels.to_pickle(self.config.DATA_INTERIM / self.tagged_ser_file)
            if self.config.SPLIT_LABEL not in self.bpmn_model_elements.columns:
                self.bpmn_model_elements = pd.merge(self.bpmn_model_elements, self.bpmn_task_labels, how='left',
                                                    on=self.config.CLEANED_LABEL)
                self.bpmn_model_elements.to_pickle(self.config.DATA_INTERIM / self.elements_ser_file)
        _logger.info("We have " + str(len(self.bpmn_task_labels)) + " labels.")

    def filter_only_english(self):
        _logger.info("Filtering for english labels.")
        self.bpmn_model_elements = self.bpmn_model_elements[self.bpmn_model_elements[self.config.DETECTED_NAT_LANG] == self.config.EN]

    def load_dictionary_if_exists(self):
        if self.dictionary_ser_file is None:
            return
        paths = sorted(self.dictionary_ser_file.glob("*.csv"))
        if len(paths) > 0:
            _logger.info("Loading dictionary from " + str(paths[-1]))
            self.dictionary = parser.parse_dict_csv_raw(paths[-1])
