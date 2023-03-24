from pathlib import Path

from semconstmining.declare.enums import Template


class Config:

    def __init__(self, project_root: Path):
        self.PROJECT_ROOT: Path = project_root
        self.DATA_ROOT = self.PROJECT_ROOT / "data"
        self.DATA_RAW = self.DATA_ROOT / "raw"
        self.DATA_LOGS = self.DATA_ROOT / "logs"
        self.DATA_RESOURCES = self.DATA_ROOT / "resources"
        self.DATA_DATASET = self.DATA_RAW / "sap_sam_2022" / "models"
        self.DATA_OPAL_DATASET = self.DATA_RAW / "opal" / "models"
        self.DATA_DATASET_DICT = self.DATA_RAW / "sap_sam_2022" / "dict"
        self.DATA_OPAL_DATASET_DICT = self.DATA_RAW / "opal" / "dict"
        self.DATA_INTERIM = self.DATA_ROOT / "interim"
        self.SRC_ROOT = self.PROJECT_ROOT / "src" / "semconstmining"
        self.FIGURES_ROOT = self.PROJECT_ROOT / "reports" / "figures"
        self.PETRI_LOGS_DIR = self.DATA_INTERIM / "bpmn_logs"

        self.MODEL_PATH = self.DATA_ROOT / "bert"

        self.ELEMENTS_SER_FILE = "bpmn_elements.pkl"
        self.LANG_SER_FILE = "bpmn_languages.pkl"
        self.MODELS_SER_FILE = "bpmn_models.pkl"
        self.LOGS_SER_FILE = "bpmn_logs.pkl"
        self.TAGGED_SER_FILE = "bpmn_tagged.pkl"
        self.CF_OBSERVATIONS_SER_FILE = "bpmn_cf_observations.pkl"
        self.MP_OBSERVATIONS_SER_FILE = "bpmn_mp_observations.pkl"
        self.CONSTRAINT_KB_SER_FILE = "constraint_kb.pkl"
        self.PREPROCESSED_CONSTRAINTS = "preprocessed_constraints.pkl"
        self.LOG_INFO = "log_info.pkl"
        # Known models
        self.MODEL_META_SER_FILE = "model_meta.pkl"
        # Known constraints
        self.DECLARE_CONST = "bpmn_declare_const.pkl"
        # Known similarities
        self.SIM_MAP = "sim_map.pkl"
        # Known embeddings
        self.EMB_MAP = "emb_map.pkl"

        self.NON_TASKS = (
        "SequenceFlow", "MessageFlow", "DataObject", "Pool", "Lane", "TextAnnotation", "Association_Undirected",
        "Association_Bidirectional", "Association_Unidirectional", "Group", "CollapsedPool", "ITSystem",
        "DataStore")

        self.BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"
        self.VERB_OCEAN_FILE = "verbocean.txt"

        self.EMAIL_DUMMY = "jane.doe@dummy.com"
        self.NAME_DUMMY = "Jane Doe"
        self.NUMBER_DUMMY = "12345678"

        self.EXAMPLE_MODEL_NAMES = ["Procure parts", "Receipt of Goods", "Receipt of Application",
                                    "Procurement of Work Equipment",
                                    "Employee Onboarding"]

        self.COLORS_SIGNAVIO_HSL = [
            "hsl(309, 88%, 33%)", "hsl(313, 81%, 35%)", "hsl(318, 75%, 36%)", "hsl(322, 70%, 39%)",
            "hsl(326, 65%, 40%)",
            "hsl(331, 60%, 43%)", "hsl(336, 56%, 45%)", "hsl(341, 53%, 46%)", "hsl(352, 47%, 50%)",
            "hsl(358, 48%, 53%)",
            "hsl(4, 51%, 53%)", "hsl(9, 54%, 53%)", "hsl(14, 57%, 52%)", "hsl(18, 60%, 52%)", "hsl(21, 64%, 52%)",
            "hsl(25, 67%, 51%)", "hsl(27, 70%, 51%)", "hsl(30, 73%, 51%)", "hsl(32, 76%, 50%)", "hsl(34, 79%, 50%)",
            "hsl(36, 83%, 50%)", "hsl(38, 87%, 49%)", "hsl(40, 91%, 49%)", "hsl(41, 96%, 48%)", "hsl(43, 100%, 48%)"
        ]

        self.COLORS_SIGNAVIO_ = [
            "#9c0a85", "#a01180", "#a3177a", "#a71e75", "#aa2470", "#ae2b6a", "#b23265", "#b53860", "#b93f5a",
            "#bc4555",
            "#c04c50", "#c4534a", "#c75945", "#cb6040", "#ce663b", "#d26d35", "#d67430", "#d97a2b", "#dd8125",
            "#e08720",
            "#e48e1b", "#e89515", "#eb9b10", "#efa20b", "#f2a805", "#f6af00"
        ]

        # Strings that represent missing values
        self.TERMS_FOR_MISSING = [
            'MISSING', 'UNDEFINED', 'undefined', 'missing', 'nan', 'empty', 'empties', 'unknown',
            'other', 'others', 'na', 'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined',
            'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na', 'nil', 'null'
        ]

        # Timeout for soundness check and Petri net play-out
        self.TIMEOUT = 10

        # XES ATTRIBUTE NAMES
        self.XES_NAME = "concept:name"
        self.XES_TIME = "time:timestamp"
        self.XES_LIFECYCLE = "lifecycle:transition"
        self.XES_RESOURCE = "org:resource"
        self.XES_GROUP = "org:group"
        self.XES_ROLE = "org:role"
        self.XES_CASE = "case:" + self.XES_NAME
        self.XES_INST = "concept:instance"

        # PROPRIETARY ATTRIBUTE NAMES
        self.DETECTED_NAT_LANG = "detected_natural_language"
        self.CLEANED_LABEL = "cleaned label"
        self.SPLIT_LABEL = "split label"
        self.ELEMENT_CATEGORY = "category"
        self.CONSTRAINT_STR = "constraint_string"
        self.LEFT_OPERAND = "left_op"
        self.RIGHT_OPERAND = "right_op"
        self.RECORD_ID = "obs_id"
        self.FITTED_RECORD_ID = "fitted_obs_id"
        self.LEVEL = "Level"
        self.OPERATOR_TYPE = "op_type"
        self.MODEL_NAME = "model_name"
        self.MODEL_ID = "model_id"
        self.MODEL_ID_BACKUP = "m_id"
        self.ELEMENT_ID = "element_id"
        self.LABEL = "label"
        self.ORIGINAL_LABEL = "original_label"
        self.GLOSSARY = "glossary"
        self.LABEL_LIST = "label_list"
        self.NAME = "name"
        self.SUPPORT = "support"
        self.REDUNDANT = "redundant"
        self.NAT_LANG_OBJ = "nat_lang_obj"
        self.NAT_LANG_TEMPLATE = "nat_lang_template"

        # DIFFERENT TYPES OF CONTEXTUAL SIMILARITIES
        self.NAME_BASED_SIM = "name_based_sim"
        self.OBJECT_BASED_SIM = "object_based_sim"
        self.LABEL_BASED_SIM = "label_based_sim"

        self.NAME_BASED_SIM_EXTERNAL = "name_based_sim_external"
        self.LABEL_BASED_SIM_EXTERNAL = "label_based_sim_external"

        self.OBJECT_BASED_SIM_EXTERNAL = "object_based_sim_external"
        self.MAX_OBJECT_BASED_SIM_EXTERNAL = "max_object_based_sim_external"

        # ARITY OF CONSTRAINTS
        self.UNARY = "Unary"
        self.BINARY = "Binary"

        # TYPES OF CONSTRAINTS
        self.OBJECT = "Object"
        self.MULTI_OBJECT = "Multi-object"
        self.RESOURCE = "Resource"
        self.DECISION = "Decision"
        self.LINGUISTIC = "Linguistic"

        # LANGUAGES
        self.EN = "english"
        self.DE = "german"
        self.FR = "french"
        self.ES = "spanish"
        self.NL = "dutch"

        self.LANGUAGE = self.EN

        # All binary constraint templates
        self.BINARY_TEMPLATES = [bin_temp.templ_str for bin_temp in Template.get_binary_templates()]
        # All unary constraint templates
        self.UNARY_TEMPLATES = [un_temp.templ_str for un_temp in Template.get_unary_templates()]
        # All cardinality constraint templates
        self.CARDINALITY_TEMPLATES = [un_temp.templ_str for un_temp in Template.get_cardinality_templates()]

        # Constraints that are really useless
        self.IRRELEVANT_CONSTRAINTS = {
            Template.EXCLUSIVE_CHOICE.templ_str: ["yes", "no"]
        }


        # REDUNDANCY RESOLUTION STRATEGIES
        # Privileges the hierarchy (eager policy): for example, if the constraint set contains AlternatePrecedence(A,
        # B) and Precedence(A, B), the latter is filtered out.
        self.EAGER_ON_HIERARCHY_OVER_SUPPORT = "hos"
        # Privileges the support (eager policy):
        # for example, if the constraint set contains AlternatePrecedence(A, B) and Precedence(A, B),
        # and AlternatePrecedence(A, B) has a support of 0.89
        # whereas Precedence(A, B) has a support of 0.9, then AlternatePrecedence(A, B) is filtered out.
        self.EAGER_ON_SUPPORT_OVER_HIERARCHY = "soh"

        # Only filters out subsuming constraints, and only if ALL the subsuming ones in the whole hierarchy have the same
        # support
        self.CONSERVATIVE = "conservative"

        # The policy to use for redundancy resolution
        self.POLICY = self.EAGER_ON_SUPPORT_OVER_HIERARCHY
        # Terms referring to objects from schema.org
        self.SCHEMA_ORG_OBJ = [tok for obj in
                               ['Product', 'Products', 'Project', 'Product Group', 'Product Model', 'Vehicle',
                                'Alignment Object', 'Audience', 'Bed Details', 'Brand', 'Broadcast Channel',
                                'Broadcast Frequency Specification', 'Class', 'Computer Language', 'Data Feed Item',
                                'Term', 'Demand', 'Document Permission', 'Program', 'Energy Consumption Details',
                                'Entry Point', 'Enumeration', 'Floor Plan', 'Game Server', 'Geospatial Geometry',
                                'Goods', 'Grant', 'Health Insurance Plan', 'Health Plan Cost Specification',
                                'Health Plan Formulary', 'Health Plan Network', 'Invoice', 'Item List',
                                'Job Posting', 'Language', 'List Item', 'Media Subscription', 'Menu Item',
                                'Merchant Return Policy', 'Merchant Return Policy', 'Observation', 'Occupation',
                                'Experience Requirements', 'Offer', 'Order', 'Order Item', 'Parcel Delivery',
                                'Permit', 'Program Membership', 'Property', 'Property Value Specification',
                                'Quantity', 'Rating', 'Reservation', 'Role', 'Schedule', 'Seat', 'Series', 'Service',
                                'Service Channel', 'Speakable Specification', 'Statistical Population',
                                'Structured Value', 'Ticket', 'Trip', 'Virtual Location']
                               for tok in obj.lower().split(" ")]
        self.SCHEMA_ORG_OBJ_PROP = [tok for obj in ['Age', 'Name', 'Type', 'Price', 'Cost'] for tok in
                                    obj.lower().split(" ")]

        # Language models used in the project
        self.SPACY_MODEL = "en_core_web_sm"
        self.SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
