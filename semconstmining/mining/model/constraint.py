from enum import Enum
from typing import Set

from pydantic.main import BaseModel


class Observation(Enum):
    XOR = "Decision"
    ACT_XOR = "Action exclusion"
    ACT_CO_OCC = "Action co-occurrence"
    ACT_ORDER = "Action order"
    OBJ_XOR = "Object exclusion"
    OBJ_CO_OCC = "Object co-occurrence"
    OBJ_ORDER = "Object order"

    RESOURCE_TASK_EXISTENCE = "Resource task relation"
    RESOURCE_CONTAINMENT = "Parent resource"

    INIT_OBJ = "Initialize object"
    END_OBJ = "Complete object"


class Constraint(BaseModel):
    const: tuple
    const_type: Observation
    count: int
    model_names: Set

    def increment_count(self, count):
        self.count += count

    def add_model_name(self, model_name):
        self.model_names.add(model_name)
