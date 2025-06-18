import datetime
from dataclasses import asdict, dataclass, field
from typing import Any

from fme.core.training_history import get_job_env


@dataclass
class DatasetMetadata:
    source: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    title: str | None = None

    @classmethod
    def from_env(cls) -> "DatasetMetadata":
        """
        Create a DatasetMetadata instance from the job environment.
        """
        return cls(
            source=get_job_env(),
            history={"created": datetime.datetime.now().isoformat()},
        )

    def as_flat_str_dict(self) -> dict[str, str]:
        """
        Convert the metadata to a flat dictionary with string values. Nested
        dictionaries are flattened with dot notation for keys, e.g. {'source': {'b': 1}}
        becomes {'source.b': '1'}.
        """
        return asdict(self, dict_factory=_to_flat_str_dict)


def _to_flat_str_dict(fields: list[tuple[str, str | dict[str, Any]]]) -> dict[str, str]:
    """
    Convert a maybe-nested dictionary to a flat dictionary with string values.
    """
    flat_dict = {}
    for field_name, value in fields:
        if isinstance(value, dict):
            nested = _to_flat_str_dict([(k, v) for k, v in value.items()])
            flat_dict.update({f"{field_name}.{k}": v for k, v in nested.items()})
        else:
            flat_dict[field_name] = str(value)
    return flat_dict
