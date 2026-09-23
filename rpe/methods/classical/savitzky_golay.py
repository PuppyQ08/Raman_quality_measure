from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import savgol_filter


SCHEMA_VERSION = "phase05-classical-pipeline-v1"
PIPELINE_ID = "sg11_poly3_interp"
INPUT_CONDITION_ID = "released_input_control"
CONDITION_ID = "released_input_plus_sg"
WINDOW_LENGTH = 11
POLYORDER = 3
DERIV = 0
MODE = "interp"
ARRAY_AXIS = -1
COORDINATE_MODE = "index"
OUTPUT_DTYPE = "float32"


class SavitzkyGolayValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SavitzkyGolayValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SavitzkyGolayValidationError(path, "keys must be strings")
    return value


def _exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise SavitzkyGolayValidationError(
            path,
            (
                f"unexpected key set: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _require_equal(path: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise SavitzkyGolayValidationError(
            path,
            f"must equal {expected!r}",
        )


def _reject_nonfinite_constant(value: str) -> None:
    raise SavitzkyGolayValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


@dataclass(frozen=True, init=False)
class SavitzkyGolayPipeline:
    pipeline_id: str = PIPELINE_ID
    input_condition_id: str = INPUT_CONDITION_ID
    condition_id: str = CONDITION_ID
    window_length: int = WINDOW_LENGTH
    polyorder: int = POLYORDER
    deriv: int = DERIV
    mode: str = MODE
    array_axis: int = ARRAY_AXIS
    coordinate_mode: str = COORDINATE_MODE
    output_dtype: str = OUTPUT_DTYPE

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        if not isinstance(spectra, np.ndarray):
            raise SavitzkyGolayValidationError(
                "spectra",
                "must be a numpy.ndarray",
            )
        if spectra.dtype != np.dtype("<f4"):
            raise SavitzkyGolayValidationError(
                "spectra dtype",
                "must be little-endian float32",
            )
        if spectra.ndim not in (1, 2):
            raise SavitzkyGolayValidationError(
                "spectra dimension",
                "must be one- or two-dimensional",
            )
        if spectra.size == 0:
            raise SavitzkyGolayValidationError(
                "spectra empty",
                "must not be empty",
            )
        if spectra.shape[-1] < self.window_length:
            raise SavitzkyGolayValidationError(
                "spectra length",
                f"must be at least {self.window_length}",
            )
        if not np.isfinite(spectra).all():
            raise SavitzkyGolayValidationError(
                "spectra finite",
                "must contain only finite values",
            )
        transformed = np.asarray(
            savgol_filter(
                spectra,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                axis=self.array_axis,
                mode=self.mode,
            ),
            dtype="<f4",
        ).copy()
        if not np.isfinite(transformed).all():
            raise SavitzkyGolayValidationError(
                "transformed finite",
                "contains non-finite values",
            )
        transformed.setflags(write=False)
        return transformed


def load_savitzky_golay_pipeline(path: Path) -> SavitzkyGolayPipeline:
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=_reject_nonfinite_constant,
        )
    except SavitzkyGolayValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SavitzkyGolayValidationError(
            path.name or "config",
            str(error),
        ) from error
    document = _object("config", value)
    if raw != _canonical_json_bytes(document):
        raise SavitzkyGolayValidationError(
            "config noncanonical",
            "must use canonical JSON encoding",
        )
    _exact_keys(
        "config",
        document,
        {
            "schema_version",
            "pipeline_id",
            "input_condition_id",
            "condition_id",
            "output_dtype",
            "steps",
        },
    )
    _require_equal(
        "schema_version",
        document["schema_version"],
        SCHEMA_VERSION,
    )
    _require_equal("pipeline_id", document["pipeline_id"], PIPELINE_ID)
    _require_equal(
        "input_condition_id",
        document["input_condition_id"],
        INPUT_CONDITION_ID,
    )
    _require_equal("condition_id", document["condition_id"], CONDITION_ID)
    _require_equal("output_dtype", document["output_dtype"], OUTPUT_DTYPE)
    steps = document["steps"]
    if not isinstance(steps, list) or len(steps) != 1:
        raise SavitzkyGolayValidationError(
            "steps",
            "must contain exactly one step",
        )
    step = _object("steps[0]", steps[0])
    _exact_keys(
        "steps[0]",
        step,
        {
            "operation",
            "window_length",
            "polyorder",
            "deriv",
            "mode",
            "array_axis",
            "coordinate_mode",
        },
    )
    _require_equal("operation", step["operation"], "savitzky_golay")
    _require_equal(
        "window_length",
        step["window_length"],
        WINDOW_LENGTH,
    )
    _require_equal("polyorder", step["polyorder"], POLYORDER)
    _require_equal("deriv", step["deriv"], DERIV)
    _require_equal("mode", step["mode"], MODE)
    _require_equal("array_axis", step["array_axis"], ARRAY_AXIS)
    _require_equal(
        "coordinate_mode",
        step["coordinate_mode"],
        COORDINATE_MODE,
    )
    return SavitzkyGolayPipeline()


__all__ = [
    "SavitzkyGolayPipeline",
    "SavitzkyGolayValidationError",
    "load_savitzky_golay_pipeline",
]
