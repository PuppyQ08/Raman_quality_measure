from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import stat
import struct
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.io.schema import axis_id
from rpe.io.sugar_mixtures_models import (
    SugarMixturesInspection,
    SugarMixturesValidationError,
    _SugarAcceptedMember,
    _SugarEvidenceMember,
    _SugarEvidenceSnapshot,
    _SugarRecipe,
    _SugarSourceContract,
)


CANONICAL_HEADER = (
    "Pixel",
    "wl",
    "cm-1",
    "Intensity",
    "Metadata",
)

SOURCE_METADATA_KEYS = (
    "Date",
    "Position [um]",
    "Temperature [C]",
    "Humidity [1/100]",
    "Spectro_Temp [C]",
    "Laser power [mW]",
    "Excitation wavelength [nm]",
    "Objective",
    "Objective_Maker",
    "Objective_Magnification",
    "Objective_NA",
    "Objective_WD",
    "Objective_Immersion",
    "Objective_Tube_Lens_f",
    "Spectrometer integration time [s]",
    "Spectrometer number of accumulations",
    "Spectrometer temperature [C]",
    "Spectrometer acquisition mode",
    "Spectrometer read mode",
    "Spectrometer trigger mode",
    "Magnification",
    "Spot_Size",
    "Laser_Offset",
)

RAW_PREFIX = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
)
EXPERIMENTAL_PREFIX = "Raw data/Experimental data from sugar mixtures/"
HIGH_DIRECTORY = f"{RAW_PREFIX}Sugar_Concentration_Test/"
LOW_DIRECTORY = f"{RAW_PREFIX}Sugar_Concentration_Test_Fast/"
TARGET_MEMBER = f"{RAW_PREFIX}Sugar_Concentrations.csv"
ORACLE_MEMBERS = frozenset(
    {
        f"{RAW_PREFIX}Sugar_Concentration_Test_ALL_spectra.csv",
        f"{RAW_PREFIX}Sugar_Concentration_Test_ALL_metadata.csv",
        f"{RAW_PREFIX}Sugar_Concentration_Test_Fast_ALL_spectra.csv",
        f"{RAW_PREFIX}Sugar_Concentration_Test_Fast_ALL_metadata.csv",
    }
)
CLAIM_README = f"{RAW_PREFIX}README.md"
EXPERIMENTAL_README = (
    "Raw data/Experimental data from sugar mixtures/README.txt"
)
PREPARED_PREFIX = (
    "Raw data/Experimental data from sugar mixtures/"
    "Raw datasets for analyses/"
)

TARGET_HEADER = (
    "Well",
    "Samp",
    "Row",
    "Column",
    "Plate",
    "Sucrose [ul]",
    "Fructose [ul]",
    "Maltose [ul]",
    "Glucose [ul]",
    "Water [ul]",
    "Total Volume [ul]",
)
COMPONENTS = (
    "sucrose",
    "fructose",
    "maltose",
    "glucose",
    "water",
)
TARGET_COMPONENTS = COMPONENTS[:-1]
ALLOWED_SUGAR_VOLUMES_UL = frozenset({0, 30, 75, 120, 375})
VIEW_IDS = (
    "high_snr",
    "high_snr_no_refs",
    "high_snr_pure_reference",
    "low_snr",
    "low_snr_no_refs",
    "low_snr_pure_reference",
    "no_refs",
    "pure_reference",
)

_HIGH_BASENAME_PATTERN = re.compile(
    r"^Sugar_Concentration_Test_"
    r"(?P<samp>\d+)_"
    r"(?P<row>[A-H])(?P<column>\d{1,2})_"
    r"(?P<plate>[1-3])_"
    r"RD(?P<round>\d+)_"
    r"M(?P<measurement>\d+)_"
    r"R(?P<repetition>\d+)\.csv$"
)
_LOW_BASENAME_PATTERN = re.compile(
    r"^Sugar_Concentration_Test_Fast_"
    r"(?P<samp>\d+)_"
    r"(?P<row>[A-H])(?P<column>\d{1,2})_"
    r"(?P<plate>[1-3])_"
    r"RD(?P<round>\d+)_"
    r"M(?P<measurement>\d+)_"
    r"R(?P<repetition>\d+)\.csv$"
)

_ROLE_AUTHORITIES = MappingProxyType(
    {
        "canonical_high": "canonical_records",
        "canonical_low": "canonical_records",
        "target": "target_authority",
        "oracle": "comparison_only",
        "prepared": "derived_comparison_only",
        "raw_support": "evidence_only",
        "experimental_readme": "evidence_only",
        "excluded_synthetic": "inventory_only",
    }
)

_PRODUCTION_SOURCE_CONTRACT = _SugarSourceContract(
    archive_name="Raw data.zip",
    archive_bytes=5_310_910_565,
    archive_md5="e6e192d197d7b1b5e47e340f255566a4",
    archive_sha256=(
        "0924efd1171416efe14a31108f6ac7a00124d1bfb2a06c4fb22045cc3f2a265b"
    ),
    archive_member_count=10_298,
    archive_uncompressed_bytes=6_733_535_694,
    central_directory_inventory_sha256=(
        "1eed5e33619b6943da112e5bfa3c6a6f61a431d31b42f810a529146a4cbcbd06"
    ),
    expected_role_counts=MappingProxyType(
        {
            "canonical_high": 1_960,
            "canonical_low": 7_840,
            "target": 1,
            "oracle": 4,
            "prepared": 22,
            "raw_support": 18,
            "experimental_readme": 1,
            "excluded_synthetic": 351,
        }
    ),
    expected_role_bytes=MappingProxyType(
        {
            "canonical_high": 134_524_655,
            "canonical_low": 534_221_640,
            "target": 6_174,
            "oracle": 119_877_054,
            "prepared": 313_204_630,
            "raw_support": 266_433,
            "experimental_readme": 252,
            "excluded_synthetic": 5_631_434_856,
        }
    ),
    expected_role_sha256=MappingProxyType(
        {
            "canonical_high": (
                "f71004d11a181d6cb0c82922d183ba96054ab8f3bbe3868b835d153d34d6d58a"
            ),
            "canonical_low": (
                "d9829b17ba5a24ae99d23e30c8bdfc84a9190a38da995755c4642cf5776de1e3"
            ),
            "target": (
                "e7902d36cbfbdc56f81aabcbfff197825ab42dc78b2ce43220098569700a8a4d"
            ),
            "oracle": (
                "fcdb85dea2015345350426193fbbfc48994ec084c48b507aa6d07189320464dc"
            ),
            "prepared": (
                "77e3fa307334cf28f6b5a4575e8698c9c3b1b3c18e241e9e82bb49c44974093d"
            ),
            "raw_support": (
                "e45d4c69affacce534d2f4d2c3175e93df9b2c8c39f43362bbf3d151d051ab95"
            ),
            "experimental_readme": (
                "86a0d4d552a2a366656f410bd2100a9afa2faae2afd746e85a2296b703a7df6f"
            ),
        }
    ),
    expected_evidence_snapshots=MappingProxyType(
        {
            "evidence/zenodo/10779223.json": (
                4_380,
                "dc6b0a16a5a67fb773614dfbe831aad36943929aa7e0c168e851d5d9f33abb9d",
            ),
            "receipts/sugar_mixtures_high_snr.json": (
                864,
                "ae5194f43694589aba1dc7ffee9b1b2a88629cd6100fc9a2029dc1b8ac6d0a3d",
            ),
            "receipts/sugar_mixtures_low_snr.json": (
                863,
                "08d7fd950e4d44f251caa17524d5f6984faaa581117917bc6337d51db5655bf5",
            ),
        }
    ),
    expected_counts=MappingProxyType(
        {
            "canonical_records": 9_800,
            "high_records": 1_960,
            "low_records": 7_840,
            "high_records_per_well": 8,
            "low_records_per_well": 32,
            "high_round_count": 2,
            "low_round_count": 8,
            "high_repetitions_per_round": 4,
            "low_repetitions_per_round": 4,
            "sample_count": 245,
            "points_per_record": 2_000,
            "maximum_record_id_bytes": 35,
            "identity_map_sha256": (
                "1b1d1e34b5ea8d86747122ae462e3fbfd0187f85368e4d4c7bcfb8650578fdb4"
            ),
            "directory_entries": 101,
            "regular_deflated_files": 10_197,
            "maximum_utf8_path_bytes": 144,
            "relevant_regular_members": 9_846,
            "relevant_evidence_members": 46,
            "excluded_synthetic_regular_members": 351,
            "pure_reference_records": 200,
            "mixture_records": 9_600,
            "core_axis_id": (
                "70d347ab6bde277a8f41e10b342fcf69db079e0dee4fff743fa34bea43a443c6"
            ),
            "pixel_axis_id": (
                "932316207d5ffd88f26c88f33a3ee7a0972eea39f794596056d45c040efd9d1b"
            ),
            "wavelength_axis_id": (
                "4a6fc3456f9edadcd06dca59f1be05f93d92e2a51683da862917dc377032b707"
            ),
            "auxiliary_axis_set_id": (
                "360af979ace7daf3898963af75425fe02b2ff980fe584ff36c539686ed11e047"
            ),
            "raw_text_inventory_sha256": (
                "0731930a28616c61722dac48aca43c9cf6dad7c368c5f03e2f3e6c057a37eb18"
            ),
            "normalized_metadata_sha256": (
                "7ae0bbfa28d3b79c7616a6224925012598e79bd2e7fc2e2b6377e90fd91d5cee"
            ),
            "core_metadata_sha256": (
                "d41c4a3351733059a0eef4446f5aad464ec3e481418c573655cbf56509475e84"
            ),
            "well_target_contract_sha256": (
                "075a2a963080716a1163a21e74526ec394b240e9484861d7c392d0ca90878544"
            ),
            "record_target_map_sha256": (
                "6c9b7a1f29a043c62a850dcad778369547ade5e61446a206b0b3af78a8b650fb"
            ),
            "record_preprocessing_map_sha256": (
                "a9df0e8a9cda14f1601c2cc5874b9cb861607edb01e0e008c29b67151077514f"
            ),
            "preprocessing_evidence_document_sha256": (
                "568355ac55b4af8aa08567111f1d4ac40e3aefb2b4770f5f6138e41bdaba7a51"
            ),
            "intensity_float32_max_abs_error": 0.0,
            "axis_float32_max_abs_error_cm1": (
                0.00012201562503832974
            ),
            "wavelength_float32_max_abs_error_nm": (
                0.000060546874919964466
            ),
            "pixel_float64_sha256": (
                "2eaa447f6c931a6a742c5af375e6980a62fa4a63a6c66a1b6d7f263d425714d7"
            ),
            "wavelength_float64_sha256": (
                "dc9fe7694416d3fffcf524c66c60eef42679653bc6786186c9a07a3b98785f7e"
            ),
            "wavenumber_float64_sha256": (
                "f6ced120b1c5f55d0c64e7671f355ec4c8420cd4788f82b4d40ecf986e4729cf"
            ),
        }
    ),
)

_STRING_FIELDS = {
    "Date",
    "Temperature [C]",
    "Humidity [1/100]",
    "Objective",
    "Objective_Maker",
    "Objective_Immersion",
}
_FLOAT_FIELDS = {
    "Laser power [mW]",
    "Objective_Magnification",
    "Objective_NA",
    "Objective_WD",
    "Objective_Tube_Lens_f",
    "Magnification",
    "Spot_Size",
}
_INTEGER_FIELDS = {
    "Spectro_Temp [C]",
    "Excitation wavelength [nm]",
    "Spectrometer number of accumulations",
    "Spectrometer temperature [C]",
    "Spectrometer acquisition mode",
    "Spectrometer read mode",
    "Spectrometer trigger mode",
}
_SPECIAL_FIELDS = {
    "Position [um]",
    "Laser_Offset",
    "Spectrometer integration time [s]",
}

assert set(SOURCE_METADATA_KEYS) == (
    _STRING_FIELDS | _FLOAT_FIELDS | _INTEGER_FIELDS | _SPECIAL_FIELDS
)


@dataclass(frozen=True)
class _ParsedSugarMember:
    source_member: str
    source_member_sha256: str
    pixel: np.ndarray
    wavelength_nm: np.ndarray
    wavenumber_cm1: np.ndarray
    intensity: np.ndarray
    source_json_text: str
    source_metadata: Mapping[str, object]


@dataclass(frozen=True)
class _SugarArchivePreflight:
    archive_path: Path
    archive_md5: str
    archive_sha256: str
    central_directory_inventory_sha256: str
    central_directory_safety: Mapping[str, int]
    infos: tuple[zipfile.ZipInfo, ...]
    role_infos: Mapping[str, tuple[zipfile.ZipInfo, ...]]
    member_sha256: Mapping[str, str]


def _fatal(
    path: str,
    reason: str,
    code: str,
) -> SugarMixturesValidationError:
    return SugarMixturesValidationError(path, reason, code)


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


def _parse_json_constant(value: str) -> float:
    if value == "NaN":
        return float("nan")
    raise _fatal(
        "source_metadata",
        f"unsupported nonfinite constant {value!r}",
        "METADATA_NONFINITE_INVALID",
    )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _require_metadata_type(
    field: str,
    value: object,
) -> None:
    if field in _STRING_FIELDS:
        valid = isinstance(value, str)
    elif field == "Position [um]":
        valid = (
            isinstance(value, list)
            and len(value) == 3
            and all(_is_number(item) for item in value)
        )
    elif field == "Laser_Offset":
        valid = (
            isinstance(value, list)
            and len(value) == 2
            and all(_is_number(item) for item in value)
        )
    elif field == "Spectrometer integration time [s]":
        valid = _is_number(value)
    elif field in _INTEGER_FIELDS:
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = isinstance(value, float)
    if not valid:
        if field == "Laser_Offset":
            raise _fatal(
                "source_metadata.Laser_Offset",
                "must contain exactly two source NaN values",
                "METADATA_NONFINITE_INVALID",
            )
        raise _fatal(
            f"source_metadata.{field}",
            f"invalid value type {type(value).__name__}",
            "METADATA_TYPE_INVALID",
        )


def _freeze_metadata(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    frozen: dict[str, object] = {}
    for field in SOURCE_METADATA_KEYS:
        value = metadata[field]
        if isinstance(value, list):
            frozen[field] = tuple(value)
        else:
            frozen[field] = value
    return MappingProxyType(frozen)


def _metadata_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _fatal(
                f"source_metadata.{key}",
                "duplicate key",
                "METADATA_KEYS_INVALID",
            )
        result[key] = value
    return result


def _validate_metadata(
    source_json_text: str,
) -> Mapping[str, object]:
    try:
        metadata = json.loads(
            source_json_text,
            parse_constant=_parse_json_constant,
            object_pairs_hook=_metadata_object,
        )
    except SugarMixturesValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise _fatal(
            "source_metadata",
            str(error),
            "METADATA_JSON_INVALID",
        ) from error
    if not isinstance(metadata, dict):
        raise _fatal(
            "source_metadata",
            "must be a JSON object",
            "METADATA_TYPE_INVALID",
        )
    if tuple(metadata) != SOURCE_METADATA_KEYS:
        raise _fatal(
            "source_metadata",
            "must have the exact ordered 23-key source schema",
            "METADATA_KEYS_INVALID",
        )
    for field in SOURCE_METADATA_KEYS:
        value = metadata[field]
        _require_metadata_type(field, value)
        if field == "Laser_Offset":
            if not all(
                isinstance(item, float) and math.isnan(item)
                for item in value
            ):
                raise _fatal(
                    "source_metadata.Laser_Offset",
                    "must contain exactly two source NaN values",
                    "METADATA_NONFINITE_INVALID",
                )
            continue
        values: Sequence[object] = (
            value if isinstance(value, list) else (value,)
        )
        if any(
            _is_number(item) and not math.isfinite(float(item))
            for item in values
        ):
            raise _fatal(
                f"source_metadata.{field}",
                "must be finite",
                "METADATA_NONFINITE_INVALID",
            )
    return _freeze_metadata(metadata)


def _parse_float(
    value: str,
    *,
    path: str,
) -> float:
    if value == "":
        raise _fatal(
            path,
            "must not be empty",
            "NUMERIC_ROW_INVALID",
        )
    try:
        parsed = float(value)
    except ValueError as error:
        raise _fatal(
            path,
            str(error),
            "NUMERIC_ROW_INVALID",
        ) from error
    if not math.isfinite(parsed):
        raise _fatal(
            path,
            "must be finite",
            "NUMERIC_NONFINITE",
        )
    return parsed


def _readonly_float64(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _parse_canonical_member_bytes(
    payload: bytes,
    *,
    source_member: str,
    expected_points: int,
) -> _ParsedSugarMember:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fatal(
            "source_member.encoding",
            str(error),
            "ENCODING_INVALID",
        ) from error
    try:
        rows = list(
            csv.reader(
                io.StringIO(text, newline=""),
                strict=True,
            )
        )
    except csv.Error as error:
        raise _fatal(
            "source_member.csv",
            str(error),
            "CSV_INVALID",
        ) from error
    if not rows or tuple(rows[0]) != CANONICAL_HEADER:
        raise _fatal(
            "source_member.header",
            f"expected {CANONICAL_HEADER!r}",
            "HEADER_INVALID",
        )

    numeric_rows: list[tuple[float, float, float, float]] = []
    metadata_rows: list[tuple[int, str]] = []
    for index, row in enumerate(rows[1:], start=1):
        row_path = f"source_member.rows[{index}]"
        if len(row) != len(CANONICAL_HEADER):
            raise _fatal(
                row_path,
                "must contain exactly five columns",
                "ROW_WIDTH_INVALID",
            )
        numeric_values = row[:4]
        metadata_text = row[4]
        has_numeric = any(value != "" for value in numeric_values)
        if metadata_text != "":
            if has_numeric:
                raise _fatal(
                    row_path,
                    "metadata row must not contain numeric fields",
                    "METADATA_ROW_POSITION",
                )
            metadata_rows.append((index, metadata_text))
            continue
        if not has_numeric:
            raise _fatal(
                row_path,
                "empty row is not permitted",
                "NUMERIC_ROW_INVALID",
            )
        if metadata_rows:
            raise _fatal(
                row_path,
                "numeric data must precede the metadata row",
                "METADATA_ROW_POSITION",
            )
        numeric_rows.append(
            tuple(
                _parse_float(
                    value,
                    path=f"{row_path}[{column}]",
                )
                for column, value in enumerate(numeric_values)
            )
        )

    if not metadata_rows:
        raise _fatal(
            "source_member.metadata",
            "exactly one metadata row is required",
            "METADATA_ROW_MISSING",
        )
    if len(metadata_rows) != 1:
        raise _fatal(
            "source_member.metadata",
            "exactly one metadata row is required",
            "METADATA_ROW_DUPLICATE",
        )
    if len(numeric_rows) != expected_points:
        raise _fatal(
            "source_member.numeric_rows",
            f"expected {expected_points}, observed {len(numeric_rows)}",
            "POINT_COUNT_MISMATCH",
        )

    numeric = np.asarray(numeric_rows, dtype=np.float64)
    pixel_values = numeric[:, 0]
    if np.any(pixel_values != np.floor(pixel_values)):
        raise _fatal(
            "source_member.pixel",
            "must contain integer-valued coordinates",
            "PIXEL_NONINTEGER",
        )
    if len(set(pixel_values.tolist())) != pixel_values.size:
        raise _fatal(
            "source_member.pixel",
            "contains duplicate coordinates",
            "PIXEL_DUPLICATE",
        )
    for name, values in (
        ("pixel", pixel_values),
        ("wavelength_nm", numeric[:, 1]),
        ("wavenumber_cm1", numeric[:, 2]),
    ):
        if np.any(np.diff(values) <= 0):
            raise _fatal(
                f"source_member.{name}",
                "must be strictly increasing",
                "AXIS_NONMONOTONIC",
            )

    source_json_text = metadata_rows[0][1]
    source_metadata = _validate_metadata(source_json_text)
    return _ParsedSugarMember(
        source_member=source_member,
        source_member_sha256=hashlib.sha256(payload).hexdigest(),
        pixel=_readonly_float64(pixel_values),
        wavelength_nm=_readonly_float64(numeric[:, 1]),
        wavenumber_cm1=_readonly_float64(numeric[:, 2]),
        intensity=_readonly_float64(numeric[:, 3]),
        source_json_text=source_json_text,
        source_metadata=source_metadata,
    )


def _hash_file_md5_sha256(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                md5.update(block)
                sha256.update(block)
    except OSError as error:
        raise _fatal(
            "raw_root.archive",
            str(error),
            "ARCHIVE_READ_FAILED",
        ) from error
    return md5.hexdigest(), sha256.hexdigest()


def _zip_inventory_sha256(infos: Sequence[zipfile.ZipInfo]) -> str:
    digest = hashlib.sha256(b"rpe-sugar-zip-inventory-v1\0")
    for info in sorted(
        infos,
        key=lambda value: value.filename.encode("utf-8"),
    ):
        encoded_path = info.filename.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded_path)))
        digest.update(encoded_path)
        digest.update(struct.pack("<B", int(info.is_dir())))
        digest.update(
            struct.pack(
                "<IQQHHI",
                info.CRC,
                info.file_size,
                info.compress_size,
                info.compress_type,
                info.flag_bits,
                info.external_attr,
            )
        )
    return digest.hexdigest()


def _path_collision_count(
    names: Sequence[str],
    normalize,
) -> int:
    values = Counter(normalize(name) for name in names)
    return sum(count - 1 for count in values.values() if count > 1)


def _central_directory_safety(
    infos: Sequence[zipfile.ZipInfo],
) -> Mapping[str, int]:
    names = tuple(info.filename for info in infos)
    duplicate_paths = sum(
        count - 1
        for count in Counter(names).values()
        if count > 1
    )
    if duplicate_paths:
        raise _fatal(
            "archive.members",
            "duplicate central-directory member path",
            "MEMBER_PATH_DUPLICATE",
        )

    for name in names:
        path = PurePosixPath(name)
        if (
            name.startswith("/")
            or re.match(r"^[A-Za-z]:/", name) is not None
            or ".." in path.parts
        ):
            raise _fatal(
                f"archive.members.{name}",
                "absolute and traversal paths are forbidden",
                "MEMBER_PATH_UNSAFE",
            )
        if "\\" in name:
            raise _fatal(
                f"archive.members.{name}",
                "backslash paths are forbidden",
                "MEMBER_PATH_BACKSLASH",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise _fatal(
                "archive.members",
                "control and NUL characters are forbidden",
                "MEMBER_PATH_CONTROL",
            )

    nfc_collisions = _path_collision_count(
        names,
        lambda name: unicodedata.normalize("NFC", name),
    )
    if nfc_collisions:
        raise _fatal(
            "archive.members",
            "NFC-normalized member paths collide",
            "MEMBER_PATH_NFC_COLLISION",
        )
    casefold_collisions = _path_collision_count(
        names,
        lambda name: name.casefold(),
    )
    if casefold_collisions:
        raise _fatal(
            "archive.members",
            "case-folded member paths collide",
            "MEMBER_PATH_CASEFOLD_COLLISION",
        )

    for name in names:
        if not name.isascii():
            raise _fatal(
                f"archive.members.{name}",
                "non-ASCII member paths are forbidden",
                "MEMBER_PATH_NONASCII",
            )

    directory_entries = 0
    regular_deflated_files = 0
    for info in infos:
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type == stat.S_IFLNK:
            raise _fatal(
                f"archive.members.{info.filename}",
                "symbolic-link members are forbidden",
                "MEMBER_SYMLINK",
            )
        if info.flag_bits & 0x1:
            raise _fatal(
                f"archive.members.{info.filename}",
                "encrypted members are forbidden",
                "MEMBER_ENCRYPTED",
            )
        if info.is_dir():
            if file_type not in (0, stat.S_IFDIR):
                raise _fatal(
                    f"archive.members.{info.filename}",
                    "directory member has unsupported type bits",
                    "MEMBER_TYPE_UNSUPPORTED",
                )
            directory_entries += 1
            continue
        if file_type not in (0, stat.S_IFREG):
            raise _fatal(
                f"archive.members.{info.filename}",
                "member is not a regular file",
                "MEMBER_TYPE_UNSUPPORTED",
            )
        if info.compress_type != zipfile.ZIP_DEFLATED:
            raise _fatal(
                f"archive.members.{info.filename}",
                "regular members must use ZIP deflate",
                "MEMBER_COMPRESSION_UNSUPPORTED",
            )
        regular_deflated_files += 1

    return MappingProxyType(
        {
            "duplicate_paths": 0,
            "absolute_or_traversal_paths": 0,
            "backslash_paths": 0,
            "control_or_nul_paths": 0,
            "non_ascii_paths": 0,
            "nfc_collisions": 0,
            "casefold_collisions": 0,
            "symlink_members": 0,
            "encrypted_members": 0,
            "unsupported_member_types": 0,
            "directory_entries": directory_entries,
            "regular_deflated_files": regular_deflated_files,
        }
    )


def _classify_member(info: zipfile.ZipInfo) -> str | None:
    if info.is_dir():
        return None
    name = info.filename
    if name.startswith(HIGH_DIRECTORY):
        if PurePosixPath(name).parent.as_posix() != HIGH_DIRECTORY.rstrip("/"):
            raise _fatal(
                f"archive.members.{name}",
                "canonical members must be direct children of the High directory",
                "CANONICAL_FILENAME_INVALID",
            )
        if _HIGH_BASENAME_PATTERN.fullmatch(PurePosixPath(name).name) is None:
            raise _fatal(
                f"archive.members.{name}",
                "unexpected member in High canonical directory",
                "CANONICAL_FILENAME_INVALID",
            )
        return "canonical_high"
    if name.startswith(LOW_DIRECTORY):
        if PurePosixPath(name).parent.as_posix() != LOW_DIRECTORY.rstrip("/"):
            raise _fatal(
                f"archive.members.{name}",
                "canonical members must be direct children of the Low directory",
                "CANONICAL_FILENAME_INVALID",
            )
        if _LOW_BASENAME_PATTERN.fullmatch(PurePosixPath(name).name) is None:
            raise _fatal(
                f"archive.members.{name}",
                "unexpected member in Low canonical directory",
                "CANONICAL_FILENAME_INVALID",
            )
        return "canonical_low"
    if name == TARGET_MEMBER:
        return "target"
    if name in ORACLE_MEMBERS:
        return "oracle"
    if name.startswith(PREPARED_PREFIX):
        return "prepared"
    if name == EXPERIMENTAL_README:
        return "experimental_readme"
    if name.startswith(RAW_PREFIX):
        return "raw_support"
    if name.startswith(EXPERIMENTAL_PREFIX):
        raise _fatal(
            f"archive.members.{name}",
            "unclassified member inside Sugar experimental subtree",
            "MEMBER_ROLE_UNKNOWN",
        )
    return "excluded_synthetic"


def _hash_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> str:
    digest = hashlib.sha256()
    try:
        with archive.open(info, "r") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            f"archive.members.{info.filename}",
            str(error),
            "ARCHIVE_CRC_FAILURE",
        ) from error
    return digest.hexdigest()


def _role_content_sha256(
    infos: Sequence[zipfile.ZipInfo],
    member_sha256: Mapping[str, str],
) -> str:
    digest = hashlib.sha256(b"rpe-sugar-member-content-v1\0")
    for info in sorted(
        infos,
        key=lambda value: value.filename.encode("utf-8"),
    ):
        encoded_path = info.filename.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded_path)))
        digest.update(encoded_path)
        digest.update(bytes.fromhex(member_sha256[info.filename]))
    return digest.hexdigest()


def _preflight_archive(
    raw_root: Path,
    contract: _SugarSourceContract,
) -> _SugarArchivePreflight:
    raw_root = Path(raw_root).resolve()
    archive_path = raw_root / contract.archive_name
    if not archive_path.is_file():
        raise _fatal(
            "raw_root.archive",
            f"missing {contract.archive_name}",
            "ARCHIVE_MISSING",
        )
    try:
        unexpected = tuple(
            sorted(
                (
                    path.name
                    for path in raw_root.glob("*.zip")
                    if path.is_file() and path.name != contract.archive_name
                ),
                key=lambda value: value.encode("utf-8"),
            )
        )
    except OSError as error:
        raise _fatal(
            "raw_root.archives",
            str(error),
            "ARCHIVE_READ_FAILED",
        ) from error
    if unexpected:
        raise _fatal(
            "raw_root.archives",
            f"unexpected ZIP archives: {unexpected}",
            "ARCHIVE_UNEXPECTED",
        )

    try:
        archive_bytes = archive_path.stat().st_size
    except OSError as error:
        raise _fatal(
            "archive.bytes",
            str(error),
            "ARCHIVE_READ_FAILED",
        ) from error
    if archive_bytes != contract.archive_bytes:
        raise _fatal(
            "archive.bytes",
            f"expected {contract.archive_bytes}, observed {archive_bytes}",
            "ARCHIVE_BYTES_MISMATCH",
        )
    archive_md5, archive_sha256 = _hash_file_md5_sha256(archive_path)
    if archive_md5 != contract.archive_md5:
        raise _fatal(
            "archive.md5",
            f"expected {contract.archive_md5}, observed {archive_md5}",
            "ARCHIVE_MD5_MISMATCH",
        )
    if archive_sha256 != contract.archive_sha256:
        raise _fatal(
            "archive.sha256",
            f"expected {contract.archive_sha256}, observed {archive_sha256}",
            "ARCHIVE_SHA256_MISMATCH",
        )

    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = tuple(archive.infolist())
            safety = _central_directory_safety(infos)
            for name in ("directory_entries", "regular_deflated_files"):
                expected = contract.expected_counts.get(name)
                if expected is not None and safety[name] != expected:
                    raise _fatal(
                        f"archive.central_directory_safety.{name}",
                        f"expected {expected}, observed {safety[name]}",
                        "ARCHIVE_SAFETY_COUNT_MISMATCH",
                    )
            maximum_path_bytes = max(
                (
                    len(info.filename.encode("utf-8"))
                    for info in infos
                ),
                default=0,
            )
            expected_maximum_path_bytes = contract.expected_counts.get(
                "maximum_utf8_path_bytes"
            )
            if (
                expected_maximum_path_bytes is not None
                and maximum_path_bytes != expected_maximum_path_bytes
            ):
                raise _fatal(
                    "archive.central_directory_safety.maximum_utf8_path_bytes",
                    (
                        f"expected {expected_maximum_path_bytes}, "
                        f"observed {maximum_path_bytes}"
                    ),
                    "ARCHIVE_SAFETY_COUNT_MISMATCH",
                )
            if len(infos) != contract.archive_member_count:
                raise _fatal(
                    "archive.member_count",
                    (
                        f"expected {contract.archive_member_count}, "
                        f"observed {len(infos)}"
                    ),
                    "ARCHIVE_MEMBER_COUNT_MISMATCH",
                )
            uncompressed_bytes = sum(info.file_size for info in infos)
            if uncompressed_bytes != contract.archive_uncompressed_bytes:
                raise _fatal(
                    "archive.uncompressed_bytes",
                    (
                        f"expected {contract.archive_uncompressed_bytes}, "
                        f"observed {uncompressed_bytes}"
                    ),
                    "ARCHIVE_UNCOMPRESSED_BYTES_MISMATCH",
                )
            inventory_sha256 = _zip_inventory_sha256(infos)
            if inventory_sha256 != contract.central_directory_inventory_sha256:
                raise _fatal(
                    "archive.central_directory_inventory_sha256",
                    (
                        f"expected {contract.central_directory_inventory_sha256}, "
                        f"observed {inventory_sha256}"
                    ),
                    "ARCHIVE_INVENTORY_MISMATCH",
                )
            failed_member = archive.testzip()
            if failed_member is not None:
                raise _fatal(
                    f"archive.members.{failed_member}",
                    "ZIP CRC verification failed",
                    "ARCHIVE_CRC_FAILURE",
                )

            role_lists = {
                role: []
                for role in _ROLE_AUTHORITIES
            }
            for info in infos:
                role = _classify_member(info)
                if role is not None:
                    role_lists[role].append(info)
            role_infos = {
                role: tuple(
                    sorted(
                        current_infos,
                        key=lambda value: value.filename.encode("utf-8"),
                    )
                )
                for role, current_infos in role_lists.items()
            }
            member_sha256 = {
                info.filename: _hash_zip_member(archive, info)
                for role in role_infos
                if role != "excluded_synthetic"
                for info in role_infos[role]
            }
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "archive",
            str(error),
            "ARCHIVE_CRC_FAILURE",
        ) from error

    observed_counts = {
        role: len(role_infos[role])
        for role in role_infos
    }
    observed_bytes = {
        role: sum(info.file_size for info in role_infos[role])
        for role in role_infos
    }
    if set(observed_counts) != set(contract.expected_role_counts):
        raise _fatal(
            "archive.member_roles",
            "role-count contract has an unexpected key set",
            "ROLE_COUNT_MISMATCH",
        )
    for role, expected in contract.expected_role_counts.items():
        if observed_counts[role] != expected:
            raise _fatal(
                f"archive.member_roles.{role}.member_count",
                f"expected {expected}, observed {observed_counts[role]}",
                "ROLE_COUNT_MISMATCH",
            )
    if set(observed_bytes) != set(contract.expected_role_bytes):
        raise _fatal(
            "archive.member_roles",
            "role-byte contract has an unexpected key set",
            "ROLE_BYTES_MISMATCH",
        )
    for role, expected in contract.expected_role_bytes.items():
        if observed_bytes[role] != expected:
            raise _fatal(
                f"archive.member_roles.{role}.uncompressed_bytes",
                f"expected {expected}, observed {observed_bytes[role]}",
                "ROLE_BYTES_MISMATCH",
            )
    expected_digest_roles = set(_ROLE_AUTHORITIES) - {"excluded_synthetic"}
    if set(contract.expected_role_sha256) != expected_digest_roles:
        raise _fatal(
            "archive.member_roles",
            (
                "role-digest contract must contain exactly the seven "
                "relevant semantic roles"
            ),
            "ROLE_DIGEST_CONTRACT_INVALID",
        )
    for role, expected in contract.expected_role_sha256.items():
        if role not in role_infos:
            raise _fatal(
                f"archive.member_roles.{role}",
                "digest contract names an unknown role",
                "ROLE_DIGEST_MISMATCH",
            )
        observed = _role_content_sha256(
            role_infos[role],
            member_sha256,
        )
        if observed != expected:
            raise _fatal(
                f"archive.member_roles.{role}.content_inventory_sha256",
                f"expected {expected}, observed {observed}",
                "ROLE_DIGEST_MISMATCH",
            )

    return _SugarArchivePreflight(
        archive_path=archive_path,
        archive_md5=archive_md5,
        archive_sha256=archive_sha256,
        central_directory_inventory_sha256=inventory_sha256,
        central_directory_safety=safety,
        infos=infos,
        role_infos=MappingProxyType(role_infos),
        member_sha256=MappingProxyType(member_sha256),
    )


def _parse_source_integer(value: str, *, path: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise _fatal(
            path,
            f"expected a nonnegative base-10 integer, observed {value!r}",
            "TARGET_VALUE_INVALID",
        )
    return int(value)


def _parse_recipes(
    archive: zipfile.ZipFile,
    target_info: zipfile.ZipInfo,
    expected_sha256: str,
) -> Mapping[str, _SugarRecipe]:
    try:
        payload = archive.read(target_info)
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise _fatal(
                f"archive.members.{target_info.filename}.sha256",
                f"expected {expected_sha256}, observed {observed_sha256}",
                "MEMBER_HASH_MISMATCH",
            )
        text = payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except SugarMixturesValidationError:
        raise
    except (UnicodeDecodeError, csv.Error) as error:
        raise _fatal(
            "target_table",
            str(error),
            "TARGET_VALUE_INVALID",
        ) from error
    if not rows or tuple(rows[0]) != TARGET_HEADER:
        raise _fatal(
            "target_table.header",
            f"expected {TARGET_HEADER!r}",
            "TARGET_HEADER_INVALID",
        )

    recipes: dict[str, _SugarRecipe] = {}
    for index, row in enumerate(rows[1:], start=1):
        path = f"target_table.rows[{index}]"
        if len(row) != len(TARGET_HEADER):
            raise _fatal(
                path,
                "target row must contain exactly 11 columns",
                "TARGET_ROW_WIDTH_INVALID",
            )
        well_key, samp_text, source_row, column_text, plate_text = row[:5]
        if (
            re.fullmatch(r"[A-H]", source_row) is None
            or re.fullmatch(r"[1-9]|1[0-2]", column_text) is None
            or re.fullmatch(r"[1-3]", plate_text) is None
        ):
            raise _fatal(
                path,
                "row, column, or plate is outside the source contract",
                "RECIPE_IDENTITY_INVALID",
            )
        source_column = int(column_text)
        source_plate = int(plate_text)
        expected_well_key = f"{source_row}{source_column}_{source_plate}"
        source_samp = _parse_source_integer(
            samp_text,
            path=f"{path}.Samp",
        )
        expected_samp = 12 * (ord(source_row) - ord("A")) + source_column
        if well_key != expected_well_key or source_samp != expected_samp:
            raise _fatal(
                path,
                "well key and source Samp must match row/column/plate",
                "RECIPE_IDENTITY_INVALID",
            )
        if well_key in recipes:
            raise _fatal(
                f"{path}.Well",
                f"duplicate well {well_key}",
                "TARGET_WELL_DUPLICATE",
            )
        volume_values = tuple(
            _parse_source_integer(
                value,
                path=f"{path}.{TARGET_HEADER[column]}",
            )
            for column, value in enumerate(row[5:], start=5)
        )
        component_volumes = dict(zip(COMPONENTS, volume_values[:5], strict=True))
        total_volume = volume_values[5]
        if total_volume != 375 or sum(component_volumes.values()) != total_volume:
            raise _fatal(
                path,
                "component volumes must sum to the fixed 375 uL total",
                "TARGET_VOLUME_INVALID",
            )
        if any(
            component_volumes[component] not in ALLOWED_SUGAR_VOLUMES_UL
            for component in TARGET_COMPONENTS
        ):
            raise _fatal(
                path,
                "sugar volumes must use only the approved nominal levels",
                "RECIPE_LEVEL_INVALID",
            )
        recipes[well_key] = _SugarRecipe(
            well_key=well_key,
            source_samp=source_samp,
            component_volumes_ul=MappingProxyType(component_volumes),
            total_volume_ul=total_volume,
            component_fractions=MappingProxyType(
                {
                    component: volume / total_volume
                    for component, volume in component_volumes.items()
                }
            ),
            concentrations_mol_l=MappingProxyType(
                {
                    f"{component}_nominal_mol_l": (
                        component_volumes[component] / total_volume
                    )
                    for component in TARGET_COMPONENTS
                }
            ),
        )
    target_vectors = Counter(
        tuple(
            recipe.component_volumes_ul[component]
            for component in TARGET_COMPONENTS
        )
        for recipe in recipes.values()
    )
    if any(count != 1 for count in target_vectors.values()):
        raise _fatal(
            "target_table.rows",
            "four-sugar target vectors must be unique by well",
            "RECIPE_TARGET_DUPLICATE",
        )
    return MappingProxyType(recipes)


def _parse_canonical_identity(
    info: zipfile.ZipInfo,
    condition: str,
) -> Mapping[str, int | str]:
    basename = PurePosixPath(info.filename).name
    pattern = (
        _HIGH_BASENAME_PATTERN
        if condition == "high_snr"
        else _LOW_BASENAME_PATTERN
    )
    match = pattern.fullmatch(basename)
    if match is None:
        raise _fatal(
            f"archive.members.{info.filename}",
            "canonical basename does not match its condition grammar",
            "CANONICAL_FILENAME_INVALID",
        )
    values = {
        name: int(value) if name != "row" else value
        for name, value in match.groupdict().items()
    }
    source_row = str(values["row"])
    source_column = int(values["column"])
    source_plate = int(values["plate"])
    source_samp = int(values["samp"])
    source_round = int(values["round"])
    source_measurement = int(values["measurement"])
    source_repetition = int(values["repetition"])
    if not 1 <= source_column <= 12:
        raise _fatal(
            f"archive.members.{info.filename}",
            "source column must be 1..12",
            "CANONICAL_IDENTITY_INVALID",
        )
    if source_samp != 12 * (ord(source_row) - ord("A")) + source_column:
        raise _fatal(
            f"archive.members.{info.filename}",
            "source Samp does not match row-major row/column identity",
            "CANONICAL_IDENTITY_INVALID",
        )
    round_limit = 2 if condition == "high_snr" else 8
    if (
        not 1 <= source_round <= round_limit
        or source_measurement != 1
        or not 1 <= source_repetition <= 4
    ):
        raise _fatal(
            f"archive.members.{info.filename}",
            "round, measurement, or repetition is outside the source contract",
            "CANONICAL_IDENTITY_INVALID",
        )
    prefix = (
        "Sugar_Concentration_Test"
        if condition == "high_snr"
        else "Sugar_Concentration_Test_Fast"
    )
    expected_basename = (
        f"{prefix}_{source_samp}_{source_row}{source_column}_"
        f"{source_plate}_RD{source_round}_M{source_measurement}_"
        f"R{source_repetition}.csv"
    )
    if basename != expected_basename:
        raise _fatal(
            f"archive.members.{info.filename}",
            (
                "canonical basename is not the exact reversible rendering "
                f"{expected_basename!r}"
            ),
            "CANONICAL_IDENTITY_INVALID",
        )
    well_key = f"{source_row}{source_column}_{source_plate}"
    sample_id = (
        f"sugar-well-{source_row.lower()}{source_column:02d}"
        f"-p{source_plate:02d}"
    )
    record_id = (
        f"{condition}-s{source_samp:03d}-"
        f"{source_row.lower()}{source_column:02d}-p{source_plate:02d}"
        f"-r{source_round:02d}-m{source_measurement:02d}"
        f"-rep{source_repetition:02d}"
    )
    return MappingProxyType(
        {
            "source_basename": basename,
            "condition": condition,
            "source_samp": source_samp,
            "source_row": source_row,
            "source_column": source_column,
            "source_plate": source_plate,
            "source_round": source_round,
            "source_measurement": source_measurement,
            "source_repetition": source_repetition,
            "well_key": well_key,
            "sample_id": sample_id,
            "acquisition_id": record_id,
            "record_id": record_id,
        }
    )


def _axis_value_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _stored_axis_id(
    domain: bytes,
    values: np.ndarray,
    dtype: str,
) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256(domain)
    digest.update(struct.pack("<Q", canonical.size))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _auxiliary_axis_set_id(
    core_axis_id: str,
    pixel_axis_id: str,
    wavelength_axis_id: str,
) -> str:
    digest = hashlib.sha256(b"rpe-sugar-aux-axis-set-v1\0")
    for value in (core_axis_id, pixel_axis_id, wavelength_axis_id):
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _identity_map_sha256(
    members: Sequence[_SugarAcceptedMember],
) -> str:
    digest = hashlib.sha256(b"rpe-sugar-identity-map-v1\0")
    for member in members:
        for value in (
            member.record_id,
            member.sample_id,
            member.acquisition_id,
            member.source_member,
        ):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _evidence_snapshots(
    raw_root: Path,
    contract: _SugarSourceContract,
) -> tuple[_SugarEvidenceSnapshot, ...]:
    evidence_root = raw_root.parent.parent
    snapshots = []
    role_by_path = {
        "evidence/zenodo/10779223.json": "zenodo_release_evidence",
        "receipts/sugar_mixtures_high_snr.json": "loader_comparison_receipt",
        "receipts/sugar_mixtures_low_snr.json": "loader_comparison_receipt",
    }
    if set(contract.expected_evidence_snapshots) != set(role_by_path):
        raise _fatal(
            "evidence_snapshots",
            "evidence contract must contain exactly the three retained paths",
            "EVIDENCE_CONTRACT_INVALID",
        )
    for relative_path in sorted(
        contract.expected_evidence_snapshots,
        key=lambda value: value.encode("utf-8"),
    ):
        path = evidence_root / relative_path
        if not path.is_file():
            raise _fatal(
                f"evidence_snapshots.{relative_path}",
                "required retained evidence file is missing",
                "EVIDENCE_MISSING",
            )
        expected_bytes, expected_sha256 = (
            contract.expected_evidence_snapshots[relative_path]
        )
        try:
            observed_bytes = path.stat().st_size
        except OSError as error:
            raise _fatal(
                f"evidence_snapshots.{relative_path}",
                str(error),
                "EVIDENCE_MISSING",
            ) from error
        if observed_bytes != expected_bytes:
            raise _fatal(
                f"evidence_snapshots.{relative_path}.bytes",
                f"expected {expected_bytes}, observed {observed_bytes}",
                "EVIDENCE_BYTES_MISMATCH",
            )
        _, observed_sha256 = _hash_file_md5_sha256(path)
        if observed_sha256 != expected_sha256:
            raise _fatal(
                f"evidence_snapshots.{relative_path}.sha256",
                f"expected {expected_sha256}, observed {observed_sha256}",
                "EVIDENCE_SHA256_MISMATCH",
            )
        snapshots.append(
            _SugarEvidenceSnapshot(
                source_artifact=relative_path,
                role=role_by_path[relative_path],
                bytes=observed_bytes,
                sha256=observed_sha256,
            )
        )
    return tuple(snapshots)


def _member_roles_summary(
    preflight: _SugarArchivePreflight,
) -> Mapping[str, Mapping[str, int | str]]:
    summaries = {}
    for role in sorted(
        (
            role
            for role in preflight.role_infos
            if role != "excluded_synthetic"
        ),
        key=lambda value: value.encode("utf-8"),
    ):
        infos = preflight.role_infos[role]
        summaries[role] = MappingProxyType(
            {
                "role": role,
                "member_count": len(infos),
                "uncompressed_bytes": sum(info.file_size for info in infos),
                "content_inventory_sha256": _role_content_sha256(
                    infos,
                    preflight.member_sha256,
                ),
                "conversion_authority": _ROLE_AUTHORITIES[role],
            }
        )
    return MappingProxyType(summaries)


def _numeric_point_count(payload: bytes) -> int:
    try:
        rows = list(
            csv.reader(
                io.StringIO(payload.decode("utf-8"), newline=""),
                strict=True,
            )
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise _fatal(
            "source_member",
            str(error),
            "CSV_INVALID",
        ) from error
    return sum(
        len(row) == len(CANONICAL_HEADER)
        and row[4] == ""
        and any(value != "" for value in row[:4])
        for row in rows[1:]
    )


def _reparse_verified_member(
    raw_root: Path,
    member: _SugarAcceptedMember,
    *,
    archive: zipfile.ZipFile | None = None,
    info: zipfile.ZipInfo | None = None,
    expected_points: int | None = None,
) -> _ParsedSugarMember:
    if archive is None:
        archive_path = (
            Path(raw_root).resolve()
            / _PRODUCTION_SOURCE_CONTRACT.archive_name
        )
        try:
            with zipfile.ZipFile(archive_path) as opened_archive:
                return _reparse_verified_member(
                    raw_root,
                    member,
                    archive=opened_archive,
                    info=info,
                    expected_points=expected_points,
                )
        except SugarMixturesValidationError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise _fatal(
                f"archive.members.{member.source_member}",
                str(error),
                "MEMBER_BINDING_MISMATCH",
            ) from error

    try:
        if info is None:
            matches = tuple(
                current_info
                for current_info in archive.infolist()
                if current_info.filename == member.source_member
            )
            if len(matches) != 1:
                raise _fatal(
                    f"archive.members.{member.source_member}",
                    f"expected one member binding, observed {len(matches)}",
                    "MEMBER_BINDING_MISMATCH",
                )
            info = matches[0]
        elif info.filename != member.source_member:
            raise _fatal(
                f"archive.members.{member.source_member}",
                f"bound ZipInfo names {info.filename!r}",
                "MEMBER_BINDING_MISMATCH",
            )
        payload = archive.read(info)
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        raise _fatal(
            f"archive.members.{member.source_member}",
            str(error),
            "MEMBER_BINDING_MISMATCH",
        ) from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != member.sha256:
        raise _fatal(
            f"archive.members.{member.source_member}.sha256",
            f"expected {member.sha256}, observed {observed_sha256}",
            "MEMBER_HASH_MISMATCH",
        )
    if info.file_size != member.bytes or info.CRC != member.crc32:
        raise _fatal(
            f"archive.members.{member.source_member}",
            "member byte or CRC binding changed after inspection",
            "MEMBER_BINDING_MISMATCH",
        )
    return _parse_canonical_member_bytes(
        payload,
        source_member=member.source_member,
        expected_points=(
            _numeric_point_count(payload)
            if expected_points is None
            else expected_points
        ),
    )


def _inspect_sugar_mixtures_source(
    raw_root: Path,
    contract: _SugarSourceContract,
) -> SugarMixturesInspection:
    raw_root = Path(raw_root).resolve()
    preflight = _preflight_archive(raw_root, contract)
    expected_points = int(contract.expected_counts["points_per_record"])

    canonical_entries = []
    identity_keys = set()
    for role, condition in (
        ("canonical_high", "high_snr"),
        ("canonical_low", "low_snr"),
    ):
        for info in preflight.role_infos[role]:
            identity = _parse_canonical_identity(info, condition)
            record_id = str(identity["record_id"])
            if record_id in identity_keys:
                raise _fatal(
                    f"archive.members.{info.filename}",
                    f"duplicate acquisition identity {record_id}",
                    "CANONICAL_IDENTITY_DUPLICATE",
                )
            identity_keys.add(record_id)
            canonical_entries.append((info, identity))

    target_infos = preflight.role_infos["target"]
    if len(target_infos) != 1:
        raise _fatal(
            "archive.member_roles.target",
            "exactly one target table is required",
            "ROLE_COUNT_MISMATCH",
        )
    try:
        with zipfile.ZipFile(preflight.archive_path) as archive:
            recipes = _parse_recipes(
                archive,
                target_infos[0],
                preflight.member_sha256[target_infos[0].filename],
            )
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "target_table",
            str(error),
            "TARGET_VALUE_INVALID",
        ) from error

    expected_recipe_count = int(contract.expected_counts["sample_count"])
    if len(recipes) != expected_recipe_count:
        raise _fatal(
            "target_table.rows",
            f"expected {expected_recipe_count}, observed {len(recipes)}",
            "TARGET_WELL_COUNT_MISMATCH",
        )

    accepted_members = []
    first_pixel = None
    first_wavelength = None
    first_wavenumber = None
    canonical_entries.sort(
        key=lambda value: str(value[1]["record_id"]).encode("utf-8")
    )
    try:
        with zipfile.ZipFile(preflight.archive_path) as archive:
            for info, identity in canonical_entries:
                well_key = str(identity["well_key"])
                if well_key not in recipes:
                    raise _fatal(
                        f"archive.members.{info.filename}.well_key",
                        f"no recipe exists for {well_key}",
                        "RECIPE_MISSING",
                    )
                recipe = recipes[well_key]
                if int(identity["source_samp"]) != recipe.source_samp:
                    raise _fatal(
                        f"archive.members.{info.filename}.source_samp",
                        "canonical identity disagrees with target recipe",
                        "RECIPE_IDENTITY_INVALID",
                    )
                payload = archive.read(info)
                observed_sha256 = hashlib.sha256(payload).hexdigest()
                expected_sha256 = preflight.member_sha256[info.filename]
                if observed_sha256 != expected_sha256:
                    raise _fatal(
                        f"archive.members.{info.filename}.sha256",
                        "member changed after archive preflight",
                        "MEMBER_HASH_MISMATCH",
                    )
                parsed = _parse_canonical_member_bytes(
                    payload,
                    source_member=info.filename,
                    expected_points=expected_points,
                )
                expected_pixel = np.arange(expected_points, dtype=np.float64)
                if not np.array_equal(parsed.pixel, expected_pixel):
                    raise _fatal(
                        f"archive.members.{info.filename}.pixel",
                        "Pixel must equal the exact zero-based coordinate",
                        "PIXEL_AXIS_INVALID",
                    )
                if first_pixel is None:
                    first_pixel = np.array(parsed.pixel, copy=True)
                    first_wavelength = np.array(
                        parsed.wavelength_nm,
                        copy=True,
                    )
                    first_wavenumber = np.array(
                        parsed.wavenumber_cm1,
                        copy=True,
                    )
                elif (
                    not np.array_equal(parsed.pixel, first_pixel)
                    or not np.array_equal(parsed.wavelength_nm, first_wavelength)
                    or not np.array_equal(parsed.wavenumber_cm1, first_wavenumber)
                ):
                    raise _fatal(
                        f"archive.members.{info.filename}.axes",
                        "canonical records must share exact source axes",
                        "AXIS_DRIFT",
                    )

                pure_components = tuple(
                    component
                    for component, volume
                    in recipe.component_volumes_ul.items()
                    if volume == recipe.total_volume_ul
                )
                if len(pure_components) > 1:
                    raise _fatal(
                        f"target_table.{well_key}",
                        "a recipe cannot contain multiple pure components",
                        "TARGET_VOLUME_INVALID",
                    )
                record_role = (
                    "pure_reference" if pure_components else "mixture"
                )
                accepted_members.append(
                    _SugarAcceptedMember(
                        source_member=info.filename,
                        source_basename=str(identity["source_basename"]),
                        bytes=info.file_size,
                        crc32=info.CRC,
                        sha256=observed_sha256,
                        condition=str(identity["condition"]),
                        source_samp=int(identity["source_samp"]),
                        source_row=str(identity["source_row"]),
                        source_column=int(identity["source_column"]),
                        source_plate=int(identity["source_plate"]),
                        source_round=int(identity["source_round"]),
                        source_measurement=int(
                            identity["source_measurement"]
                        ),
                        source_repetition=int(identity["source_repetition"]),
                        well_key=well_key,
                        sample_id=str(identity["sample_id"]),
                        acquisition_id=str(identity["acquisition_id"]),
                        record_id=str(identity["record_id"]),
                        record_role=record_role,
                        pure_component=(
                            pure_components[0] if pure_components else None
                        ),
                    )
                )
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "archive.canonical_members",
            str(error),
            "ARCHIVE_CRC_FAILURE",
        ) from error

    if first_pixel is None or first_wavelength is None or first_wavenumber is None:
        raise _fatal(
            "archive.canonical_members",
            "at least one canonical member is required",
            "ROLE_COUNT_MISMATCH",
        )

    expected_record_count = int(contract.expected_counts["canonical_records"])
    high_count = sum(member.condition == "high_snr" for member in accepted_members)
    low_count = sum(member.condition == "low_snr" for member in accepted_members)
    sample_ids = {member.sample_id for member in accepted_members}
    if len(accepted_members) != expected_record_count:
        raise _fatal(
            "inspection.record_count",
            f"expected {expected_record_count}, observed {len(accepted_members)}",
            "CANONICAL_COUNT_MISMATCH",
        )
    if high_count != int(contract.expected_counts["high_records"]):
        raise _fatal(
            "inspection.high_records",
            "High canonical count differs from the source contract",
            "CANONICAL_COUNT_MISMATCH",
        )
    if low_count != int(contract.expected_counts["low_records"]):
        raise _fatal(
            "inspection.low_records",
            "Low canonical count differs from the source contract",
            "CANONICAL_COUNT_MISMATCH",
        )
    if len(sample_ids) != expected_recipe_count:
        raise _fatal(
            "inspection.sample_count",
            f"expected {expected_recipe_count}, observed {len(sample_ids)}",
            "SAMPLE_COUNT_MISMATCH",
        )
    well_condition_counts = Counter(
        (member.well_key, member.condition)
        for member in accepted_members
    )
    acquisition_grids: dict[
        tuple[str, str],
        set[tuple[int, int]],
    ] = {}
    for member in accepted_members:
        acquisition_grids.setdefault(
            (member.well_key, member.condition),
            set(),
        ).add(
            (member.source_round, member.source_repetition)
        )
    for condition, contract_name in (
        ("high_snr", "high_records_per_well"),
        ("low_snr", "low_records_per_well"),
    ):
        expected_per_well = contract.expected_counts.get(contract_name)
        if expected_per_well is None:
            continue
        for well_key in recipes:
            observed = well_condition_counts[(well_key, condition)]
            if observed != expected_per_well:
                raise _fatal(
                    f"inspection.wells.{well_key}.{condition}",
                    (
                        f"expected {expected_per_well} records, "
                        f"observed {observed}"
                    ),
                    "CANONICAL_WELL_COUNT_MISMATCH",
                )
        prefix = "high" if condition == "high_snr" else "low"
        round_count = contract.expected_counts.get(
            f"{prefix}_round_count"
        )
        repetitions_per_round = contract.expected_counts.get(
            f"{prefix}_repetitions_per_round"
        )
        if round_count is None or repetitions_per_round is None:
            continue
        expected_grid = {
            (source_round, source_repetition)
            for source_round in range(1, int(round_count) + 1)
            for source_repetition in range(
                1,
                int(repetitions_per_round) + 1,
            )
        }
        for well_key in recipes:
            observed_grid = acquisition_grids.get(
                (well_key, condition),
                set(),
            )
            if observed_grid != expected_grid:
                raise _fatal(
                    f"inspection.wells.{well_key}.{condition}",
                    (
                        f"expected acquisition grid {sorted(expected_grid)}, "
                        f"observed {sorted(observed_grid)}"
                    ),
                    "CANONICAL_ACQUISITION_GRID_MISMATCH",
                )
    canonical_wells = {member.well_key for member in accepted_members}
    missing_wells = set(recipes) - canonical_wells
    if missing_wells:
        raise _fatal(
            "inspection.recipe_coverage",
            f"recipes have no canonical records: {sorted(missing_wells)}",
            "RECIPE_COVERAGE_INVALID",
        )

    core_values = np.ascontiguousarray(first_wavenumber, dtype="<f4")
    pixel_values = np.ascontiguousarray(first_pixel, dtype="<u2")
    if not np.array_equal(
        pixel_values.astype(np.float64),
        first_pixel,
    ):
        raise _fatal(
            "inspection.pixel_axis",
            "Pixel cannot be represented exactly as uint16",
            "PIXEL_AXIS_INVALID",
        )
    wavelength_values = np.ascontiguousarray(first_wavelength, dtype="<f4")
    core_axis_identifier = axis_id(core_values)
    pixel_axis_identifier = _stored_axis_id(
        b"rpe-sugar-pixel-axis-v1\0",
        pixel_values,
        "<u2",
    )
    wavelength_axis_identifier = _stored_axis_id(
        b"rpe-sugar-wavelength-axis-v1\0",
        wavelength_values,
        "<f4",
    )
    auxiliary_set_identifier = _auxiliary_axis_set_id(
        core_axis_identifier,
        pixel_axis_identifier,
        wavelength_axis_identifier,
    )
    observed_axis_contract = {
        "core_axis_id": core_axis_identifier,
        "pixel_axis_id": pixel_axis_identifier,
        "wavelength_axis_id": wavelength_axis_identifier,
        "auxiliary_axis_set_id": auxiliary_set_identifier,
        "pixel_float64_sha256": _axis_value_sha256(first_pixel, "<f8"),
        "wavelength_float64_sha256": _axis_value_sha256(
            first_wavelength,
            "<f8",
        ),
        "wavenumber_float64_sha256": _axis_value_sha256(
            first_wavenumber,
            "<f8",
        ),
    }
    for name, observed in observed_axis_contract.items():
        expected = contract.expected_counts.get(name)
        if expected is not None and observed != expected:
            raise _fatal(
                f"inspection.{name}",
                f"expected {expected}, observed {observed}",
                "AXIS_CONTRACT_MISMATCH",
            )

    view_record_ids = {}
    view_source_members = {}
    for view_id in VIEW_IDS:
        condition = (
            "high_snr"
            if view_id.startswith("high_snr")
            else "low_snr"
            if view_id.startswith("low_snr")
            else None
        )
        role = (
            "pure_reference"
            if view_id.endswith("pure_reference")
            or view_id == "pure_reference"
            else "mixture"
            if view_id.endswith("no_refs") or view_id == "no_refs"
            else None
        )
        selected = tuple(
            member
            for member in accepted_members
            if (condition is None or member.condition == condition)
            and (role is None or member.record_role == role)
        )
        view_record_ids[view_id] = tuple(
            member.record_id
            for member in selected
        )
        view_source_members[view_id] = tuple(
            sorted(
                (member.source_member for member in selected),
                key=lambda value: value.encode("utf-8"),
            )
        )

    endmember_record_ids = {}
    endmember_source_members = {}
    for condition in ("high_snr", "low_snr"):
        for component in COMPONENTS:
            key = f"{condition}/{component}"
            selected = tuple(
                member
                for member in accepted_members
                if member.condition == condition
                and member.pure_component == component
            )
            if not selected:
                raise _fatal(
                    f"inspection.endmembers.{key}",
                    "every condition/component needs canonical constituents",
                    "ENDMEMBER_CONSTITUENTS_MISSING",
                )
            endmember_record_ids[key] = tuple(
                member.record_id
                for member in selected
            )
            endmember_source_members[key] = tuple(
                sorted(
                    (member.source_member for member in selected),
                    key=lambda value: value.encode("utf-8"),
                )
            )

    relevant_roles = (
        "target",
        "oracle",
        "prepared",
        "raw_support",
        "experimental_readme",
    )
    evidence_members = tuple(
        _SugarEvidenceMember(
            source_member=info.filename,
            role=role,
            bytes=info.file_size,
            crc32=info.CRC,
            sha256=preflight.member_sha256[info.filename],
        )
        for role in relevant_roles
        for info in preflight.role_infos[role]
    )
    evidence_members = tuple(
        sorted(
            evidence_members,
            key=lambda value: value.source_member.encode("utf-8"),
        )
    )
    snapshots = _evidence_snapshots(raw_root, contract)
    pure_reference_records = sum(
        member.record_role == "pure_reference"
        for member in accepted_members
    )
    identity_map_sha256 = _identity_map_sha256(accepted_members)
    maximum_record_id_bytes = max(
        len(member.record_id.encode("utf-8"))
        for member in accepted_members
    )
    for name, observed in (
        ("identity_map_sha256", identity_map_sha256),
        ("maximum_record_id_bytes", maximum_record_id_bytes),
    ):
        expected = contract.expected_counts.get(name)
        if expected is not None and observed != expected:
            raise _fatal(
                f"inspection.{name}",
                f"expected {expected}, observed {observed}",
                "IDENTITY_CONTRACT_MISMATCH",
            )
    source_statistics = MappingProxyType(
        {
            "archive_md5": preflight.archive_md5,
            "archive_sha256": preflight.archive_sha256,
            "central_directory_inventory_sha256": (
                preflight.central_directory_inventory_sha256
            ),
            "archive_bytes": preflight.archive_path.stat().st_size,
            "archive_member_count": len(preflight.infos),
            "archive_uncompressed_bytes": sum(
                info.file_size for info in preflight.infos
            ),
            "record_count": len(accepted_members),
            "canonical_source_members": len(accepted_members),
            "rejected_canonical_members": 0,
            "sample_count": len(sample_ids),
            "recipe_count": len(recipes),
            "points_per_record": expected_points,
            "high_records": high_count,
            "low_records": low_count,
            "pure_reference_records": pure_reference_records,
            "mixture_records": len(accepted_members) - pure_reference_records,
            "axis_groups": 1,
            "identity_map_sha256": identity_map_sha256,
            "maximum_record_id_bytes": maximum_record_id_bytes,
            "raw_text_inventory_sha256": contract.expected_counts.get(
                "raw_text_inventory_sha256",
                "",
            ),
            "normalized_metadata_sha256": contract.expected_counts.get(
                "normalized_metadata_sha256",
                "",
            ),
            "core_metadata_sha256": contract.expected_counts.get(
                "core_metadata_sha256",
                "",
            ),
            "well_target_contract_sha256": contract.expected_counts.get(
                "well_target_contract_sha256",
                "",
            ),
            "record_target_map_sha256": contract.expected_counts.get(
                "record_target_map_sha256",
                "",
            ),
            "record_preprocessing_map_sha256": contract.expected_counts.get(
                "record_preprocessing_map_sha256",
                "",
            ),
            "preprocessing_evidence_document_sha256": (
                contract.expected_counts.get(
                    "preprocessing_evidence_document_sha256",
                    "",
                )
            ),
            "intensity_float32_max_abs_error": contract.expected_counts.get(
                "intensity_float32_max_abs_error",
                0.0,
            ),
            "axis_float32_max_abs_error_cm1": contract.expected_counts.get(
                "axis_float32_max_abs_error_cm1",
                0.0,
            ),
            "wavelength_float32_max_abs_error_nm": (
                contract.expected_counts.get(
                    "wavelength_float32_max_abs_error_nm",
                    0.0,
                )
            ),
            "relevant_regular_members": (
                len(accepted_members) + len(evidence_members)
            ),
            "relevant_evidence_members": len(evidence_members),
            "excluded_synthetic_regular_members": len(
                preflight.role_infos["excluded_synthetic"]
            ),
            "excluded_synthetic_uncompressed_bytes": sum(
                info.file_size
                for info in preflight.role_infos["excluded_synthetic"]
            ),
            **observed_axis_contract,
        }
    )

    for name in (
        "pure_reference_records",
        "mixture_records",
        "relevant_regular_members",
        "relevant_evidence_members",
        "excluded_synthetic_regular_members",
    ):
        expected = contract.expected_counts.get(name)
        if expected is not None and source_statistics[name] != expected:
            raise _fatal(
                f"inspection.source_statistics.{name}",
                f"expected {expected}, observed {source_statistics[name]}",
                "SOURCE_STATISTICS_MISMATCH",
            )

    return SugarMixturesInspection(
        raw_root=raw_root,
        archive_md5=preflight.archive_md5,
        archive_sha256=preflight.archive_sha256,
        central_directory_inventory_sha256=(
            preflight.central_directory_inventory_sha256
        ),
        central_directory_safety=preflight.central_directory_safety,
        member_roles=_member_roles_summary(preflight),
        accepted_members=tuple(accepted_members),
        relevant_evidence_members=evidence_members,
        evidence_snapshots=snapshots,
        recipes=recipes,
        view_record_ids=MappingProxyType(
            {
                key: tuple(value)
                for key, value in view_record_ids.items()
            }
        ),
        view_source_members=MappingProxyType(
            {
                key: tuple(value)
                for key, value in view_source_members.items()
            }
        ),
        endmember_constituent_record_ids=MappingProxyType(
            {
                key: tuple(value)
                for key, value in endmember_record_ids.items()
            }
        ),
        endmember_constituent_source_members=MappingProxyType(
            {
                key: tuple(value)
                for key, value in endmember_source_members.items()
            }
        ),
        source_statistics=source_statistics,
        core_axis_id=core_axis_identifier,
        pixel_axis_id=pixel_axis_identifier,
        wavelength_axis_id=wavelength_axis_identifier,
        auxiliary_axis_set_id=auxiliary_set_identifier,
    )
