from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "phase05-d4-protocol-v1"
EXPERIMENT_ID = "d4_sugar_low_snr_pls2_sg11"
CONFIG_BYTES = 4799
CONFIG_SHA256 = (
    "69a5dba7ab45cbc421f988439e4f3b3aa9b99505d6ff13cbb3d3b89389d071c8"
)
AUDIT_BYTES = 7421
AUDIT_SHA256 = (
    "7d6ea22b7a96a5e7546d6fd6bc51e68024b59a5214888b0d83c61908aafbd506"
)
AUDIT_RELATIVE_PATH = "reports/phase05/d4_step01_protocol_audit.json"
ARCHIVE_BYTES = 5_310_910_565
ARCHIVE_SHA256 = (
    "0924efd1171416efe14a31108f6ac7a00124d1bfb2a06c4fb22045cc3f2a265b"
)
TARGET_MEMBER = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
    "Sugar_Concentrations.csv"
)
TARGET_MEMBER_BYTES = 6174
TARGET_MEMBER_SHA256 = (
    "f39b27c27c7c660b44f925fc4da8fe697a4388acfa76f33de02a76b2d56be3d9"
)
README_MEMBER = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/README.md"
)
README_BYTES = 8873
README_SHA256 = (
    "72cf1320baedd78e4b51bd9f08ff621f3fca933ceacb87a6a7c27af9fdad7af1"
)
LOW_PREFIX = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
    "Sugar_Concentration_Test_Fast/"
)
TARGET_HEADER = (
    "Cell Number",
    "Sucrose [ul]",
    "Fructose [ul]",
    "Maltose [ul]",
    "Glucose [ul]",
    "Water [ul]",
    "Total Volume [ul]",
)
SPECTRUM_HEADER = ("Pixel", "wl", "cm-1", "Intensity", "Metadata")
TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
PURE_REFERENCE_WELLS = ("E1_3", "E2_3", "E3_3", "E4_3", "E5_3")
BLANK_WELL = "E1_3"
EXPECTED_LOW_MEMBERS = 7840
EXPECTED_MIXTURE_RECORDS = 7680
EXPECTED_BLANK_RECORDS = 32
EXPECTED_MIXTURE_WELLS = 240
EXPECTED_POINTS = 2000
RECORDS_PER_WELL = 32
AXIS_SHA256 = (
    "9b0b88641a767abc74439e21539f7c41366adcf41c9bc3ab60d6de91021431d7"
)
MIXTURE_RECORD_IDS_SHA256 = (
    "9b102daf59be106e9fe46ef092373ae5881403ee45b3a033e2ec5c732365f76d"
)
MIXTURE_SOURCE_MEMBERS_SHA256 = (
    "7cdcb2f7fe7268891f5cfb6f93dd99c139fc058d6ee61a7f4ed5c055040c466d"
)
BLANK_RECORD_IDS_SHA256 = (
    "5bda1918b3088228a57c4d97281b285a9dd5debc43db0f76a094620a6ff28ecf"
)
BLANK_SOURCE_MEMBERS_SHA256 = (
    "0876fd82b2e739d7b23c03f5351d399a72cf436dca3e0022c6d0feb175c88648"
)
FOLD_RECORD_IDS_SHA256 = (
    "6bad59e135ea206d0c98ae602a89b9c43fa5dd3da015342d504210796bc0b19d",
    "1df98cb38c2ab7495b7682d0cfc58fb8ccadba01696653c60a8bbe518be944f2",
    "d48c935eafe9ea833b562bfd3309215e4bcd2bfdc249d38b6e89c6675825e7b3",
    "8e2e9c243a257d299010ca9a9c4971acab9f86b2dc87fb07ab94532679223daf",
    "fd5bf3debe8ca65a80482547a57ff5d2bb7e4d560cb6c56c4b7635366f4842a0",
)
FOLD_SOURCE_MEMBERS_SHA256 = (
    "9afd31915298bff284124bf3ee0ff0481863a8c55e2b13b9bea99ad192134087",
    "aca11230ec6b1753dc97d522a188ce17d963b5d14049d696c6b170c9acb6fbad",
    "244b2ff63d67ca2169e72bdb7b1cadcc8a5e6ecb2541d9f61d2fa1351761411c",
    "59b55ab2b2b18b15db06f947da9f725b48e003594e79d70f1028b497285250b7",
    "1082e0552f64f77be6209b1867074f26f7817ebf1547559a423d163265b66fc6",
)
ROOT = Path(__file__).resolve().parents[2]

_LOW_BASENAME_PATTERN = re.compile(
    r"^Sugar_Concentration_Test_Fast_"
    r"(?P<samp>\d+)_"
    r"(?P<row>[A-H])(?P<column>\d{1,2})_"
    r"(?P<plate>[1-3])_"
    r"RD(?P<round>\d+)_"
    r"M(?P<measurement>\d+)_"
    r"R(?P<repetition>\d+)\.csv$"
)


class D4LoaderValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class D4ProtocolConfig:
    path: Path
    sha256: str
    byte_count: int
    audit_sha256: str
    expected_mixture_records: int
    expected_blank_records: int
    target_count: int
    fold_count: int
    records_per_well: int
    raman_data_loader_forbidden: bool


@dataclass(frozen=True)
class D4WellSplit:
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_folds: tuple[int, ...]
    validation_fold: int
    test_fold: int


@dataclass(frozen=True)
class D4SugarCohort:
    protocol_config_sha256: str
    intensity: np.ndarray
    targets: np.ndarray
    wavenumber: np.ndarray
    target_names: tuple[str, ...]
    record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    source_members: tuple[str, ...]
    rounds: np.ndarray
    repetitions: np.ndarray
    blank_intensity: np.ndarray
    blank_targets: np.ndarray
    blank_record_ids: tuple[str, ...]
    blank_well_ids: tuple[str, ...]
    blank_source_members: tuple[str, ...]
    folds: tuple[np.ndarray, ...]
    splits: tuple[D4WellSplit, ...]


@dataclass(frozen=True)
class _LowMember:
    source_member: str
    record_id: str
    well_id: str
    source_round: int
    source_repetition: int


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


def _reject_nonfinite(value: str) -> None:
    raise D4LoaderValidationError(
        "nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _canonical_order_digest(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D4LoaderValidationError(path, "must be an object")
    return value


def load_d4_protocol_config(path: Path) -> D4ProtocolConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except D4LoaderValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D4LoaderValidationError(path.name or "config", str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise D4LoaderValidationError(
            "config noncanonical",
            "must use canonical JSON",
        )
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES or digest != CONFIG_SHA256:
        raise D4LoaderValidationError(
            "config identity",
            "bytes or SHA256 mismatch",
        )
    root = _object("config", document)
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("experiment_id") != EXPERIMENT_ID
    ):
        raise D4LoaderValidationError(
            "config contract",
            "schema or experiment identity mismatch",
        )
    source = _object("source", root.get("source"))
    audit = _object("protocol audit", source.get("protocol_audit"))
    return D4ProtocolConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        audit_sha256=str(audit["sha256"]),
        expected_mixture_records=EXPECTED_MIXTURE_RECORDS,
        expected_blank_records=EXPECTED_BLANK_RECORDS,
        target_count=len(TARGET_NAMES),
        fold_count=5,
        records_per_well=RECORDS_PER_WELL,
        raman_data_loader_forbidden=bool(
            source["raman_data_loader_forbidden"]
        ),
    )


def _load_protocol_audit() -> Mapping[str, object]:
    path = ROOT / AUDIT_RELATIVE_PATH
    raw = path.read_bytes()
    document = json.loads(raw, parse_constant=_reject_nonfinite)
    if raw != _canonical_json_bytes(document):
        raise D4LoaderValidationError(
            "protocol audit",
            "must use canonical JSON",
        )
    if len(raw) != AUDIT_BYTES or hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise D4LoaderValidationError(
            "protocol audit identity",
            "bytes or SHA256 mismatch",
        )
    return _object("protocol audit", document)


def _validate_archive(path: Path) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != ARCHIVE_BYTES
        or _sha256_file(path) != ARCHIVE_SHA256
    ):
        raise D4LoaderValidationError(
            "source archive identity",
            "path, bytes, or SHA256 mismatch",
        )


def _parse_target_table(payload: bytes) -> tuple[
    dict[str, np.ndarray],
    list[str],
]:
    if (
        len(payload) != TARGET_MEMBER_BYTES
        or hashlib.sha256(payload).hexdigest() != TARGET_MEMBER_SHA256
    ):
        raise D4LoaderValidationError(
            "target member identity",
            "bytes or SHA256 mismatch",
        )
    try:
        rows = list(
            csv.DictReader(
                io.StringIO(payload.decode("utf-8-sig"), newline="")
            )
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise D4LoaderValidationError("target table", str(error)) from error
    if not rows or tuple(rows[0]) != TARGET_HEADER:
        raise D4LoaderValidationError(
            "target table header",
            f"must equal {TARGET_HEADER!r}",
        )
    targets = {}
    order = []
    for row in rows:
        well_id = row.get("Cell Number")
        if not isinstance(well_id, str) or well_id in targets:
            raise D4LoaderValidationError(
                "target well",
                "must be unique and nonempty",
            )
        try:
            volumes = [
                int(row[name])
                for name in TARGET_HEADER[1:6]
            ]
            total = int(row[TARGET_HEADER[6]])
        except (TypeError, ValueError) as error:
            raise D4LoaderValidationError(
                f"target {well_id}",
                "volumes must be integers",
            ) from error
        if (
            total != 375
            or sum(volumes) != total
            or any(value < 0 for value in volumes)
        ):
            raise D4LoaderValidationError(
                f"target {well_id}",
                "component volumes must sum to 375 uL",
            )
        targets[well_id] = np.asarray(
            [value / 375.0 for value in volumes[:4]],
            dtype="<f8",
        )
        order.append(well_id)
    if len(targets) != 245 or len(set(order)) != 245:
        raise D4LoaderValidationError(
            "target rows",
            "must contain exactly 245 unique wells",
        )
    return targets, order


def _parse_low_identity(source_member: str) -> _LowMember:
    if not source_member.startswith(LOW_PREFIX):
        raise D4LoaderValidationError(
            "low source member",
            "path prefix mismatch",
        )
    basename = Path(source_member).name
    match = _LOW_BASENAME_PATTERN.fullmatch(basename)
    if match is None:
        raise D4LoaderValidationError(
            source_member,
            "basename grammar mismatch",
        )
    values = {
        key: int(value) if key != "row" else value
        for key, value in match.groupdict().items()
    }
    column = int(values["column"])
    row = str(values["row"])
    samp = int(values["samp"])
    plate = int(values["plate"])
    source_round = int(values["round"])
    measurement = int(values["measurement"])
    repetition = int(values["repetition"])
    if (
        not 1 <= column <= 12
        or samp != 12 * (ord(row) - ord("A")) + column
        or not 1 <= source_round <= 8
        or measurement != 1
        or not 1 <= repetition <= 4
    ):
        raise D4LoaderValidationError(
            source_member,
            "identity fields are outside the Low-SNR contract",
        )
    well_id = f"{row}{column}_{plate}"
    record_id = (
        f"low_snr-s{samp:03d}-{row.lower()}{column:02d}-p{plate:02d}"
        f"-r{source_round:02d}-m{measurement:02d}-rep{repetition:02d}"
    )
    return _LowMember(
        source_member=source_member,
        record_id=record_id,
        well_id=well_id,
        source_round=source_round,
        source_repetition=repetition,
    )


def _parse_spectrum(
    payload: bytes,
    *,
    source_member: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise D4LoaderValidationError(source_member, str(error)) from error
    if (
        len(lines) != EXPECTED_POINTS + 2
        or tuple(lines[0].split(",")) != SPECTRUM_HEADER
    ):
        raise D4LoaderValidationError(
            source_member,
            "header or row count mismatch",
        )
    numeric_lines = lines[1 : EXPECTED_POINTS + 1]
    if any(
        not line.endswith(",") or line.count(",") != 4
        for line in numeric_lines
    ):
        raise D4LoaderValidationError(
            source_member,
            "numeric row grammar mismatch",
        )
    flattened = ",".join(line[:-1] for line in numeric_lines)
    values = np.fromstring(flattened, sep=",", dtype=np.float64)
    if values.size != EXPECTED_POINTS * 4:
        raise D4LoaderValidationError(
            source_member,
            "numeric cell count mismatch",
        )
    matrix = values.reshape(EXPECTED_POINTS, 4)
    if (
        not np.isfinite(matrix).all()
        or not np.array_equal(
            matrix[:, 0],
            np.arange(EXPECTED_POINTS, dtype=np.float64),
        )
        or not np.all(np.diff(matrix[:, 2]) > 0)
    ):
        raise D4LoaderValidationError(
            source_member,
            "numeric arrays are invalid",
        )
    metadata_rows = list(csv.reader([lines[-1]]))
    if (
        len(metadata_rows) != 1
        or len(metadata_rows[0]) != 5
        or metadata_rows[0][:4] != ["", "", "", ""]
    ):
        raise D4LoaderValidationError(
            source_member,
            "metadata row grammar mismatch",
        )
    metadata = json.loads(
        metadata_rows[0][4],
        parse_constant=lambda value: None,
    )
    if (
        metadata.get("Spectrometer integration time [s]") != 0.5
        or metadata.get("Laser power [mW]") != 30.3
        or metadata.get("Excitation wavelength [nm]") != 785
        or metadata.get("Spectrometer number of accumulations") != 1
    ):
        raise D4LoaderValidationError(
            source_member,
            "metadata does not equal the Low-SNR acquisition contract",
        )
    return (
        np.asarray(matrix[:, 2], dtype="<f4"),
        np.asarray(matrix[:, 3], dtype="<f4"),
    )


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def load_d4_sugar_cohort(
    config_path: Path,
    archive_path: Path,
) -> D4SugarCohort:
    config = load_d4_protocol_config(config_path)
    audit = _load_protocol_audit()
    archive_path = Path(archive_path)
    _validate_archive(archive_path)
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise D4LoaderValidationError("source archive", str(error)) from error
    try:
        infos = archive.infolist()
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise D4LoaderValidationError(
                "source archive",
                "contains duplicate member paths",
            )
        try:
            target_payload = archive.read(info_by_name[TARGET_MEMBER])
            readme_payload = archive.read(info_by_name[README_MEMBER])
        except KeyError as error:
            raise D4LoaderValidationError(
                "source evidence member",
                f"missing {error.args[0]!r}",
            ) from error
        if (
            len(readme_payload) != README_BYTES
            or hashlib.sha256(readme_payload).hexdigest() != README_SHA256
        ):
            raise D4LoaderValidationError(
                "README identity",
                "bytes or SHA256 mismatch",
            )
        if b"no pre-processing has been performed" not in readme_payload:
            raise D4LoaderValidationError(
                "README preprocessing evidence",
                "source statement is absent",
            )
        targets_by_well, target_order = _parse_target_table(target_payload)
        low_members = [
            _parse_low_identity(name)
            for name in info_by_name
            if name.startswith(LOW_PREFIX) and name.endswith(".csv")
        ]
        if len(low_members) != EXPECTED_LOW_MEMBERS:
            raise D4LoaderValidationError(
                "Low-SNR member count",
                f"must equal {EXPECTED_LOW_MEMBERS}",
            )
        if len({member.record_id for member in low_members}) != len(low_members):
            raise D4LoaderValidationError(
                "Low-SNR identities",
                "record IDs must be unique",
            )
        by_well: dict[str, list[_LowMember]] = defaultdict(list)
        for member in low_members:
            by_well[member.well_id].append(member)
        if set(by_well) != set(targets_by_well) or any(
            len(members) != RECORDS_PER_WELL
            for members in by_well.values()
        ):
            raise D4LoaderValidationError(
                "Low-SNR well grid",
                "must contain 32 records for every target well",
            )
        expected_round_repetition = {
            (source_round, repetition)
            for source_round in range(1, 9)
            for repetition in range(1, 5)
        }
        for well_id, members in by_well.items():
            observed = {
                (member.source_round, member.source_repetition)
                for member in members
            }
            if observed != expected_round_repetition:
                raise D4LoaderValidationError(
                    f"well {well_id}",
                    "round/repetition grid mismatch",
                )

        mixture_wells = [
            well_id
            for well_id in target_order
            if well_id not in PURE_REFERENCE_WELLS
        ]
        if (
            len(mixture_wells) != EXPECTED_MIXTURE_WELLS
            or _canonical_order_digest(mixture_wells)
            != audit["cohort"]["mixture_well_ids_sha256"]
        ):
            raise D4LoaderValidationError(
                "mixture well identity",
                "count or SHA256 mismatch",
            )
        folds_wells = [mixture_wells[fold::5] for fold in range(5)]
        if [
            _canonical_order_digest(wells) for wells in folds_wells
        ] != [
            fold["well_ids_sha256"] for fold in audit["folds"]
        ]:
            raise D4LoaderValidationError(
                "fold identity",
                "well-ID SHA256 mismatch",
            )

        selected_members = sorted(
            (
                member
                for well_id in mixture_wells
                for member in by_well[well_id]
            ),
            key=lambda member: member.record_id,
        )
        blank_members = sorted(
            by_well[BLANK_WELL],
            key=lambda member: member.record_id,
        )
        if (
            len(selected_members) != EXPECTED_MIXTURE_RECORDS
            or _ids_digest(
                tuple(member.record_id for member in selected_members)
            )
            != MIXTURE_RECORD_IDS_SHA256
            or _canonical_order_digest(
                [member.source_member for member in selected_members]
            )
            != MIXTURE_SOURCE_MEMBERS_SHA256
            or _ids_digest(tuple(member.record_id for member in blank_members))
            != BLANK_RECORD_IDS_SHA256
            or _canonical_order_digest(
                [member.source_member for member in blank_members]
            )
            != BLANK_SOURCE_MEMBERS_SHA256
        ):
            raise D4LoaderValidationError(
                "selected member identities",
                "count or SHA256 mismatch",
            )

        axis: np.ndarray | None = None
        intensity_rows = []
        target_rows = []
        for member in selected_members:
            current_axis, intensity = _parse_spectrum(
                archive.read(info_by_name[member.source_member]),
                source_member=member.source_member,
            )
            if axis is None:
                axis = current_axis
            elif not np.array_equal(axis, current_axis):
                raise D4LoaderValidationError(
                    member.source_member,
                    "axis differs from the shared Low-SNR axis",
                )
            intensity_rows.append(intensity)
            target_rows.append(targets_by_well[member.well_id])
        blank_intensity_rows = []
        blank_target_rows = []
        for member in blank_members:
            current_axis, intensity = _parse_spectrum(
                archive.read(info_by_name[member.source_member]),
                source_member=member.source_member,
            )
            if axis is None or not np.array_equal(axis, current_axis):
                raise D4LoaderValidationError(
                    member.source_member,
                    "blank axis differs from the mixture axis",
                )
            blank_intensity_rows.append(intensity)
            blank_target_rows.append(targets_by_well[member.well_id])
    finally:
        archive.close()

    if axis is None or hashlib.sha256(axis.tobytes()).hexdigest() != AXIS_SHA256:
        raise D4LoaderValidationError(
            "shared axis identity",
            "SHA256 mismatch",
        )
    intensity = _read_only(np.stack(intensity_rows).astype("<f4", copy=False))
    targets = _read_only(np.stack(target_rows).astype("<f8", copy=False))
    blank_intensity = _read_only(
        np.stack(blank_intensity_rows).astype("<f4", copy=False)
    )
    blank_targets = _read_only(
        np.stack(blank_target_rows).astype("<f8", copy=False)
    )
    axis = _read_only(np.asarray(axis, dtype="<f4"))
    record_ids = tuple(member.record_id for member in selected_members)
    well_ids = tuple(member.well_id for member in selected_members)
    source_members = tuple(
        member.source_member for member in selected_members
    )
    rounds = _read_only(
        np.asarray(
            [member.source_round for member in selected_members],
            dtype="<i8",
        )
    )
    repetitions = _read_only(
        np.asarray(
            [member.source_repetition for member in selected_members],
            dtype="<i8",
        )
    )
    fold_indices = []
    for fold, wells in enumerate(folds_wells):
        well_set = set(wells)
        indices = _read_only(
            np.asarray(
                [
                    index
                    for index, well_id in enumerate(well_ids)
                    if well_id in well_set
                ],
                dtype="<i8",
            )
        )
        if (
            indices.size != 1536
            or _ids_digest(
                tuple(record_ids[int(index)] for index in indices)
            )
            != FOLD_RECORD_IDS_SHA256[fold]
            or _canonical_order_digest(
                [source_members[int(index)] for index in indices]
            )
            != FOLD_SOURCE_MEMBERS_SHA256[fold]
        ):
            raise D4LoaderValidationError(
                f"fold {fold}",
                "record or source-member identity mismatch",
            )
        fold_indices.append(indices)
    splits = []
    for seed in range(5):
        test_fold = seed
        validation_fold = (seed + 1) % 5
        train_folds = tuple(
            fold
            for fold in range(5)
            if fold not in {test_fold, validation_fold}
        )
        train = _read_only(
            np.sort(
                np.concatenate(
                    [fold_indices[fold] for fold in train_folds]
                )
            ).astype("<i8", copy=False)
        )
        splits.append(
            D4WellSplit(
                seed=seed,
                train_indices=train,
                validation_indices=fold_indices[validation_fold],
                test_indices=fold_indices[test_fold],
                train_folds=train_folds,
                validation_fold=validation_fold,
                test_fold=test_fold,
            )
        )
    return D4SugarCohort(
        protocol_config_sha256=config.sha256,
        intensity=intensity,
        targets=targets,
        wavenumber=axis,
        target_names=TARGET_NAMES,
        record_ids=record_ids,
        well_ids=well_ids,
        source_members=source_members,
        rounds=rounds,
        repetitions=repetitions,
        blank_intensity=blank_intensity,
        blank_targets=blank_targets,
        blank_record_ids=tuple(member.record_id for member in blank_members),
        blank_well_ids=tuple(member.well_id for member in blank_members),
        blank_source_members=tuple(
            member.source_member for member in blank_members
        ),
        folds=tuple(fold_indices),
        splits=tuple(splits),
    )


__all__ = [
    "D4LoaderValidationError",
    "D4ProtocolConfig",
    "D4SugarCohort",
    "D4WellSplit",
    "load_d4_protocol_config",
    "load_d4_sugar_cohort",
]
