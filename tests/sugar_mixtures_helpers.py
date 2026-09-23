from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import struct
import warnings
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np


ZIP_MEMBER_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_EXTERNAL_ATTR = 0o100644 << 16
ZIP_COMPRESSION_LEVEL = 6

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
HIGH_DIRECTORY = f"{RAW_PREFIX}Sugar_Concentration_Test/"
LOW_DIRECTORY = f"{RAW_PREFIX}Sugar_Concentration_Test_Fast/"
TARGET_MEMBER = f"{RAW_PREFIX}Sugar_Concentrations.csv"
HIGH_SPECTRA_ORACLE = (
    f"{RAW_PREFIX}Sugar_Concentration_Test_ALL_spectra.csv"
)
HIGH_METADATA_ORACLE = (
    f"{RAW_PREFIX}Sugar_Concentration_Test_ALL_metadata.csv"
)
LOW_SPECTRA_ORACLE = (
    f"{RAW_PREFIX}Sugar_Concentration_Test_Fast_ALL_spectra.csv"
)
LOW_METADATA_ORACLE = (
    f"{RAW_PREFIX}Sugar_Concentration_Test_Fast_ALL_metadata.csv"
)
CLAIM_README = f"{RAW_PREFIX}README.md"
EXPERIMENTAL_README = (
    "Raw data/Experimental data from sugar mixtures/README.txt"
)
PREPARED_PREFIX = (
    "Raw data/Experimental data from sugar mixtures/"
    "Raw datasets for analyses/"
)


@dataclass(frozen=True)
class SyntheticSugarSource:
    raw_root: Path
    archive_path: Path
    archive_bytes: int
    archive_md5: str
    archive_sha256: str
    archive_members: Sequence[str]
    role_members: Mapping[str, Sequence[str]]
    member_bytes: Mapping[str, int]
    member_crc32: Mapping[str, int]
    member_sha256: Mapping[str, str]
    canonical_members: Sequence[str]
    target_member: str
    high_oracle_members: tuple[str, str]
    low_oracle_members: tuple[str, str]
    prepared_members: Sequence[str]
    record_ids: Sequence[str]
    sample_ids: Sequence[str]
    source_json_text: Mapping[str, str]


@dataclass(frozen=True)
class SyntheticZipEntry:
    name: str
    payload: bytes
    compress_type: int = zipfile.ZIP_DEFLATED
    external_attr: int = ZIP_EXTERNAL_ATTR
    create_system: int = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(
    *,
    condition: str,
    record_index: int,
) -> dict[str, object]:
    is_high = condition == "high_snr"
    integration_time: int | float = 5 if is_high else 0.5
    laser_power = 36.3 if is_high else 30.3
    spectro_temp = -59 if is_high else -55
    detector_temp = -60 if is_high else -50
    return {
        "Date": f"2023-09-0{6 + int(not is_high)} "
        f"{19 + (record_index % 2):02d}:{record_index % 60:02d}:17",
        "Position [um]": [
            float(record_index),
            float(record_index + 1),
            0.0,
        ],
        "Temperature [C]": "N/A",
        "Humidity [1/100]": "N/A",
        "Spectro_Temp [C]": spectro_temp,
        "Laser power [mW]": laser_power,
        "Excitation wavelength [nm]": 785,
        "Objective": "20x/0.4",
        "Objective_Maker": "Synthetic Optics",
        "Objective_Magnification": 20.0,
        "Objective_NA": 0.4,
        "Objective_WD": 1.2,
        "Objective_Immersion": "Air",
        "Objective_Tube_Lens_f": 180.0,
        "Spectrometer integration time [s]": integration_time,
        "Spectrometer number of accumulations": 1,
        "Spectrometer temperature [C]": detector_temp,
        "Spectrometer acquisition mode": 0,
        "Spectrometer read mode": 1,
        "Spectrometer trigger mode": 0,
        "Magnification": 20.0,
        "Spot_Size": 2.5,
        "Laser_Offset": [float("nan"), float("nan")],
    }


def source_json_text(
    *,
    condition: str = "high_snr",
    record_index: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> str:
    document = (
        _source_metadata(condition=condition, record_index=record_index)
        if metadata is None
        else dict(metadata)
    )
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(", ", ": "),
        allow_nan=True,
    )


def build_member_payload(
    *,
    condition: str = "high_snr",
    record_index: int = 0,
    point_count: int = 16,
    header: Sequence[str] = (
        "Pixel",
        "wl",
        "cm-1",
        "Intensity",
        "Metadata",
    ),
    numeric_rows: Sequence[Sequence[object]] | None = None,
    metadata_text: str | None = None,
    metadata_rows: int = 1,
    metadata_first: bool = False,
) -> bytes:
    pixel = np.arange(point_count, dtype=np.int64)
    wavelength = np.linspace(800.0, 815.0, point_count, dtype=np.float64)
    wavenumber = 1.0e7 / 785.0 - 1.0e7 / wavelength
    intensity = (
        100
        + record_index * 10
        + np.arange(point_count, dtype=np.int64)
    )
    rows = (
        tuple(
            (
                str(int(pixel_value)),
                f"{wavelength_value:.12f}",
                f"{wavenumber_value:.12f}",
                str(int(intensity_value)),
                "",
            )
            for (
                pixel_value,
                wavelength_value,
                wavenumber_value,
                intensity_value,
            ) in zip(pixel, wavelength, wavenumber, intensity, strict=True)
        )
        if numeric_rows is None
        else tuple(tuple(str(value) for value in row) for row in numeric_rows)
    )
    exact_metadata_text = (
        source_json_text(
            condition=condition,
            record_index=record_index,
        )
        if metadata_text is None
        else metadata_text
    )
    metadata = tuple(
        ("", "", "", "", exact_metadata_text)
        for _ in range(metadata_rows)
    )
    body = (*metadata, *rows) if metadata_first else (*rows, *metadata)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(tuple(header))
    writer.writerows(body)
    return stream.getvalue().encode("utf-8")


def _well_recipes() -> tuple[tuple[str, int, str, int, int, ...], ...]:
    return (
        ("A1_1", 1, "A", 1, 1, 30, 75, 120, 0, 150),
        ("A2_1", 2, "A", 2, 1, 75, 30, 0, 120, 150),
        ("E1_3", 49, "E", 1, 3, 0, 0, 0, 0, 375),
        ("E2_3", 50, "E", 2, 3, 375, 0, 0, 0, 0),
        ("E3_3", 51, "E", 3, 3, 0, 375, 0, 0, 0),
        ("E4_3", 52, "E", 4, 3, 0, 0, 375, 0, 0),
        ("E5_3", 53, "E", 5, 3, 0, 0, 0, 375, 0),
    )


def _canonical_filename(
    *,
    condition: str,
    samp: int,
    row: str,
    column: int,
    plate: int,
    repetition: int,
) -> str:
    if condition == "high_snr":
        directory = HIGH_DIRECTORY
        prefix = "Sugar_Concentration_Test"
    else:
        directory = LOW_DIRECTORY
        prefix = "Sugar_Concentration_Test_Fast"
    return (
        f"{directory}{prefix}_{samp}_{row}{column}_{plate}_"
        f"RD1_M1_R{repetition}.csv"
    )


def _well_from_canonical_member(member: str) -> str:
    identity = _identity_from_canonical_member(member)
    return f"{identity['row']}{identity['column']}_{identity['plate']}"


def _identity_from_canonical_member(member: str) -> Mapping[str, int | str]:
    basename = member.rsplit("/", 1)[-1].removesuffix(".csv")
    for prefix in (
        "Sugar_Concentration_Test_Fast_",
        "Sugar_Concentration_Test_",
    ):
        if basename.startswith(prefix):
            samp, row_column, plate, round_text, measurement, repetition = (
                basename.removeprefix(prefix).split("_")
            )
            return {
                "filename": basename,
                "samp": int(samp),
                "row": row_column[0],
                "column": int(row_column[1:]),
                "plate": int(plate),
                "round": int(round_text.removeprefix("RD")),
                "measurement": int(measurement.removeprefix("M")),
                "repetition": int(repetition.removeprefix("R")),
            }
    raise ValueError(member)


def _record_id(
    *,
    condition: str,
    samp: int,
    row: str,
    column: int,
    plate: int,
    repetition: int,
) -> str:
    return (
        f"{condition}-s{samp:03d}-{row.lower()}{column:02d}-p{plate:02d}"
        f"-r01-m01-rep{repetition:02d}"
    )


def _target_payload() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
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
    )
    for recipe in _well_recipes():
        (
            well,
            samp,
            row,
            column,
            plate,
            sucrose,
            fructose,
            maltose,
            glucose,
            water,
        ) = recipe
        writer.writerow(
            (
                well,
                samp,
                row,
                column,
                plate,
                sucrose,
                fructose,
                maltose,
                glucose,
                water,
                375,
            )
        )
    return stream.getvalue().encode("utf-8")


def _oracle_payloads(
    canonical_payloads: Mapping[str, bytes],
) -> Mapping[str, bytes]:
    by_condition = {
        "high_snr": [
            name
            for name in canonical_payloads
            if name.startswith(HIGH_DIRECTORY)
        ],
        "low_snr": [
            name
            for name in canonical_payloads
            if name.startswith(LOW_DIRECTORY)
        ],
    }
    result: dict[str, bytes] = {}
    for condition, members in by_condition.items():
        ordered = sorted(members, key=lambda value: value.encode("utf-8"))
        parsed_by_member = {}
        for member in ordered:
            rows = list(
                csv.reader(
                    io.StringIO(
                        canonical_payloads[member].decode("utf-8")
                    )
                )
            )
            parsed_by_member[member] = rows
        spectra_stream = io.StringIO(newline="")
        spectra_writer = csv.writer(spectra_stream, lineterminator="\n")
        spectra_writer.writerow(
            (
                "cm-1",
                *(
                    member.rsplit("/", 1)[-1].removesuffix(".csv")
                    for member in ordered
                ),
            )
        )
        axis = [row[2] for row in parsed_by_member[ordered[0]][1:-1]]
        for index, wavenumber in enumerate(axis):
            spectra_writer.writerow(
                (
                    wavenumber,
                    *(
                        parsed_by_member[member][index + 1][3]
                        for member in ordered
                    ),
                )
            )
        metadata_stream = io.StringIO(newline="")
        metadata_writer = csv.writer(metadata_stream, lineterminator="\n")
        metadata_writer.writerow(
            (
                "filename",
                "samp",
                "row",
                "col",
                "plate",
                "round",
                "meas",
                "rep",
                "Sucrose [ul]",
                "Fructose [ul]",
                "Maltose [ul]",
                "Glucose [ul]",
                "Water [ul]",
                "Total Volume [ul]",
            )
        )
        recipes = {
            f"{row}{column}_{plate}": (
                sucrose,
                fructose,
                maltose,
                glucose,
                water,
                375,
            )
            for (
                _well,
                _samp,
                row,
                column,
                plate,
                sucrose,
                fructose,
                maltose,
                glucose,
                water,
            ) in _well_recipes()
        }
        for member in ordered:
            identity = _identity_from_canonical_member(member)
            well = _well_from_canonical_member(member)
            metadata_writer.writerow(
                (
                    identity["filename"],
                    identity["samp"],
                    identity["row"],
                    identity["column"],
                    identity["plate"],
                    identity["round"],
                    identity["measurement"],
                    identity["repetition"],
                    *recipes[well],
                )
            )
        if condition == "high_snr":
            result[HIGH_SPECTRA_ORACLE] = spectra_stream.getvalue().encode()
            result[HIGH_METADATA_ORACLE] = metadata_stream.getvalue().encode()
        else:
            result[LOW_SPECTRA_ORACLE] = spectra_stream.getvalue().encode()
            result[LOW_METADATA_ORACLE] = metadata_stream.getvalue().encode()
    return MappingProxyType(result)


def _prepared_payloads(
    canonical_payloads: Mapping[str, bytes],
) -> Mapping[str, bytes]:
    result: dict[str, bytes] = {
        f"{PREPARED_PREFIX}Raman_Sugar_Dataset.ipynb": (
            b'{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\n'
        ),
        f"{PREPARED_PREFIX}README.md": b"Synthetic prepared evidence.\n",
    }
    view_names = (
        "High SNR",
        "High SNR (no refs)",
        "Low SNR",
        "Low SNR (no refs)",
    )
    component_wells = {
        "sucrose": "_E2_3_",
        "fructose": "_E3_3_",
        "maltose": "_E4_3_",
        "glucose": "_E5_3_",
        "water": "_E1_3_",
    }
    prepared_endmembers = {}
    prepared_intensity = {}
    prepared_axis = {}
    prepared_fractions = {}
    recipe_by_well = {
        f"{row}{column}_{plate}": np.array(
            [sucrose, fructose, maltose, glucose, water],
            dtype=np.float64,
        )
        / 375.0
        for (
            _well,
            _samp,
            row,
            column,
            plate,
            sucrose,
            fructose,
            maltose,
            glucose,
            water,
        ) in _well_recipes()
    }
    for condition in ("high_snr", "low_snr"):
        directory = HIGH_DIRECTORY if condition == "high_snr" else LOW_DIRECTORY
        condition_members = sorted(
            (
                member
                for member in canonical_payloads
                if member.startswith(directory)
            ),
            key=lambda value: value.encode("utf-8"),
        )
        intensities_by_member = {}
        for member in condition_members:
            rows = list(
                csv.reader(
                    io.StringIO(
                        canonical_payloads[member].decode("utf-8")
                    )
                )
            )
            intensities_by_member[member] = np.array(
                [float(row[3]) for row in rows[1:-1]],
                dtype=np.float64,
            )
            if condition not in prepared_axis:
                prepared_axis[condition] = np.array(
                    [row[2] for row in rows[1:-1]],
                    dtype=object,
                )
        prepared_intensity[condition] = intensities_by_member
        prepared_fractions[condition] = {
            member: recipe_by_well[_well_from_canonical_member(member)]
            for member in condition_members
        }
        arrays = []
        for component in (
            "sucrose",
            "fructose",
            "maltose",
            "glucose",
            "water",
        ):
            members = sorted(
                (
                    member
                    for member in canonical_payloads
                    if member.startswith(directory)
                    and component_wells[component] in member
                ),
                key=lambda value: value.encode("utf-8"),
            )
            intensities = []
            for member in members:
                rows = list(
                    csv.reader(
                        io.StringIO(
                            canonical_payloads[member].decode("utf-8")
                        )
                    )
                )
                intensities.append(
                    np.array(
                        [float(row[3]) for row in rows[1:-1]],
                        dtype=np.float64,
                    )
                )
            arrays.append(
                np.median(np.stack(intensities, axis=0), axis=0)
            )
        prepared_endmembers[condition] = np.stack(arrays, axis=0)
    for view_name in view_names:
        condition = "high_snr" if view_name.startswith("High") else "low_snr"
        members = [
            member
            for member in canonical_payloads
            if (member.startswith(HIGH_DIRECTORY)) == (condition == "high_snr")
        ]
        if "no refs" in view_name:
            members = [
                member
                for member in reversed(members)
                if "_E1_3_" not in member
                and "_E2_3_" not in member
                and "_E3_3_" not in member
                and "_E4_3_" not in member
                and "_E5_3_" not in member
            ]
        payloads = {
            "data.pkl": np.stack(
                [prepared_intensity[condition][member] for member in members],
                axis=0,
            ),
            "spectral_axis.pkl": prepared_axis[condition],
            "gt_abundance_image.pkl": np.stack(
                [prepared_fractions[condition][member] for member in members],
                axis=0,
            ),
            "gt_endmembers.pkl": prepared_endmembers[condition],
        }
        for basename, value in payloads.items():
            result[f"{PREPARED_PREFIX}{view_name}/{basename}"] = pickle.dumps(
                value,
                protocol=4,
            )
        metadata_stream = io.StringIO(newline="")
        metadata_writer = csv.writer(metadata_stream, lineterminator="\n")
        metadata_writer.writerow(
            (
                "",
                "samp",
                "row",
                "col",
                "plate",
                "round",
                "meas",
                "rep",
                "Sucrose [ul]",
                "Fructose [ul]",
                "Maltose [ul]",
                "Glucose [ul]",
                "Water [ul]",
                "Total Volume [ul]",
            )
        )
        for member in members:
            identity = _identity_from_canonical_member(member)
            volumes = (
                prepared_fractions[condition][member] * 375.0
            ).astype(np.int64)
            metadata_writer.writerow(
                (
                    identity["filename"],
                    identity["samp"],
                    identity["row"],
                    identity["column"],
                    identity["plate"],
                    identity["round"],
                    identity["measurement"],
                    identity["repetition"],
                    *volumes.tolist(),
                    375,
                )
            )
        result[
            f"{PREPARED_PREFIX}{view_name}/metadata.csv"
        ] = metadata_stream.getvalue().encode()
    return MappingProxyType(result)


def _support_payloads() -> Mapping[str, bytes]:
    result = {
        f"{RAW_PREFIX}support/support_{index:02d}.txt": (
            f"synthetic support evidence {index}\n".encode()
        )
        for index in range(17)
    }
    result[CLAIM_README] = (
        b"Synthetic acquisition description.\n"
        b"Note that no pre-processing has been performed such that the user "
        b"can decide what is best for the analysis. \n"
    )
    result[EXPERIMENTAL_README] = b"Synthetic experimental overview.\n"
    result[
        "Raw data/Unrelated synthetic data/synthetic.csv"
    ] = b"x,y\n0,0\n"
    return MappingProxyType(result)


def _write_deterministic_zip(
    path: Path,
    members: Mapping[str, bytes],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=ZIP_COMPRESSION_LEVEL,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            info = zipfile.ZipInfo(name, ZIP_MEMBER_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_EXTERNAL_ATTR
            info.create_system = 3
            archive.writestr(
                info,
                members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=ZIP_COMPRESSION_LEVEL,
            )


def read_synthetic_zip_entries(
    path: Path,
) -> tuple[SyntheticZipEntry, ...]:
    entries = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            entries.append(
                SyntheticZipEntry(
                    name=info.filename,
                    payload=b"" if info.is_dir() else archive.read(info),
                    compress_type=info.compress_type,
                    external_attr=info.external_attr,
                    create_system=info.create_system,
                )
            )
    return tuple(entries)


def rewrite_synthetic_zip(
    source: SyntheticSugarSource,
    transform: Callable[
        [tuple[SyntheticZipEntry, ...]],
        Sequence[SyntheticZipEntry],
    ],
) -> None:
    entries = tuple(transform(read_synthetic_zip_entries(source.archive_path)))
    with zipfile.ZipFile(
        source.archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=ZIP_COMPRESSION_LEVEL,
        strict_timestamps=True,
    ) as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for entry in entries:
                info = zipfile.ZipInfo(entry.name, ZIP_MEMBER_TIMESTAMP)
                info.compress_type = entry.compress_type
                info.external_attr = entry.external_attr
                info.create_system = entry.create_system
                archive.writestr(
                    info,
                    entry.payload,
                    compress_type=entry.compress_type,
                    compresslevel=(
                        ZIP_COMPRESSION_LEVEL
                        if entry.compress_type == zipfile.ZIP_DEFLATED
                        else None
                    ),
                )


def set_synthetic_zip_member_flag_bits(
    path: Path,
    member: str,
    flag_bits: int,
) -> None:
    payload = bytearray(path.read_bytes())
    signature = b"PK\x01\x02"
    offset = 0
    found = False
    while True:
        offset = payload.find(signature, offset)
        if offset < 0:
            break
        name_length = struct.unpack_from("<H", payload, offset + 28)[0]
        extra_length = struct.unpack_from("<H", payload, offset + 30)[0]
        comment_length = struct.unpack_from("<H", payload, offset + 32)[0]
        name_start = offset + 46
        name_end = name_start + name_length
        name = bytes(payload[name_start:name_end]).decode("utf-8")
        if name == member:
            struct.pack_into("<H", payload, offset + 8, flag_bits)
            local_offset = struct.unpack_from("<I", payload, offset + 42)[0]
            struct.pack_into("<H", payload, local_offset + 6, flag_bits)
            found = True
            break
        offset = name_end + extra_length + comment_length
    if not found:
        raise KeyError(member)
    path.write_bytes(payload)


def corrupt_synthetic_zip_member_data(
    path: Path,
    member: str,
) -> None:
    payload = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        local_offset = info.header_offset
        name_length = struct.unpack_from("<H", payload, local_offset + 26)[0]
        extra_length = struct.unpack_from("<H", payload, local_offset + 28)[0]
        data_start = local_offset + 30 + name_length + extra_length
        data_offset = data_start + max(1, info.compress_size // 2)
    payload[data_offset] ^= 0x01
    path.write_bytes(payload)


def central_directory_inventory_sha256(path: Path) -> str:
    digest = hashlib.sha256(b"rpe-sugar-zip-inventory-v1\0")
    with zipfile.ZipFile(path) as archive:
        for info in sorted(
            archive.infolist(),
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


def create_synthetic_sugar_source(root: Path) -> SyntheticSugarSource:
    root = Path(root)
    raw_root = root / "cache" / "10779223"
    archive_path = raw_root / "Raw data.zip"
    canonical_payloads: dict[str, bytes] = {}
    source_text_by_member: dict[str, str] = {}
    record_ids = []
    sample_ids = set()
    record_index = 0
    for (
        _well,
        samp,
        row,
        column,
        plate,
        _sucrose,
        _fructose,
        _maltose,
        _glucose,
        _water,
    ) in _well_recipes():
        for condition, repetitions in (
            ("high_snr", (1,)),
            ("low_snr", (1, 2)),
        ):
            for repetition in repetitions:
                member = _canonical_filename(
                    condition=condition,
                    samp=samp,
                    row=row,
                    column=column,
                    plate=plate,
                    repetition=repetition,
                )
                text = source_json_text(
                    condition=condition,
                    record_index=record_index,
                )
                canonical_payloads[member] = build_member_payload(
                    condition=condition,
                    record_index=record_index,
                    metadata_text=text,
                )
                source_text_by_member[member] = text
                record_ids.append(
                    _record_id(
                        condition=condition,
                        samp=samp,
                        row=row,
                        column=column,
                        plate=plate,
                        repetition=repetition,
                    )
                )
                sample_ids.add(
                    f"sugar-well-{row.lower()}{column:02d}-p{plate:02d}"
                )
                record_index += 1

    oracle_payloads = dict(_oracle_payloads(canonical_payloads))
    prepared_payloads = dict(
        _prepared_payloads(canonical_payloads)
    )
    support_payloads = dict(_support_payloads())
    members = {
        **canonical_payloads,
        TARGET_MEMBER: _target_payload(),
        **oracle_payloads,
        **prepared_payloads,
        **support_payloads,
    }
    _write_deterministic_zip(archive_path, members)

    evidence_root = raw_root.parent.parent
    (evidence_root / "evidence" / "zenodo").mkdir(
        parents=True,
        exist_ok=True,
    )
    (evidence_root / "receipts").mkdir(parents=True, exist_ok=True)
    evidence_documents = {
        evidence_root / "evidence" / "zenodo" / "10779223.json": {
            "record_id": "10779223",
            "archive": "Raw data.zip",
            "license": "cc-by-4.0",
        },
        evidence_root / "receipts" / "sugar_mixtures_high_snr.json": {
            "dataset": "high_snr",
            "records": 7,
        },
        evidence_root / "receipts" / "sugar_mixtures_low_snr.json": {
            "dataset": "low_snr",
            "records": 14,
        },
    }
    for path, document in evidence_documents.items():
        path.write_text(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    with zipfile.ZipFile(archive_path) as archive:
        infos = {
            info.filename: info
            for info in archive.infolist()
        }
    role_members = {
        "canonical_high": tuple(
            name for name in canonical_payloads if name.startswith(HIGH_DIRECTORY)
        ),
        "canonical_low": tuple(
            name for name in canonical_payloads if name.startswith(LOW_DIRECTORY)
        ),
        "target": (TARGET_MEMBER,),
        "oracle": tuple(oracle_payloads),
        "prepared": tuple(prepared_payloads),
        "raw_support": tuple(
            name
            for name in support_payloads
            if "/support/" in name or name == CLAIM_README
        ),
        "experimental_readme": (EXPERIMENTAL_README,),
        "excluded_synthetic": (
            "Raw data/Unrelated synthetic data/synthetic.csv",
        ),
    }
    return SyntheticSugarSource(
        raw_root=raw_root,
        archive_path=archive_path,
        archive_bytes=archive_path.stat().st_size,
        archive_md5=file_md5(archive_path),
        archive_sha256=file_sha256(archive_path),
        archive_members=tuple(sorted(members)),
        role_members=MappingProxyType(
            {
                role: tuple(sorted(names))
                for role, names in role_members.items()
            }
        ),
        member_bytes=MappingProxyType(
            {name: len(payload) for name, payload in members.items()}
        ),
        member_crc32=MappingProxyType(
            {name: infos[name].CRC for name in members}
        ),
        member_sha256=MappingProxyType(
            {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in members.items()
            }
        ),
        canonical_members=tuple(sorted(canonical_payloads)),
        target_member=TARGET_MEMBER,
        high_oracle_members=(
            HIGH_SPECTRA_ORACLE,
            HIGH_METADATA_ORACLE,
        ),
        low_oracle_members=(
            LOW_SPECTRA_ORACLE,
            LOW_METADATA_ORACLE,
        ),
        prepared_members=tuple(sorted(prepared_payloads)),
        record_ids=tuple(sorted(record_ids)),
        sample_ids=tuple(sorted(sample_ids)),
        source_json_text=MappingProxyType(dict(source_text_by_member)),
    )


def synthetic_sugar_contract(source: SyntheticSugarSource):
    from rpe.io.sugar_mixtures_models import _SugarSourceContract

    role_counts = {
        role: len(members)
        for role, members in source.role_members.items()
    }
    role_bytes = {
        role: sum(source.member_bytes[name] for name in members)
        for role, members in source.role_members.items()
    }
    role_sha256 = {}
    for role, members in source.role_members.items():
        if role == "excluded_synthetic":
            continue
        digest = hashlib.sha256(b"rpe-sugar-member-content-v1\0")
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            encoded_name = name.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded_name)))
            digest.update(encoded_name)
            digest.update(bytes.fromhex(source.member_sha256[name]))
        role_sha256[role] = digest.hexdigest()
    evidence_root = source.raw_root.parent.parent
    evidence_paths = (
        evidence_root / "evidence" / "zenodo" / "10779223.json",
        evidence_root / "receipts" / "sugar_mixtures_high_snr.json",
        evidence_root / "receipts" / "sugar_mixtures_low_snr.json",
    )
    return _SugarSourceContract(
        archive_name="Raw data.zip",
        archive_bytes=source.archive_bytes,
        archive_md5=source.archive_md5,
        archive_sha256=source.archive_sha256,
        archive_member_count=len(source.archive_members),
        archive_uncompressed_bytes=sum(source.member_bytes.values()),
        central_directory_inventory_sha256=(
            central_directory_inventory_sha256(source.archive_path)
        ),
        expected_role_counts=MappingProxyType(role_counts),
        expected_role_bytes=MappingProxyType(role_bytes),
        expected_role_sha256=MappingProxyType(role_sha256),
        expected_evidence_snapshots=MappingProxyType(
            {
                path.relative_to(evidence_root).as_posix(): (
                    path.stat().st_size,
                    file_sha256(path),
                )
                for path in evidence_paths
            }
        ),
        expected_counts=MappingProxyType(
            {
                "canonical_records": 21,
                "high_records": 7,
                "low_records": 14,
                "high_records_per_well": 1,
                "low_records_per_well": 2,
                "high_round_count": 1,
                "low_round_count": 1,
                "high_repetitions_per_round": 1,
                "low_repetitions_per_round": 2,
                "sample_count": 7,
                "points_per_record": 16,
                "raw_text_inventory_sha256": (
                    "d5b2b47d5ec0785d09c449f56e7575935e3f1f5b8d710d9d1d419fdc2835e3da"
                ),
                "normalized_metadata_sha256": (
                    "1e3ab230a2dce7054712d143cdc7262b23b74d35df63e2c557ab60af6dbb47d2"
                ),
                "core_metadata_sha256": (
                    "7614044838509ce26cedf5d6de7b74b90d544458dc6e4f7c4ff553e0528839e2"
                ),
                "well_target_contract_sha256": (
                    "f4f612748ffa11a077ed9833ca78391ea8cef0a2df50f0fa3bdc0f927f318b98"
                ),
                "record_target_map_sha256": (
                    "0b9c60bb0fd6cd4e6d13d0f5e57daecad138626892ee001682c440a2deb66727"
                ),
                "record_preprocessing_map_sha256": (
                    "30e2d841cc7290fe5decab17302d57b49eccb32ef8bc03193341811179791e5c"
                ),
                "preprocessing_evidence_document_sha256": (
                    "af13cffa89ffe8f9041493914029a355857990c552d75a1c4ce22c75bd0f70f3"
                ),
                "intensity_float32_max_abs_error": 0.0,
                "axis_float32_max_abs_error_cm1": (
                    1.4110402389633236e-05
                ),
                "wavelength_float32_max_abs_error_nm": 0.0,
            }
        ),
    )


def refresh_synthetic_sugar_contract(
    source: SyntheticSugarSource,
    *,
    refresh_roles: bool = False,
    role_members: Mapping[str, Sequence[str]] | None = None,
):
    contract = synthetic_sugar_contract(source)
    with zipfile.ZipFile(source.archive_path) as archive:
        infos = tuple(archive.infolist())
        uncompressed_bytes = sum(info.file_size for info in infos)
        if refresh_roles:
            payload_sha256 = {
                info.filename: hashlib.sha256(archive.read(info)).hexdigest()
                for info in infos
                if not info.is_dir()
            }
            role_counts = {}
            role_bytes = {}
            role_sha256 = {}
            info_by_name = {
                info.filename: info
                for info in infos
            }
            current_role_members = (
                source.role_members
                if role_members is None
                else role_members
            )
            for role, names in current_role_members.items():
                role_counts[role] = len(names)
                role_bytes[role] = sum(
                    info_by_name[name].file_size
                    for name in names
                )
                if role == "excluded_synthetic":
                    continue
                digest = hashlib.sha256(
                    b"rpe-sugar-member-content-v1\0"
                )
                for name in sorted(
                    names,
                    key=lambda value: value.encode("utf-8"),
                ):
                    encoded_name = name.encode("utf-8")
                    digest.update(struct.pack("<Q", len(encoded_name)))
                    digest.update(encoded_name)
                    digest.update(bytes.fromhex(payload_sha256[name]))
                role_sha256[role] = digest.hexdigest()
        else:
            role_counts = dict(contract.expected_role_counts)
            role_bytes = dict(contract.expected_role_bytes)
            role_sha256 = dict(contract.expected_role_sha256)
    return replace(
        contract,
        archive_bytes=source.archive_path.stat().st_size,
        archive_md5=file_md5(source.archive_path),
        archive_sha256=file_sha256(source.archive_path),
        archive_member_count=len(infos),
        archive_uncompressed_bytes=uncompressed_bytes,
        central_directory_inventory_sha256=(
            central_directory_inventory_sha256(source.archive_path)
        ),
        expected_role_counts=MappingProxyType(role_counts),
        expected_role_bytes=MappingProxyType(role_bytes),
        expected_role_sha256=MappingProxyType(role_sha256),
    )

