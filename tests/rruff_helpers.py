from __future__ import annotations

import hashlib
import re
import struct
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from rpe.io.rruff_source import _ArchiveContract, _RruffSourceContract


ZIP_MEMBER_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ARCHIVE_NAMES = (
    "LR-Raman.zip",
    "excellent_oriented.zip",
    "excellent_unoriented.zip",
    "fair_oriented.zip",
    "fair_unoriented.zip",
    "poor_unoriented.zip",
    "unrated_oriented.zip",
    "unrated_unoriented.zip",
)

QUARTZ_RAW_MEMBER = (
    "Quartz__R000001__Raman__532__0__unoriented__"
    "Raman_Data_Raw__0000000000000000000000000001.txt"
)
QUARTZ_PROCESSED_MEMBER = (
    "Quartz__R000001__Raman__532__0__unoriented__"
    "Raman_Data_Processed__0000000000000000000000000002.txt"
)
CALCITE_RAW_MEMBER = (
    "Calcite__R000002__Raman__785__0__oriented__"
    "Raman_Data_RAW__0000000000000000000000000003.txt"
)
CP1252_RAW_MEMBER = (
    "Siderite__R000003__Raman__532__0__unoriented__"
    "Raman_Data_Raw__0000000000000000000000000004.txt"
)
_TEST_MEMBER_PATTERN = re.compile(
    r"^.+__Raman_Data_(?P<kind>Raw|RAW|Processed)__"
    r"(?P<member_id>[0-9a-f]+)\.txt$"
)


@dataclass(frozen=True)
class SyntheticRruffSource:
    raw_root: Path
    archive_bytes: Mapping[str, int]
    archive_sha256: Mapping[str, str]
    archive_members: Mapping[str, tuple[str, ...]]


def _member_text(
    *,
    name: str,
    rruff_id: str,
    filetype: str,
    wavelength: str,
    rows: tuple[str, ...],
    optional_headers: tuple[str, ...] = (),
    extra_lines: tuple[str, ...] = (),
    duplicate_description: bool = False,
) -> str:
    headers = [
        f"##NAMES={name}",
        f"##RRUFFID={rruff_id}",
        f"##FILETYPE={filetype}",
        f"##RAMAN WAVELENGTH={wavelength}",
        f"##URL=https://rruff.info/{rruff_id}",
        *optional_headers,
    ]
    if duplicate_description:
        headers.extend(
            (
                "##CELL PARAMETERS=coarse",
                "##CELL PARAMETERS=precise",
            )
        )
    return "\n".join((*headers, "", *extra_lines, *rows, ""))


def _baseline_members_by_archive() -> Mapping[
    str,
    tuple[tuple[str, bytes], ...],
]:
    empty = tuple()
    archives: dict[str, tuple[tuple[str, bytes], ...]] = {
        name: empty for name in ARCHIVE_NAMES
    }
    archives["LR-Raman.zip"] = (
        (
            QUARTZ_RAW_MEMBER,
            _member_text(
                name="Quartz",
                rruff_id="R000001",
                filetype="Raman RAW",
                wavelength="532",
                extra_lines=(
                    "# Instrument=LabRAM",
                    "instrument export preamble",
                    "X,Y",
                ),
                rows=("100,1", "101,2", "102,3"),
                duplicate_description=True,
            ).encode("utf-8"),
        ),
        (
            QUARTZ_PROCESSED_MEMBER,
            _member_text(
                name="Quartz",
                rruff_id="R000001",
                filetype="Raman Processed",
                wavelength="532",
                rows=("100 0.1", "101 0.2", "102 0.3"),
            ).encode("utf-8"),
        ),
    )
    archives["excellent_oriented.zip"] = (
        (
            CALCITE_RAW_MEMBER,
            _member_text(
                name="Calcite",
                rruff_id="R000002",
                filetype="Raman RAW",
                wavelength="785 nm",
                extra_lines=(
                    "#Instrument=\tXploRA",
                    "#Acq. time (s)=\t2.5",
                    "#Accumulations=\t3",
                    "#Grating=\t1200 gr/mm",
                    "#Detector=\tSyncerity",
                    "#Laser (nm)=\t785",
                ),
                rows=("200,4", "201,5", "202,6"),
            ).encode("utf-8"),
        ),
    )
    archives["fair_unoriented.zip"] = (
        (
            CP1252_RAW_MEMBER,
            _member_text(
                name="Sidérite",
                rruff_id="R000003",
                filetype="Raman RAW",
                wavelength="532",
                rows=("300,7", "301,8", "302,9"),
            ).encode("cp1252"),
        ),
    )
    return MappingProxyType(archives)


def _pairing_member(
    measurement_key: str,
    *,
    member_id: str,
    kind: str,
    mineral_name: str,
    rruff_id: str,
    rows: tuple[str, ...],
    pin_id: str | None = None,
    orientation: str | None = None,
) -> tuple[str, bytes]:
    filename_kind = "RAW" if kind == "raw" else "Processed"
    filetype = "Raman RAW" if kind == "raw" else "Raman Processed"
    optional_headers = tuple(
        header
        for header in (
            None if pin_id is None else f"##PIN_ID={pin_id}",
            None if orientation is None else f"##ORIENTATION={orientation}",
        )
        if header is not None
    )
    return (
        f"{measurement_key}__Raman_Data_{filename_kind}__{member_id}.txt",
        _member_text(
            name=mineral_name,
            rruff_id=rruff_id,
            filetype=filetype,
            wavelength="532",
            optional_headers=optional_headers,
            rows=rows,
        ).encode("utf-8"),
    )


PAIR_EXACT_KEY = "Exactite__R100001__Raman__532__0__unoriented"
PAIR_PROCESSED_SUBSET_KEY = (
    "Procsubsetite__R100002__Raman__532__0__unoriented"
)
PAIR_RAW_SUBSET_KEY = "Rawsubsetite__R100003__Raman__532__0__unoriented"
PAIR_ALIGNMENT_KEY = "Alignite__R100004__Raman__532__0__unoriented"
PAIR_NO_OVERLAP_KEY = "Nooverlapite__R100005__Raman__532__0__unoriented"
PAIR_REJECTED_DUPLICATE_KEY = (
    "Rejectedduplicateite__R100010__Raman__532__0__unoriented"
)
PAIR_CROSS_ARCHIVE_KEY = (
    "Crossarchiveite__R100011__Raman__532__0__unoriented"
)

PAIR_EXACT_RAW_MEMBER_ID = "1000000000000000000000000001"
PAIR_EXACT_PROCESSED_MEMBER_ID = "1000000000000000000000000002"


def _pairing_members_by_archive() -> Mapping[
    str,
    tuple[tuple[str, bytes], ...],
]:
    archives: dict[str, tuple[tuple[str, bytes], ...]] = {
        name: () for name in ARCHIVE_NAMES
    }
    archives["LR-Raman.zip"] = (
        _pairing_member(
            PAIR_EXACT_KEY,
            member_id=PAIR_EXACT_RAW_MEMBER_ID,
            kind="raw",
            mineral_name="Exactite",
            rruff_id="R100001",
            pin_id="PIN-EXACT",
            orientation="unoriented",
            rows=("100,1", "101,2", "102,3"),
        ),
        _pairing_member(
            PAIR_EXACT_KEY,
            member_id=PAIR_EXACT_PROCESSED_MEMBER_ID,
            kind="processed",
            mineral_name="Exactite",
            rruff_id="R100001",
            pin_id="PIN-EXACT",
            orientation="unoriented",
            rows=("100,0.1", "101,0.2", "102,0.3"),
        ),
        _pairing_member(
            PAIR_PROCESSED_SUBSET_KEY,
            member_id="1000000000000000000000000003",
            kind="raw",
            mineral_name="Procsubsetite",
            rruff_id="R100002",
            rows=("200,1", "201,2", "202,3", "203,4"),
        ),
        _pairing_member(
            PAIR_PROCESSED_SUBSET_KEY,
            member_id="1000000000000000000000000004",
            kind="processed",
            mineral_name="Procsubsetite",
            rruff_id="R100002",
            rows=("201,0.2", "202,0.3"),
        ),
        _pairing_member(
            PAIR_RAW_SUBSET_KEY,
            member_id="1000000000000000000000000005",
            kind="raw",
            mineral_name="Rawsubsetite",
            rruff_id="R100003",
            rows=("301,1", "302,2"),
        ),
        _pairing_member(
            PAIR_RAW_SUBSET_KEY,
            member_id="1000000000000000000000000006",
            kind="processed",
            mineral_name="Rawsubsetite",
            rruff_id="R100003",
            rows=("300,0.1", "301,0.2", "302,0.3", "303,0.4"),
        ),
        _pairing_member(
            PAIR_ALIGNMENT_KEY,
            member_id="1000000000000000000000000007",
            kind="raw",
            mineral_name="Alignite",
            rruff_id="R100004",
            rows=("400,1", "401,2", "402,3"),
        ),
        _pairing_member(
            PAIR_ALIGNMENT_KEY,
            member_id="1000000000000000000000000008",
            kind="processed",
            mineral_name="Alignite",
            rruff_id="R100004",
            rows=("400.5,0.1", "401.5,0.2", "402.5,0.3"),
        ),
        _pairing_member(
            PAIR_NO_OVERLAP_KEY,
            member_id="1000000000000000000000000009",
            kind="raw",
            mineral_name="Nooverlapite",
            rruff_id="R100005",
            rows=("500,1", "501,2"),
        ),
        _pairing_member(
            PAIR_NO_OVERLAP_KEY,
            member_id="1000000000000000000000000010",
            kind="processed",
            mineral_name="Nooverlapite",
            rruff_id="R100005",
            rows=("600,0.1", "601,0.2"),
        ),
        _pairing_member(
            "Rawonlyite__R100006__Raman__532__0__unoriented",
            member_id="1000000000000000000000000011",
            kind="raw",
            mineral_name="Rawonlyite",
            rruff_id="R100006",
            rows=("700,1", "701,2"),
        ),
        _pairing_member(
            "Processedonlyite__R100007__Raman__532__0__unoriented",
            member_id="1000000000000000000000000012",
            kind="processed",
            mineral_name="Processedonlyite",
            rruff_id="R100007",
            rows=("800,0.1", "801,0.2"),
        ),
        _pairing_member(
            "Tworawite__R100008__Raman__532__0__unoriented",
            member_id="1000000000000000000000000013",
            kind="raw",
            mineral_name="Tworawite",
            rruff_id="R100008",
            rows=("900,1", "901,2"),
        ),
        _pairing_member(
            "Tworawite__R100008__Raman__532__0__unoriented",
            member_id="1000000000000000000000000014",
            kind="raw",
            mineral_name="Tworawite",
            rruff_id="R100008",
            rows=("902,3", "903,4"),
        ),
        _pairing_member(
            "Twoprocessedite__R100009__Raman__532__0__unoriented",
            member_id="1000000000000000000000000015",
            kind="processed",
            mineral_name="Twoprocessedite",
            rruff_id="R100009",
            rows=("1000,0.1", "1001,0.2"),
        ),
        _pairing_member(
            "Twoprocessedite__R100009__Raman__532__0__unoriented",
            member_id="1000000000000000000000000016",
            kind="processed",
            mineral_name="Twoprocessedite",
            rruff_id="R100009",
            rows=("1002,0.3", "1003,0.4"),
        ),
        _pairing_member(
            "Rejectedonlyite__R100012__Raman__532__0__unoriented",
            member_id="1000000000000000000000000017",
            kind="raw",
            mineral_name="Rejectedonlyite",
            rruff_id="R100012",
            rows=("1100,1", "1100,2"),
        ),
        _pairing_member(
            PAIR_REJECTED_DUPLICATE_KEY,
            member_id="1000000000000000000000000018",
            kind="raw",
            mineral_name="Rejectedduplicateite",
            rruff_id="R100010",
            rows=("1200,1", "1201,2"),
        ),
        _pairing_member(
            PAIR_REJECTED_DUPLICATE_KEY,
            member_id="1000000000000000000000000019",
            kind="raw",
            mineral_name="Rejectedduplicateite",
            rruff_id="R100010",
            rows=("1202,1", "1202,2"),
        ),
        _pairing_member(
            PAIR_CROSS_ARCHIVE_KEY,
            member_id="1000000000000000000000000020",
            kind="raw",
            mineral_name="Crossarchiveite",
            rruff_id="R100011",
            rows=("1300,1", "1301,2"),
        ),
    )
    archives["fair_oriented.zip"] = (
        _pairing_member(
            PAIR_CROSS_ARCHIVE_KEY,
            member_id="1000000000000000000000000021",
            kind="raw",
            mineral_name="Crossarchiveite",
            rruff_id="R100011",
            rows=("1300,9", "1301,8"),
        ),
    )
    return MappingProxyType(archives)


PAIRING_REJECTIONS = frozenset(
    {
        (
            "LR-Raman.zip",
            (
                "Rejectedonlyite__R100012__Raman__532__0__unoriented__"
                "Raman_Data_RAW__1000000000000000000000000017.txt"
            ),
            "raw",
            "AXIS_DUPLICATE",
        ),
        (
            "LR-Raman.zip",
            (
                f"{PAIR_REJECTED_DUPLICATE_KEY}__Raman_Data_RAW__"
                "1000000000000000000000000019.txt"
            ),
            "raw",
            "AXIS_DUPLICATE",
        ),
    }
)

PAIRING_STATISTICS = MappingProxyType(
    {
        "source_measurement_groups": 13,
        "source_multiplicity_1_1": 5,
        "source_multiplicity_1_0": 4,
        "source_multiplicity_0_1": 1,
        "source_multiplicity_2_0": 2,
        "source_multiplicity_0_2": 1,
        "paired_unique": 5,
        "raw_only": 3,
        "processed_only": 1,
        "ambiguous": 3,
        "rejected_only": 1,
        "exact_equal": 1,
        "processed_exact_contiguous_subset_of_raw": 1,
        "raw_exact_contiguous_subset_of_processed": 1,
        "overlap_requires_alignment": 1,
        "no_axis_overlap": 1,
        "pointwise_comparable_pairs": 3,
    }
)


def _write_deterministic_zip(
    path: Path,
    members: list[tuple[str, bytes]] | tuple[tuple[str, bytes], ...],
) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=ZIP_MEMBER_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(info, payload)


def _read_archive_members(path: Path) -> tuple[tuple[str, bytes], ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(
            (info.filename, archive.read(info))
            for info in archive.infolist()
            if not info.is_dir()
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_source(raw_root: Path) -> SyntheticRruffSource:
    archives = tuple(sorted(raw_root.glob("*.zip"), key=lambda item: item.name))
    archive_bytes = MappingProxyType(
        {path.name: path.stat().st_size for path in archives}
    )
    archive_sha256 = MappingProxyType(
        {path.name: file_sha256(path) for path in archives}
    )
    archive_members = MappingProxyType(
        {
            path.name: _archive_member_names(path)
            for path in archives
        }
    )
    return SyntheticRruffSource(
        raw_root=raw_root,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_members=archive_members,
    )


def _archive_member_names(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        )


def create_synthetic_rruff_source(raw_root: Path) -> SyntheticRruffSource:
    raw_root.mkdir(parents=True)
    for archive_name, members in _baseline_members_by_archive().items():
        _write_deterministic_zip(raw_root / archive_name, members)
    return _snapshot_source(raw_root)


def create_synthetic_rruff_pairing_source(
    raw_root: Path,
) -> SyntheticRruffSource:
    raw_root.mkdir(parents=True)
    for archive_name, members in _pairing_members_by_archive().items():
        _write_deterministic_zip(raw_root / archive_name, members)
    return _snapshot_source(raw_root)


def rewrite_archive(
    source: SyntheticRruffSource,
    archive: str,
    mutation: Callable[
        [list[tuple[str, bytes]]],
        list[tuple[str, bytes]],
    ],
) -> SyntheticRruffSource:
    members = _read_archive_members(source.raw_root / archive)
    _write_deterministic_zip(
        source.raw_root / archive,
        mutation(list(members)),
    )
    return _snapshot_source(source.raw_root)


def replace_member_payload(
    source: SyntheticRruffSource,
    archive: str,
    source_member: str,
    payload: bytes,
) -> SyntheticRruffSource:
    def mutation(
        members: list[tuple[str, bytes]],
    ) -> list[tuple[str, bytes]]:
        replaced = [
            (name, payload if name == source_member else current_payload)
            for name, current_payload in members
        ]
        if replaced == members:
            raise ValueError(f"member not found: {source_member}")
        return replaced

    return rewrite_archive(source, archive, mutation)


def corrupt_archive_member_crc(
    source: SyntheticRruffSource,
    archive: str,
    source_member: str,
) -> SyntheticRruffSource:
    path = source.raw_root / archive
    with zipfile.ZipFile(path) as current_archive:
        info = next(
            item
            for item in current_archive.infolist()
            if item.filename == source_member
        )
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("CRC corruption helper requires ZIP_STORED")
        header_offset = info.header_offset

    payload = bytearray(path.read_bytes())
    filename_length, extra_length = struct.unpack_from(
        "<HH",
        payload,
        header_offset + 26,
    )
    data_offset = (
        header_offset
        + 30
        + filename_length
        + extra_length
    )
    payload[data_offset] ^= 0x01
    path.write_bytes(payload)
    return _snapshot_source(source.raw_root)


def synthetic_source_contract(
    source: SyntheticRruffSource,
    *,
    expected_rejections: frozenset[
        tuple[str, str, str, str]
    ] = frozenset(),
) -> _RruffSourceContract:
    archives: dict[str, _ArchiveContract] = {}
    for name in sorted(source.archive_members):
        raw_members = 0
        processed_members = 0
        for member in source.archive_members[name]:
            match = _TEST_MEMBER_PATTERN.fullmatch(member)
            if match is None:
                continue
            if match.group("kind") in {"Raw", "RAW"}:
                raw_members += 1
            else:
                processed_members += 1
        archives[name] = _ArchiveContract(
            archive=name,
            bytes=source.archive_bytes[name],
            sha256=source.archive_sha256[name],
            member_count=len(source.archive_members[name]),
            raw_members=raw_members,
            processed_members=processed_members,
        )

    source_raw = sum(item.raw_members for item in archives.values())
    source_processed = sum(
        item.processed_members for item in archives.values()
    )
    rejected_raw = sum(
        kind == "raw" for _, _, kind, _ in expected_rejections
    )
    rejected_processed = sum(
        kind == "processed" for _, _, kind, _ in expected_rejections
    )
    return _RruffSourceContract(
        archives=MappingProxyType(archives),
        source_members=source_raw + source_processed,
        source_raw=source_raw,
        source_processed=source_processed,
        accepted_records=(
            source_raw + source_processed - len(expected_rejections)
        ),
        accepted_raw=source_raw - rejected_raw,
        accepted_processed=source_processed - rejected_processed,
        expected_rejections=expected_rejections,
        expected_statistics=None,
        expected_pair_statistics=None,
    )


def synthetic_pairing_contract(
    source: SyntheticRruffSource,
) -> _RruffSourceContract:
    contract = synthetic_source_contract(
        source,
        expected_rejections=PAIRING_REJECTIONS,
    )
    return _RruffSourceContract(
        archives=contract.archives,
        source_members=contract.source_members,
        source_raw=contract.source_raw,
        source_processed=contract.source_processed,
        accepted_records=contract.accepted_records,
        accepted_raw=contract.accepted_raw,
        accepted_processed=contract.accepted_processed,
        expected_rejections=contract.expected_rejections,
        expected_statistics=None,
        expected_pair_statistics=PAIRING_STATISTICS,
    )
