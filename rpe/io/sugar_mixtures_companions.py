from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pickle
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from rpe.io.schema import axis_id
from rpe.io.sugar_mixtures import (
    DATASET_ID,
    INSTRUMENT,
    _canonical_json_bytes,
    _identity_map_sha256,
    _normalized_acquisition_metadata,
    _revalidate_record_authorities,
    _require_matching_inspection,
    _validate_member_identity,
    _well_target_contract_sha256,
)
from rpe.io.sugar_mixtures_models import (
    SugarMixturesCompanionSummary,
    SugarMixturesInspection,
    SugarMixturesValidationError,
    _SugarAcceptedMember,
)
from rpe.io.sugar_mixtures_source import (
    COMPONENTS,
    _PRODUCTION_SOURCE_CONTRACT,
    _auxiliary_axis_set_id,
    _reparse_verified_member,
    _stored_axis_id,
)


VIEW_SCHEMA_VERSION = "1.0.0"
AUXILIARY_SCHEMA_VERSION = "1.0.0"
AUXILIARY_LOGICAL_DOMAIN = b"rpe-sugar-auxiliary-axes-logical-v1\0"
AUXILIARY_FORMULA = (
    "raman_shift_cm1=1e7/excitation_nm-1e7/wavelength_nm"
)
AUXILIARY_RESIDUAL_CEILINGS = {
    "source_formula_max_abs_residual_cm1": 4.997978066967335e-07,
    "source_formula_mean_abs_residual_cm1": 2.1814789097618358e-07,
    "stored_formula_max_abs_residual_cm1": 0.0006707758566335542,
    "stored_formula_mean_abs_residual_cm1": 0.0002177388297368452,
}
ENDMEMBER_SCHEMA_VERSION = "1.0.0"
ENDMEMBER_LOGICAL_DOMAIN = b"rpe-sugar-derived-endmembers-logical-v1\0"
ENDMEMBER_CONDITIONS = ("high_snr", "low_snr")
ENDMEMBER_PREPARED_MEMBERS = {
    "high_snr": (
        "Raw data/Experimental data from sugar mixtures/"
        "Raw datasets for analyses/High SNR/gt_endmembers.pkl",
        "Raw data/Experimental data from sugar mixtures/"
        "Raw datasets for analyses/High SNR (no refs)/gt_endmembers.pkl",
    ),
    "low_snr": (
        "Raw data/Experimental data from sugar mixtures/"
        "Raw datasets for analyses/Low SNR/gt_endmembers.pkl",
        "Raw data/Experimental data from sugar mixtures/"
        "Raw datasets for analyses/Low SNR (no refs)/gt_endmembers.pkl",
    ),
}
_HDF5_PHYSICAL_POLICY_BYTES = zlib.decompress(
    base64.b85decode(
        "".join(
            (
                "c-p;J%Z}r=65ZdgAhdSdq$KJwb8&$HL6CibYz6~?5-FP*SyCjb+nr$k{hp#^wcYK01PQWev_+NbdFoX0&p=Hl+gL",
                "R`7V2ci;M1Q$Y1-;=RKvHTD@QZj8%?K!sk-CQwA$#Rwv8#uM@RA^&v?GMdfgu$M_Z|;7!N9xN$@GijgDm%Mk=gzq",
                "Pb3DQKqqoQ=LQ+4tQ*|)=636htx8xql)udRG~_BUUD5=8yJkzg=$~`hpmPee9Jjc;yen|JddL=4wK}WR}X`A=K2c",
                "}rBRYZJk9bvk}}J`Zvw5H8jbUrpp5K#dom3@RfnnlhVw5WY<-2Kd+ZH@YpipPxq}s}S~l6%i3AQKczQicL$O!(pz",
                "cj$+Pymz?eXJE>!wpK3U6pV8mGwR;<vGDuM~<n6Bi1}uk1pBfBhXAsF-*^PJ1;JM|*P9V8&ulN7WYB;Mg7{6zqp?",
                ">I>!EKvupk2lU&A7~rJo_P+i|wKJ`|z(Fl|r7K?QOr}v(hFO@VJWQpjDwWDG<*Jks5EY{uX_aSTq(xQN34taO5#g",
                "Zg+-f~ggR`}&TrqTrV)OJoRtIxb#l0Cui^QyTEi8fQ6$dGy08Fo1d{Yo~=A53tMXH<`JTyoQdGg9X1^Yj2A8alUi",
                ">@aO#mH}C9Gn?YzE5tL43%K2YfAe3=mu4NE1W%2JJqTg%~BQUFy4M+O9TQtMh%TMql2wur>SfY!C65!Xp->?UQ^l",
                ")rPZ30ehEe-z;tzuz@WpN5Lv4lHo9(5d&=d#9o&SIwIg6uojiVah|>O{%Eny2Rg>%3g1=3Ge{J^w>@G?3HU`DI@;",
                "6htk#z&3GTL%-GiqjFRcLN!DC!S}<!d&W*O^OImFXQApi<~2_-n2REEI8)r5u%71?vgi;z{O$E}|$-MH)t&OVC#+",
                "FXRX1Y*)ChXs4qYY(?T>8f83{DNkg=rMwdF9$skYJk8@=@GO@}l!+wGFFtriU1Ey{$s*E^4v{>D?}WhoIxdb*3Zv",
                "-!?wOJ_-$t=Wc$CE%k7O*Vx+ur#satDRhgr>~)^VEZx<-GBI<I7sCc2VYEp;N&tb(-Bsn(S+SzPi=OCwWL)?q-UI",
                "jDATiha|SG!UGx)Kbh#s<qUJqjithq$rckLX018xc4aZRzuF52;$*qdmm|1Vsn(dB!Ir8@_)nQ4?Fud(eKnSsK<r",
                "iZ&QypwVE2o%*l?<wtG($hLz@^?rk@`!AQaDpIkZmBxg>2V=L><<idOmpIO_nnb+A2D+1W{IKl|BBXRu8xwK5+|1",
                "J{$Ba!Q2ay25J4Un%dPCf;EE4G|`K$sR}Q=o81LvVOwk0KnzetGeDxoTPTVxDdm5$aRx{2;m^m<Yakbtj8?7|Hi`",
                "GEr6_5h);?m0>9I(C{cSDosU&c`Q<%CaMypP!LGtSg5j0OcnD|m;^KS)=t*7rU1#l+|5KN1hzAd)hx{NEP8)83-d",
                "@wP+XMeQl^RUTta#9WQ$2fSc5&bB%iIxXFdMBETdMV%#ZkGQHn=40$+2BEy6^y<M^40B$RiQKUQ_4V^rk{L#j1*Y",
                "ate;2#^V|`%7tredjxAk+dT_b9nF(x)U3*`Kx>2Ms7Z^3rg|!GU(qENoe$9JG0Z=y<dVg{)N7fla^M7zJnHOjJGz",
                "(b88HyHUk>MTr|mh`{Y}jWL&Bw%K%a>urXE`;5pCJRK+UEGeP%<M8s7U%T%T4szd>Hd5mcl0#+u;{)J6a8|*zt+u",
                "G4@`cJ#G1?GDKKIGTFJgD(tyeI#_W$ACLK<d{zz31;gKqP-#1^hkeS{7lkM`f^L78j!h;6vie#@t#X3x9;Qvm3{v",
                "A39=#!u*cK_r^3W=;XHSqWNAmv|`Q!!nR|NU6*-!11EtWMvU4$w!{`2FcHZxwah(qcZ)W#S)(4oX6HQLQJ$H%0k-",
                "(ebcZSE7$SNEIk6kjvI&(Mm^*-dMu%gXP^QH;-?e0yuy~L?m?wy~BEbZkbBE_}%l#vE8vj1r%5CU*{`bia^N=}Tf",
                "UplVeo>#3^v*4ZsKpnk?=aaPFNUaCY+8aBLM+M-0iaDhf%!~dIe+fBH7D!{zU>uq_iz9at(jwf7&2{ZFwbY!PbDm",
                "`d`uZMXk*r+NUZD}lKf=Gm*!g0MX#?UL`|=3Ylgdv7Qu>QDU=%47FJ;A;D|L}4C1pVew>^onFkU|j5_1mMsBw7nh",
                "Ll3`NyB+&ev!zu=m!S)DzM?cki%YBFE!AsEzHNH-^iSULsy2BnD@{{QLtP^2*LRKKsACB1hKy@kqG3g!`9Z_JMeH",
                "=Sh6GP~~pD=&|LhI7|BRGw%FU8h7)IYtaR}^%6;|5PQBevv|W6bTN-L_S1?!e-J#Ig-dd9|LfYTilQOeJg^bSWY|",
                "n<S1HhTj|(>mUl>6ShpC@;0yT~Jgw`LdskXuFBQ#&)K^@K7-=zAw-_JwB0WZFMWDA?FZ)T^CJu>4j&Q$jE?RnQ&B",
                "xv{i*1r4o$6s#W&FP8^Cy$EYzBa*pEjusOexGIFpKiij|Nfu<0}Z?0+y",
            )
        ).encode("ascii")
    )
)
if (
    len(_HDF5_PHYSICAL_POLICY_BYTES) != 5_149
    or hashlib.sha256(
        b"rpe-sugar-hdf5-physical-contract-v1\0"
        + _HDF5_PHYSICAL_POLICY_BYTES
    ).hexdigest()
    != "ec9facfb64ffec4f786ceac90f76f4fcf85150564a41d16f8b50ed1288efe42e"
):
    raise RuntimeError("embedded Sugar HDF5 physical policy is invalid")
_HDF5_PHYSICAL_POLICY = json.loads(_HDF5_PHYSICAL_POLICY_BYTES)

VIEW_DEFINITIONS = {
    "high_snr": (
        "source_acquisition_condition",
        {"source_acquisition_condition": "high_snr"},
    ),
    "high_snr_no_refs": (
        "condition_role_intersection",
        {
            "source_acquisition_condition": "high_snr",
            "source_record_role": "mixture",
        },
    ),
    "high_snr_pure_reference": (
        "condition_role_intersection",
        {
            "source_acquisition_condition": "high_snr",
            "source_record_role": "pure_reference",
        },
    ),
    "low_snr": (
        "source_acquisition_condition",
        {"source_acquisition_condition": "low_snr"},
    ),
    "low_snr_no_refs": (
        "condition_role_intersection",
        {
            "source_acquisition_condition": "low_snr",
            "source_record_role": "mixture",
        },
    ),
    "low_snr_pure_reference": (
        "condition_role_intersection",
        {
            "source_acquisition_condition": "low_snr",
            "source_record_role": "pure_reference",
        },
    ),
    "no_refs": (
        "source_reference_exclusion",
        {"source_record_role": "mixture"},
    ),
    "pure_reference": (
        "source_record_role",
        {"source_record_role": "pure_reference"},
    ),
}

_VIEW_TOP_KEYS = {
    "artifact_role",
    "artifact_schema_version",
    "dataset_id",
    "views",
}
_VIEW_KEYS = {
    "view_id",
    "semantic_role",
    "selector",
    "train_test_semantics",
    "record_count",
    "record_id_sha256",
    "source_member_path_sha256",
    "record_ids",
}
_ACQUISITION_KEYS = {
    "record_id",
    "source_member",
    "source_member_sha256",
    "source_json_text",
}
_AUXILIARY_ROOT_ATTRS = {
    "artifact_role",
    "artifact_schema_version",
    "dataset_id",
    "logical_content_sha256",
}
_AUXILIARY_METADATA_KEYS = {
    "artifact_role",
    "artifact_schema_version",
    "dataset_id",
    "auxiliary_axis_set_id",
    "core_axis_id",
    "point_count",
    "record_coverage",
    "source_excitation_nm",
    "formula",
    "source_formula_max_abs_residual_cm1",
    "source_formula_mean_abs_residual_cm1",
    "stored_formula_max_abs_residual_cm1",
    "stored_formula_mean_abs_residual_cm1",
    "axes",
}
_AUXILIARY_AXIS_KEYS = {
    "axis_id",
    "coordinate",
    "unit",
    "source_dtype",
    "source_value_sha256",
    "stored_dtype",
    "stored_value_sha256",
    "source_to_stored_max_abs_error",
}
_ENDMEMBER_ROOT_ATTRS = {
    "artifact_role",
    "artifact_schema_version",
    "dataset_id",
    "logical_content_sha256",
}
_ENDMEMBER_METADATA_KEYS = {
    "artifact_role",
    "artifact_schema_version",
    "dataset_id",
    "core_axis_id",
    "point_count",
    "condition_order",
    "component_order",
    "endmember_count",
    "entries",
    "source_semantic_term",
    "stored_semantic_term",
    "prepared_artifacts",
    "direct_zip_recomputation_equals_prepared",
}
_ENDMEMBER_ENTRY_KEYS = {
    "acquisition_condition",
    "component",
    "semantic_role",
    "derivation",
    "constituent_count",
    "constituent_record_ids",
    "constituent_record_id_sha256",
    "constituent_source_member_path_sha256",
    "source_float64_value_sha256",
    "stored_dtype",
    "stored_value_sha256",
    "source_to_stored_max_abs_error",
}
_ENDMEMBER_PREPARED_KEYS = {"source_member", "sha256"}


def _fatal(
    path: str,
    reason: str,
    code: str,
) -> SugarMixturesValidationError:
    return SugarMixturesValidationError(path, reason, code)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_READ_FAILED",
        ) from error
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_WRITE_FAILED",
        ) from error


def _length_prefixed_digest(domain: bytes, values: Sequence[str]) -> str:
    digest = hashlib.sha256(domain)
    for value in sorted(values, key=lambda item: item.encode("utf-8")):
        encoded = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _expected_view_members(
    inspection: SugarMixturesInspection,
) -> Mapping[str, tuple[_SugarAcceptedMember, ...]]:
    try:
        members = tuple(inspection.accepted_members)
        for member in members:
            _validate_member_identity(member)
            try:
                recipe = inspection.recipes[member.well_key]
            except KeyError:
                raise _fatal(
                    f"inspection.accepted_members.{member.record_id}",
                    f"missing recipe {member.well_key}",
                    "VIEWS_INSPECTION_MISMATCH",
                ) from None
            pure_components = tuple(
                component
                for component, volume
                in recipe.component_volumes_ul.items()
                if volume == recipe.total_volume_ul
            )
            expected_role = (
                "pure_reference" if pure_components else "mixture"
            )
            expected_component = (
                pure_components[0] if pure_components else None
            )
            if (
                member.record_role != expected_role
                or member.pure_component != expected_component
            ):
                raise _fatal(
                    f"inspection.accepted_members.{member.record_id}",
                    "record role or pure component disagrees with recipe",
                    "VIEWS_INSPECTION_MISMATCH",
                )
        if _identity_map_sha256(members) != inspection.source_statistics.get(
            "identity_map_sha256"
        ):
            raise _fatal(
                "inspection.accepted_members",
                "identity digest differs from inspection contract",
                "VIEWS_INSPECTION_MISMATCH",
            )
        if _well_target_contract_sha256(
            inspection
        ) != inspection.source_statistics.get(
            "well_target_contract_sha256"
        ):
            raise _fatal(
                "inspection.recipes",
                "well target digest differs from inspection contract",
                "VIEWS_INSPECTION_MISMATCH",
            )
    except SugarMixturesValidationError as error:
        if error.code == "VIEWS_INSPECTION_MISMATCH":
            raise
        raise _fatal(
            error.path,
            error.reason,
            "VIEWS_INSPECTION_MISMATCH",
        ) from error
    expected: dict[str, tuple[_SugarAcceptedMember, ...]] = {}
    for view_id in sorted(
        VIEW_DEFINITIONS,
        key=lambda value: value.encode("utf-8"),
    ):
        selector = VIEW_DEFINITIONS[view_id][1]
        condition = selector.get("source_acquisition_condition")
        role = selector.get("source_record_role")
        selected = tuple(
            member
            for member in inspection.accepted_members
            if (
                condition is None
                or member.condition == condition
            )
            and (
                role is None
                or member.record_role == role
            )
        )
        expected[view_id] = tuple(
            sorted(
                selected,
                key=lambda member: member.record_id.encode("utf-8"),
            )
        )
    return expected


def _validate_view_inspection_cache(
    inspection: SugarMixturesInspection,
    expected: Mapping[str, tuple[_SugarAcceptedMember, ...]],
) -> None:
    if set(inspection.view_record_ids) != set(expected) or set(
        inspection.view_source_members
    ) != set(expected):
        raise _fatal(
            "inspection.views",
            "cached view key sets differ from the fixed view contract",
            "VIEWS_INSPECTION_MISMATCH",
        )
    for view_id, members in expected.items():
        expected_record_ids = tuple(
            member.record_id
            for member in members
        )
        expected_source_members = tuple(
            sorted(
                (member.source_member for member in members),
                key=lambda value: value.encode("utf-8"),
            )
        )
        if (
            tuple(inspection.view_record_ids[view_id])
            != expected_record_ids
            or tuple(inspection.view_source_members[view_id])
            != expected_source_members
        ):
            raise _fatal(
                f"inspection.views.{view_id}",
                "cached view membership differs from independent derivation",
                "VIEWS_INSPECTION_MISMATCH",
            )


def _view_document(
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    expected = _expected_view_members(inspection)
    _validate_view_inspection_cache(inspection, expected)
    views = []
    for view_id in sorted(
        expected,
        key=lambda value: value.encode("utf-8"),
    ):
        members = expected[view_id]
        record_ids = tuple(member.record_id for member in members)
        source_members = tuple(
            sorted(
                (member.source_member for member in members),
                key=lambda value: value.encode("utf-8"),
            )
        )
        semantic_role, selector = VIEW_DEFINITIONS[view_id]
        views.append(
            {
                "view_id": view_id,
                "semantic_role": semantic_role,
                "selector": selector,
                "train_test_semantics": False,
                "record_count": len(record_ids),
                "record_id_sha256": _length_prefixed_digest(
                    b"rpe-sugar-view-record-ids-v1\0",
                    record_ids,
                ),
                "source_member_path_sha256": _length_prefixed_digest(
                    b"rpe-sugar-view-source-members-v1\0",
                    source_members,
                ),
                "record_ids": list(record_ids),
            }
        )
    return {
        "artifact_role": "record_views",
        "artifact_schema_version": VIEW_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "views": views,
    }


def _write_views(
    path: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    path = Path(path)
    payload = _canonical_json_bytes(
        _view_document(inspection),
        newline=True,
    )
    _atomic_write_bytes(path, payload)
    return SugarMixturesCompanionSummary(
        path=path,
        artifact_role="record_views",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        logical_content_sha256=None,
        item_count=len(VIEW_DEFINITIONS),
    )


def _read_canonical_json(path: Path) -> Mapping[str, object]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _fatal(
            str(path),
            str(error),
            "VIEWS_JSON_NONCANONICAL",
        ) from error
    if not isinstance(value, Mapping):
        raise _fatal(
            str(path),
            "views document must be an object",
            "VIEWS_KEYS_INVALID",
        )
    try:
        canonical = _canonical_json_bytes(value, newline=True)
    except (TypeError, ValueError) as error:
        raise _fatal(
            str(path),
            str(error),
            "VIEWS_JSON_NONCANONICAL",
        ) from error
    if payload != canonical:
        raise _fatal(
            str(path),
            "views document is not canonical finite JSON",
            "VIEWS_JSON_NONCANONICAL",
        )
    return value


def _validate_view_relations(
    observed: Mapping[str, set[str]],
) -> None:
    if (
        observed["high_snr"] & observed["low_snr"]
        or observed["high_snr"] | observed["low_snr"]
        != observed["pure_reference"] | observed["no_refs"]
        or observed["pure_reference"] & observed["no_refs"]
        or (
            observed["high_snr_pure_reference"]
            != observed["high_snr"] & observed["pure_reference"]
        )
        or (
            observed["high_snr_no_refs"]
            != observed["high_snr"] & observed["no_refs"]
        )
        or (
            observed["low_snr_pure_reference"]
            != observed["low_snr"] & observed["pure_reference"]
        )
        or (
            observed["low_snr_no_refs"]
            != observed["low_snr"] & observed["no_refs"]
        )
    ):
        raise _fatal(
            "views.relations",
            "view relations are not disjoint and exhaustive",
            "VIEWS_RELATION_INVALID",
        )


def _validate_views(
    path: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    expected_core_ids = {
        member.record_id
        for member in inspection.accepted_members
    }
    if record_ids != expected_core_ids:
        raise _fatal(
            "views.record_ids",
            "core record ID set differs from inspection",
            "VIEWS_RECORD_IDS_INVALID",
        )
    independently_derived = _expected_view_members(inspection)
    _validate_view_inspection_cache(
        inspection,
        independently_derived,
    )
    document = _read_canonical_json(path)
    if set(document) != _VIEW_TOP_KEYS:
        raise _fatal(
            "views",
            "top-level key set differs from contract",
            "VIEWS_KEYS_INVALID",
        )
    if (
        document["artifact_role"] != "record_views"
        or document["artifact_schema_version"] != VIEW_SCHEMA_VERSION
        or document["dataset_id"] != DATASET_ID
        or not isinstance(document["views"], list)
    ):
        raise _fatal(
            "views",
            "fixed views scalars are invalid",
            "VIEWS_KEYS_INVALID",
        )
    views = document["views"]
    observed_ids = tuple(
        view.get("view_id")
        if isinstance(view, Mapping)
        else None
        for view in views
    )
    expected_order = tuple(
        sorted(VIEW_DEFINITIONS, key=lambda value: value.encode("utf-8"))
    )
    if observed_ids != expected_order:
        raise _fatal(
            "views",
            f"expected view order {expected_order}, observed {observed_ids}",
            "VIEWS_ORDER_INVALID",
        )
    expected_members = independently_derived
    member_by_record_id = {
        member.record_id: member
        for member in inspection.accepted_members
    }
    observed_sets = {}
    counts = {}
    memberships = 0
    for index, observed in enumerate(views):
        path_prefix = f"views[{index}]"
        if not isinstance(observed, Mapping) or set(observed) != _VIEW_KEYS:
            raise _fatal(
                path_prefix,
                "view key set differs from contract",
                "VIEWS_KEYS_INVALID",
            )
        view_id = observed["view_id"]
        semantic_role, selector = VIEW_DEFINITIONS[view_id]
        if (
            observed["semantic_role"] != semantic_role
            or observed["selector"] != selector
        ):
            raise _fatal(
                path_prefix,
                "selector or semantic role differs from contract",
                "VIEWS_SELECTOR_INVALID",
            )
        if observed["train_test_semantics"] is not False:
            raise _fatal(
                path_prefix,
                "views must not have train/test semantics",
                "VIEWS_SEMANTICS_INVALID",
            )
        observed_record_ids = observed["record_ids"]
        if not isinstance(observed_record_ids, list) or any(
            not isinstance(record_id, str)
            for record_id in observed_record_ids
        ):
            raise _fatal(
                path_prefix,
                "record_ids must be an array of strings",
                "VIEWS_RECORD_IDS_INVALID",
            )
        expected_record_ids = tuple(
            member.record_id
            for member in expected_members[view_id]
        )
        if tuple(observed_record_ids) != expected_record_ids:
            raise _fatal(
                path_prefix,
                "stored IDs differ from independently rederived selector",
                "VIEWS_RECORD_IDS_INVALID",
            )
        if not set(observed_record_ids) <= record_ids:
            raise _fatal(
                path_prefix,
                "view contains an unknown core record ID",
                "VIEWS_RECORD_IDS_INVALID",
            )
        if (
            isinstance(observed["record_count"], bool)
            or observed["record_count"] != len(expected_record_ids)
        ):
            raise _fatal(
                path_prefix,
                "record_count differs from rederived membership",
                "VIEWS_COUNT_INVALID",
            )
        expected_record_digest = _length_prefixed_digest(
            b"rpe-sugar-view-record-ids-v1\0",
            expected_record_ids,
        )
        source_members = tuple(
            sorted(
                (
                    member_by_record_id[record_id].source_member
                    for record_id in expected_record_ids
                ),
                key=lambda value: value.encode("utf-8"),
            )
        )
        expected_source_digest = _length_prefixed_digest(
            b"rpe-sugar-view-source-members-v1\0",
            source_members,
        )
        if (
            observed["record_id_sha256"] != expected_record_digest
            or observed["source_member_path_sha256"]
            != expected_source_digest
        ):
            raise _fatal(
                path_prefix,
                "view digest differs from rederived membership",
                "VIEWS_DIGEST_INVALID",
            )
        observed_sets[view_id] = set(observed_record_ids)
        counts[view_id] = len(observed_record_ids)
        memberships += len(observed_record_ids)
    _validate_view_relations(observed_sets)
    return {
        "view_count": len(views),
        "view_counts": counts,
        "view_memberships": memberships,
        "bytes": Path(path).stat().st_size,
        "sha256": _sha256_file(Path(path)),
    }


def _write_acquisition_json(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    path = Path(path)
    _revalidate_record_authorities(
        Path(raw_root).resolve(),
        inspection,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    digest = hashlib.sha256()
    total_bytes = 0
    lines = 0
    raw_inventory = hashlib.sha256(
        b"rpe-sugar-acquisition-json-text-v1\0"
    )
    archive_path = (
        Path(raw_root).resolve()
        / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with zipfile.ZipFile(archive_path) as archive:
                infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
                for info in archive.infolist():
                    infos_by_name.setdefault(info.filename, []).append(info)
                for member in inspection.accepted_members:
                    matches = infos_by_name.get(member.source_member, [])
                    if len(matches) != 1:
                        raise _fatal(
                            f"acquisition_json.{member.source_member}",
                            (
                                "expected one member binding, "
                                f"observed {len(matches)}"
                            ),
                            "MEMBER_BINDING_MISMATCH",
                        )
                    parsed = _reparse_verified_member(
                        raw_root,
                        member,
                        archive=archive,
                        info=matches[0],
                        expected_points=int(
                            inspection.source_statistics[
                                "points_per_record"
                            ]
                        ),
                    )
                    line = _canonical_json_bytes(
                        {
                            "record_id": member.record_id,
                            "source_member": member.source_member,
                            "source_member_sha256": member.sha256,
                            "source_json_text": parsed.source_json_text,
                        },
                        newline=True,
                    )
                    output.write(line)
                    digest.update(line)
                    total_bytes += len(line)
                    lines += 1
                    source_member = member.source_member.encode("utf-8")
                    raw_text = parsed.source_json_text.encode("utf-8")
                    raw_inventory.update(
                        struct.pack("<Q", len(source_member))
                    )
                    raw_inventory.update(source_member)
                    raw_inventory.update(struct.pack("<Q", len(raw_text)))
                    raw_inventory.update(raw_text)
                    del parsed
            expected_raw_digest = inspection.source_statistics.get(
                "raw_text_inventory_sha256"
            )
            if (
                expected_raw_digest
                and raw_inventory.hexdigest() != expected_raw_digest
            ):
                raise _fatal(
                    "acquisition_json.raw_text_inventory_sha256",
                    (
                        f"expected {expected_raw_digest}, "
                        f"observed {raw_inventory.hexdigest()}"
                    ),
                    "ACQUISITION_JSONL_TEXT_MISMATCH",
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except SugarMixturesValidationError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_WRITE_FAILED",
        ) from error
    return SugarMixturesCompanionSummary(
        path=path,
        artifact_role="acquisition_json_text",
        bytes=total_bytes,
        sha256=digest.hexdigest(),
        logical_content_sha256=None,
        item_count=lines,
    )


def _parse_jsonl_line(
    line: bytes,
    *,
    index: int,
) -> Mapping[str, object]:
    path = f"acquisition_json[{index}]"
    if not line.endswith(b"\n"):
        raise _fatal(
            path,
            "every JSONL record must end with one newline",
            "ACQUISITION_JSONL_NEWLINE_INVALID",
        )
    try:
        value = json.loads(
            line,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _fatal(
            path,
            str(error),
            "ACQUISITION_JSONL_NONCANONICAL",
        ) from error
    if not isinstance(value, Mapping):
        raise _fatal(
            path,
            "JSONL line must be an object",
            "ACQUISITION_JSONL_KEYS_INVALID",
        )
    if line != _canonical_json_bytes(value, newline=True):
        raise _fatal(
            path,
            "JSONL line is not canonical finite JSON",
            "ACQUISITION_JSONL_NONCANONICAL",
        )
    if set(value) != _ACQUISITION_KEYS:
        raise _fatal(
            path,
            "JSONL line key set differs from contract",
            "ACQUISITION_JSONL_KEYS_INVALID",
        )
    return value


def _validate_acquisition_json(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    path = Path(path)
    _revalidate_record_authorities(
        Path(raw_root).resolve(),
        inspection,
    )
    members = tuple(inspection.accepted_members)
    expected_ids = tuple(member.record_id for member in members)
    if set(expected_ids) != record_ids:
        raise _fatal(
            "acquisition_json.record_ids",
            "core record IDs differ from inspection",
            "ACQUISITION_JSONL_RECORD_IDS_INVALID",
        )
    if tuple(member.source_member for member in members) != tuple(
        sorted(
            (member.source_member for member in members),
            key=lambda value: value.encode("utf-8"),
        )
    ):
        raise _fatal(
            "inspection.accepted_members",
            "source member order differs from metadata digest contract",
            "INSPECTION_IDENTITY_MISMATCH",
        )
    archive_path = (
        Path(raw_root).resolve()
        / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    line_count = 0
    file_digest = hashlib.sha256()
    raw_inventory = hashlib.sha256(
        b"rpe-sugar-acquisition-json-text-v1\0"
    )
    normalized_inventory = hashlib.sha256(
        b"rpe-sugar-acquisition-metadata-v1\0"
    )
    core_inventory = hashlib.sha256(
        b"rpe-sugar-core-acquisition-metadata-v1\0"
    )
    raw_total = 0
    minimum = None
    maximum = None
    unique_text_sha256 = set()
    unique_normalized_sha256 = set()
    source_nonfinite_values = 0

    def count_source_nonfinite(value: object) -> int:
        if isinstance(value, Mapping):
            if value == {"source_nonfinite_number": "NaN"}:
                return 1
            return sum(
                count_source_nonfinite(item)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return sum(count_source_nonfinite(item) for item in value)
        return 0

    try:
        with path.open("rb") as source, zipfile.ZipFile(
            archive_path
        ) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for line in source:
                file_digest.update(line)
                document = _parse_jsonl_line(
                    line,
                    index=line_count,
                )
                if line_count >= len(members):
                    raise _fatal(
                        "acquisition_json.record_ids",
                        "JSONL has more lines than canonical records",
                        "ACQUISITION_JSONL_RECORD_IDS_INVALID",
                    )
                member = members[line_count]
                if document["record_id"] != member.record_id:
                    code = (
                        "ACQUISITION_JSONL_ORDER_INVALID"
                        if document["record_id"] in record_ids
                        else "ACQUISITION_JSONL_RECORD_IDS_INVALID"
                    )
                    raise _fatal(
                        f"acquisition_json[{line_count}].record_id",
                        (
                            f"expected {member.record_id!r}, "
                            f"observed {document['record_id']!r}"
                        ),
                        code,
                    )
                if (
                    document["source_member"] != member.source_member
                    or document["source_member_sha256"] != member.sha256
                ):
                    raise _fatal(
                        f"acquisition_json[{line_count}]",
                        "source member binding differs from inspection",
                        "ACQUISITION_JSONL_MEMBER_MISMATCH",
                    )
                matches = infos_by_name.get(member.source_member, [])
                if len(matches) != 1:
                    raise _fatal(
                        f"acquisition_json[{line_count}]",
                        "source member binding is missing or duplicated",
                        "MEMBER_BINDING_MISMATCH",
                    )
                parsed = _reparse_verified_member(
                    raw_root,
                    member,
                    archive=archive,
                    info=matches[0],
                    expected_points=int(
                        inspection.source_statistics["points_per_record"]
                    ),
                )
                if document["source_json_text"] != parsed.source_json_text:
                    raise _fatal(
                        f"acquisition_json[{line_count}].source_json_text",
                        "exact decoded source JSON differs from ZIP member",
                        "ACQUISITION_JSONL_TEXT_MISMATCH",
                    )
                normalized = _normalized_acquisition_metadata(
                    member,
                    parsed.source_metadata,
                )
                if (
                    float(normalized["excitation_nm"]) != 785.0
                    or int(normalized["n_accumulations"]) != 1
                    or normalized["integration_time_s"]
                    != (5.0 if member.condition == "high_snr" else 0.5)
                    or INSTRUMENT
                    != "B-Raman custom Raman microspectroscopy platform"
                ):
                    raise _fatal(
                        f"acquisition_json[{line_count}]",
                        "normalized/core metadata reconstruction failed",
                        "ACQUISITION_JSONL_METADATA_MISMATCH",
                    )
                normalized_bytes = _canonical_json_bytes(
                    normalized,
                    newline=False,
                )
                unique_normalized_sha256.add(
                    hashlib.sha256(normalized_bytes).digest()
                )
                source_nonfinite_values += count_source_nonfinite(
                    normalized
                )
                core_bytes = _canonical_json_bytes(
                    {
                        "instrument": INSTRUMENT,
                        "excitation_nm": float(normalized["excitation_nm"]),
                        "integration_time_s": float(
                            normalized["integration_time_s"]
                        ),
                        "n_accumulations": int(
                            normalized["n_accumulations"]
                        ),
                        "grating": None,
                        "detector": None,
                    },
                    newline=False,
                )
                raw_text = parsed.source_json_text.encode("utf-8")
                source_member = member.source_member.encode("utf-8")
                for digest, payload in (
                    (raw_inventory, raw_text),
                    (normalized_inventory, normalized_bytes),
                    (core_inventory, core_bytes),
                ):
                    digest.update(struct.pack("<Q", len(source_member)))
                    digest.update(source_member)
                    digest.update(struct.pack("<Q", len(payload)))
                    digest.update(payload)
                raw_length = len(raw_text)
                raw_total += raw_length
                minimum = (
                    raw_length
                    if minimum is None
                    else min(minimum, raw_length)
                )
                maximum = (
                    raw_length
                    if maximum is None
                    else max(maximum, raw_length)
                )
                unique_text_sha256.add(hashlib.sha256(raw_text).digest())
                line_count += 1
                del parsed
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_READ_FAILED",
        ) from error
    if line_count != len(members):
        raise _fatal(
            "acquisition_json.record_ids",
            f"expected {len(members)} lines, observed {line_count}",
            "ACQUISITION_JSONL_RECORD_IDS_INVALID",
        )
    expected_raw_digest = inspection.source_statistics.get(
        "raw_text_inventory_sha256"
    )
    observed_raw_digest = raw_inventory.hexdigest()
    if (
        expected_raw_digest
        and observed_raw_digest != expected_raw_digest
    ):
        raise _fatal(
            "acquisition_json.raw_text_inventory_sha256",
            (
                f"expected {expected_raw_digest}, "
                f"observed {observed_raw_digest}"
            ),
            "ACQUISITION_JSONL_TEXT_MISMATCH",
        )
    for name, digest in (
        ("normalized_metadata_sha256", normalized_inventory),
        ("core_metadata_sha256", core_inventory),
    ):
        expected = inspection.source_statistics.get(name)
        observed = digest.hexdigest()
        if expected and observed != expected:
            raise _fatal(
                f"acquisition_json.{name}",
                f"expected {expected}, observed {observed}",
                "ACQUISITION_JSONL_METADATA_MISMATCH",
            )
    return {
        "lines": line_count,
        "bytes": path.stat().st_size,
        "sha256": file_digest.hexdigest(),
        "decoded_source_json_bytes": raw_total,
        "minimum_source_json_bytes": minimum,
        "maximum_source_json_bytes": maximum,
        "unique_source_json_texts": len(unique_text_sha256),
        "unique_normalized_metadata_objects": len(
            unique_normalized_sha256
        ),
        "source_nonfinite_values": source_nonfinite_values,
        "raw_text_inventory_sha256": observed_raw_digest,
    }


def _axis_value_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _max_abs_error(
    source: np.ndarray,
    stored: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(source, dtype=np.float64)
                - np.asarray(stored, dtype=np.float64)
            )
        )
    )


def _auxiliary_source_values(
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    members = tuple(inspection.accepted_members)
    statistics = inspection.source_statistics
    expected_points = statistics.get("points_per_record")
    if (
        not members
        or isinstance(expected_points, bool)
        or not isinstance(expected_points, int)
        or expected_points <= 0
        or statistics.get("record_count") != len(members)
        or statistics.get("axis_groups") != 1
        or _identity_map_sha256(members)
        != statistics.get("identity_map_sha256")
    ):
        raise _fatal(
            "inspection.auxiliary_axes",
            "record inventory differs from inspected axis contract",
            "AUXILIARY_INSPECTION_MISMATCH",
        )
    for member in members:
        try:
            _validate_member_identity(member)
        except SugarMixturesValidationError as error:
            raise _fatal(
                error.path,
                error.reason,
                "AUXILIARY_INSPECTION_MISMATCH",
            ) from error

    parsed = _reparse_verified_member(
        inspection.raw_root,
        members[0],
        expected_points=expected_points,
    )
    source_pixel = np.ascontiguousarray(parsed.pixel, dtype="<f8")
    source_wavelength = np.ascontiguousarray(
        parsed.wavelength_nm,
        dtype="<f8",
    )
    source_wavenumber = np.ascontiguousarray(
        parsed.wavenumber_cm1,
        dtype="<f8",
    )
    stored_pixel = np.ascontiguousarray(source_pixel, dtype="<u2")
    stored_wavelength = np.ascontiguousarray(
        source_wavelength,
        dtype="<f4",
    )
    stored_wavenumber = np.ascontiguousarray(
        source_wavenumber,
        dtype="<f4",
    )
    if not np.array_equal(
        stored_pixel.astype(np.float64),
        source_pixel,
    ):
        raise _fatal(
            "inspection.auxiliary_axes.pixel",
            "source Pixel is not exactly representable as uint16",
            "AUXILIARY_INSPECTION_MISMATCH",
        )

    pixel_identifier = _stored_axis_id(
        b"rpe-sugar-pixel-axis-v1\0",
        stored_pixel,
        "<u2",
    )
    wavelength_identifier = _stored_axis_id(
        b"rpe-sugar-wavelength-axis-v1\0",
        stored_wavelength,
        "<f4",
    )
    core_identifier = axis_id(stored_wavenumber)
    set_identifier = _auxiliary_axis_set_id(
        core_identifier,
        pixel_identifier,
        wavelength_identifier,
    )
    source_hashes = {
        "pixel_float64_sha256": _axis_value_sha256(
            source_pixel,
            "<f8",
        ),
        "wavelength_float64_sha256": _axis_value_sha256(
            source_wavelength,
            "<f8",
        ),
        "wavenumber_float64_sha256": _axis_value_sha256(
            source_wavenumber,
            "<f8",
        ),
    }
    wavelength_error = _max_abs_error(
        source_wavelength,
        stored_wavelength,
    )
    wavenumber_error = _max_abs_error(
        source_wavenumber,
        stored_wavenumber,
    )
    expected_contract = {
        "core_axis_id": core_identifier,
        "pixel_axis_id": pixel_identifier,
        "wavelength_axis_id": wavelength_identifier,
        "auxiliary_axis_set_id": set_identifier,
        **source_hashes,
        "wavelength_float32_max_abs_error_nm": wavelength_error,
        "axis_float32_max_abs_error_cm1": wavenumber_error,
    }
    for name, expected in expected_contract.items():
        if (
            statistics.get(name) != expected
            or (
                hasattr(inspection, name)
                and getattr(inspection, name) != expected
            )
        ):
            raise _fatal(
                f"inspection.{name}",
                (
                    f"expected independently derived {expected!r}, "
                    f"observed {statistics.get(name)!r}"
                ),
                "AUXILIARY_INSPECTION_MISMATCH",
            )

    try:
        excitation_nm = float(
            parsed.source_metadata["Excitation wavelength [nm]"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _fatal(
            "inspection.auxiliary_axes.source_excitation_nm",
            "source excitation is unavailable",
            "AUXILIARY_INSPECTION_MISMATCH",
        ) from error
    if not np.isfinite(excitation_nm) or excitation_nm != 785.0:
        raise _fatal(
            "inspection.auxiliary_axes.source_excitation_nm",
            f"expected 785.0, observed {excitation_nm!r}",
            "AUXILIARY_INSPECTION_MISMATCH",
        )

    source_formula = (
        1.0e7 / excitation_nm
        - 1.0e7 / source_wavelength
    )
    source_residual = np.abs(source_wavenumber - source_formula)
    stored_formula = (
        1.0e7 / 785.0
        - 1.0e7 / stored_wavelength.astype(np.float64)
    )
    stored_residual = np.abs(
        stored_wavenumber.astype(np.float64) - stored_formula
    )
    residuals = {
        "source_formula_max_abs_residual_cm1": float(
            np.max(source_residual)
        ),
        "source_formula_mean_abs_residual_cm1": float(
            np.mean(source_residual)
        ),
        "stored_formula_max_abs_residual_cm1": float(
            np.max(stored_residual)
        ),
        "stored_formula_mean_abs_residual_cm1": float(
            np.mean(stored_residual)
        ),
    }
    for name, observed in residuals.items():
        if observed > AUXILIARY_RESIDUAL_CEILINGS[name]:
            raise _fatal(
                f"inspection.auxiliary_axes.{name}",
                (
                    f"observed residual {observed!r} exceeds "
                    f"{AUXILIARY_RESIDUAL_CEILINGS[name]!r}"
                ),
                "AUXILIARY_FORMULA_RESIDUAL_INVALID",
            )
    metadata = {
        "artifact_role": "auxiliary_axes",
        "artifact_schema_version": AUXILIARY_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "auxiliary_axis_set_id": set_identifier,
        "core_axis_id": core_identifier,
        "point_count": expected_points,
        "record_coverage": len(members),
        "source_excitation_nm": excitation_nm,
        "formula": AUXILIARY_FORMULA,
        **residuals,
        "axes": {
            "pixel": {
                "axis_id": pixel_identifier,
                "coordinate": "pixel",
                "unit": "detector_pixel_index",
                "source_dtype": "float64",
                "source_value_sha256": source_hashes[
                    "pixel_float64_sha256"
                ],
                "stored_dtype": "uint16",
                "stored_value_sha256": _axis_value_sha256(
                    stored_pixel,
                    "<u2",
                ),
                "source_to_stored_max_abs_error": _max_abs_error(
                    source_pixel,
                    stored_pixel,
                ),
            },
            "wavelength": {
                "axis_id": wavelength_identifier,
                "coordinate": "wavelength",
                "unit": "nm",
                "source_dtype": "float64",
                "source_value_sha256": source_hashes[
                    "wavelength_float64_sha256"
                ],
                "stored_dtype": "float32",
                "stored_value_sha256": _axis_value_sha256(
                    stored_wavelength,
                    "<f4",
                ),
                "source_to_stored_max_abs_error": wavelength_error,
            },
        },
    }
    return {
        "metadata": metadata,
        "metadata_bytes": _canonical_json_bytes(
            metadata,
            newline=True,
        ),
        "pixel": stored_pixel,
        "wavelength": stored_wavelength,
    }


def _auxiliary_logical_sha256(
    metadata_bytes: bytes,
    pixel: np.ndarray,
    wavelength: np.ndarray,
) -> str:
    digest = hashlib.sha256(AUXILIARY_LOGICAL_DOMAIN)
    for payload in (
        metadata_bytes,
        np.ascontiguousarray(pixel, dtype="<u2").tobytes(order="C"),
        np.ascontiguousarray(
            wavelength,
            dtype="<f4",
        ).tobytes(order="C"),
    ):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _auxiliary_dataset_options(length: int) -> Mapping[str, object]:
    return {
        "chunks": (length,),
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "fletcher32": True,
        "track_times": False,
    }


def _write_auxiliary_axes(
    path: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    path = Path(path)
    values = _auxiliary_source_values(inspection)
    metadata_bytes = values["metadata_bytes"]
    pixel = values["pixel"]
    wavelength = values["wavelength"]
    logical_sha256 = _auxiliary_logical_sha256(
        metadata_bytes,
        pixel,
        wavelength,
    )
    metadata_array = np.frombuffer(metadata_bytes, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with h5py.File(
            temporary_path,
            "w",
            libver="earliest",
            track_order=False,
            track_times=False,
        ) as artifact:
            artifact.attrs["artifact_role"] = "auxiliary_axes"
            artifact.attrs[
                "artifact_schema_version"
            ] = AUXILIARY_SCHEMA_VERSION
            artifact.attrs["dataset_id"] = DATASET_ID
            artifact.attrs[
                "logical_content_sha256"
            ] = logical_sha256
            artifact.create_dataset(
                "metadata_json",
                data=metadata_array,
                **_auxiliary_dataset_options(metadata_array.size),
            )
            axes = artifact.create_group(
                "axes",
                track_order=False,
                track_times=False,
            )
            axes.create_dataset(
                "pixel",
                data=pixel,
                **_auxiliary_dataset_options(pixel.size),
            )
            axes.create_dataset(
                "wavelength_nm",
                data=wavelength,
                **_auxiliary_dataset_options(wavelength.size),
            )
        with temporary_path.open("rb") as completed:
            os.fsync(completed.fileno())
        os.replace(temporary_path, path)
    except SugarMixturesValidationError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_WRITE_FAILED",
        ) from error
    return SugarMixturesCompanionSummary(
        path=path,
        artifact_role="auxiliary_axes",
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        logical_content_sha256=logical_sha256,
        item_count=1,
    )


def _require_auxiliary_dataset(
    dataset: h5py.Dataset,
    *,
    path: str,
    dtype: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    creation = dataset.id.get_create_plist()
    filters = tuple(
        creation.get_filter(index)[0]
        for index in range(creation.get_nfilters())
    )
    if (
        dataset.dtype.str != dtype
        or dataset.shape != shape
        or dataset.chunks != chunks
        or dataset.maxshape != shape
        or dataset.compression != "gzip"
        or dataset.compression_opts != 4
        or dataset.shuffle is not True
        or dataset.fletcher32 is not True
        or dataset.scaleoffset is not None
        or dataset.is_virtual
        or creation.get_layout() != h5py.h5d.CHUNKED
        or creation.get_external_count() != 0
        or set(filters)
        != {
            h5py.h5z.FILTER_DEFLATE,
            h5py.h5z.FILTER_SHUFFLE,
            h5py.h5z.FILTER_FLETCHER32,
        }
        or len(filters) != 3
        or set(dataset.attrs)
        or creation.get_attr_creation_order() != 0
    ):
        raise _fatal(
            path,
            "dataset physical layout differs from contract",
            "AUXILIARY_STRUCTURE_INVALID",
        )


def _require_hard_link(group: h5py.Group, name: str, path: str) -> None:
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise _fatal(
            path,
            "only direct hard-linked objects are permitted",
            "AUXILIARY_STRUCTURE_INVALID",
        )


def _require_group_creation_order_disabled(
    group: h5py.Group,
    path: str,
    *,
    code: str = "AUXILIARY_STRUCTURE_INVALID",
) -> None:
    creation = group.id.get_create_plist()
    if (
        creation.get_link_creation_order() != 0
        or creation.get_attr_creation_order() != 0
    ):
        raise _fatal(
            path,
            "group creation-order tracking must be disabled",
            code,
        )


def _validate_auxiliary_metadata_shape(
    metadata: object,
) -> None:
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != _AUXILIARY_METADATA_KEYS
        or not isinstance(metadata.get("axes"), Mapping)
        or set(metadata["axes"]) != {"pixel", "wavelength"}
    ):
        raise _fatal(
            "auxiliary.metadata_json",
            "metadata key set differs from contract",
            "AUXILIARY_METADATA_INVALID",
        )
    for name in ("pixel", "wavelength"):
        axis_metadata = metadata["axes"][name]
        if (
            not isinstance(axis_metadata, Mapping)
            or set(axis_metadata) != _AUXILIARY_AXIS_KEYS
        ):
            raise _fatal(
                f"auxiliary.metadata_json.axes.{name}",
                "axis metadata key set differs from contract",
                "AUXILIARY_METADATA_INVALID",
            )


def _independent_auxiliary_expectations(
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    members = tuple(inspection.accepted_members)
    statistics = inspection.source_statistics
    point_count = statistics.get("points_per_record")
    if (
        not members
        or isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count <= 0
        or statistics.get("record_count") != len(members)
        or statistics.get("axis_groups") != 1
        or _identity_map_sha256(members)
        != statistics.get("identity_map_sha256")
    ):
        raise _fatal(
            "inspection.auxiliary_axes",
            "record inventory differs from inspected axis contract",
            "AUXILIARY_INSPECTION_MISMATCH",
        )
    for member in members:
        try:
            _validate_member_identity(member)
        except SugarMixturesValidationError as error:
            raise _fatal(
                error.path,
                error.reason,
                "AUXILIARY_INSPECTION_MISMATCH",
            ) from error

    parsed = _reparse_verified_member(
        inspection.raw_root,
        members[0],
        expected_points=point_count,
    )
    source_pixel = np.asarray(parsed.pixel, dtype=np.float64)
    source_wavelength = np.asarray(
        parsed.wavelength_nm,
        dtype=np.float64,
    )
    source_wavenumber = np.asarray(
        parsed.wavenumber_cm1,
        dtype=np.float64,
    )
    pixel = np.ascontiguousarray(source_pixel, dtype="<u2")
    wavelength = np.ascontiguousarray(source_wavelength, dtype="<f4")
    wavenumber = np.ascontiguousarray(source_wavenumber, dtype="<f4")
    if not np.array_equal(pixel.astype(np.float64), source_pixel):
        raise _fatal(
            "inspection.auxiliary_axes.pixel",
            "source Pixel is not exactly representable as uint16",
            "AUXILIARY_INSPECTION_MISMATCH",
        )

    pixel_bytes = pixel.tobytes(order="C")
    wavelength_bytes = wavelength.tobytes(order="C")
    pixel_id_digest = hashlib.sha256(b"rpe-sugar-pixel-axis-v1\0")
    pixel_id_digest.update(struct.pack("<Q", pixel.size))
    pixel_id_digest.update(pixel_bytes)
    pixel_identifier = pixel_id_digest.hexdigest()
    wavelength_id_digest = hashlib.sha256(
        b"rpe-sugar-wavelength-axis-v1\0"
    )
    wavelength_id_digest.update(struct.pack("<Q", wavelength.size))
    wavelength_id_digest.update(wavelength_bytes)
    wavelength_identifier = wavelength_id_digest.hexdigest()
    core_identifier = axis_id(wavenumber)
    set_digest = hashlib.sha256(b"rpe-sugar-aux-axis-set-v1\0")
    for identifier in (
        core_identifier,
        pixel_identifier,
        wavelength_identifier,
    ):
        encoded = identifier.encode("utf-8")
        set_digest.update(struct.pack("<Q", len(encoded)))
        set_digest.update(encoded)
    set_identifier = set_digest.hexdigest()

    source_pixel_bytes = np.ascontiguousarray(
        source_pixel,
        dtype="<f8",
    ).tobytes(order="C")
    source_wavelength_bytes = np.ascontiguousarray(
        source_wavelength,
        dtype="<f8",
    ).tobytes(order="C")
    source_wavenumber_bytes = np.ascontiguousarray(
        source_wavenumber,
        dtype="<f8",
    ).tobytes(order="C")
    source_hashes = {
        "pixel_float64_sha256": hashlib.sha256(
            source_pixel_bytes
        ).hexdigest(),
        "wavelength_float64_sha256": hashlib.sha256(
            source_wavelength_bytes
        ).hexdigest(),
        "wavenumber_float64_sha256": hashlib.sha256(
            source_wavenumber_bytes
        ).hexdigest(),
    }
    wavelength_error = float(
        np.max(
            np.abs(
                source_wavelength
                - wavelength.astype(np.float64)
            )
        )
    )
    wavenumber_error = float(
        np.max(
            np.abs(
                source_wavenumber
                - wavenumber.astype(np.float64)
            )
        )
    )
    inspection_contract = {
        "core_axis_id": core_identifier,
        "pixel_axis_id": pixel_identifier,
        "wavelength_axis_id": wavelength_identifier,
        "auxiliary_axis_set_id": set_identifier,
        **source_hashes,
        "wavelength_float32_max_abs_error_nm": wavelength_error,
        "axis_float32_max_abs_error_cm1": wavenumber_error,
    }
    for name, independently_derived in inspection_contract.items():
        if (
            statistics.get(name) != independently_derived
            or (
                hasattr(inspection, name)
                and getattr(inspection, name) != independently_derived
            )
        ):
            raise _fatal(
                f"inspection.{name}",
                (
                    f"expected independently derived "
                    f"{independently_derived!r}, observed "
                    f"{statistics.get(name)!r}"
                ),
                "AUXILIARY_INSPECTION_MISMATCH",
            )

    try:
        excitation_nm = float(
            parsed.source_metadata["Excitation wavelength [nm]"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _fatal(
            "inspection.auxiliary_axes.source_excitation_nm",
            "source excitation is unavailable",
            "AUXILIARY_INSPECTION_MISMATCH",
        ) from error
    if not np.isfinite(excitation_nm) or excitation_nm != 785.0:
        raise _fatal(
            "inspection.auxiliary_axes.source_excitation_nm",
            f"expected 785.0, observed {excitation_nm!r}",
            "AUXILIARY_INSPECTION_MISMATCH",
        )
    source_formula = (
        1.0e7 / excitation_nm
        - 1.0e7 / source_wavelength
    )
    stored_formula = (
        1.0e7 / 785.0
        - 1.0e7 / wavelength.astype(np.float64)
    )
    source_residual = np.abs(source_wavenumber - source_formula)
    stored_residual = np.abs(
        wavenumber.astype(np.float64) - stored_formula
    )
    residuals = {
        "source_formula_max_abs_residual_cm1": float(
            np.max(source_residual)
        ),
        "source_formula_mean_abs_residual_cm1": float(
            np.mean(source_residual)
        ),
        "stored_formula_max_abs_residual_cm1": float(
            np.max(stored_residual)
        ),
        "stored_formula_mean_abs_residual_cm1": float(
            np.mean(stored_residual)
        ),
    }
    for name, observed in residuals.items():
        if observed > AUXILIARY_RESIDUAL_CEILINGS[name]:
            raise _fatal(
                f"inspection.auxiliary_axes.{name}",
                (
                    f"observed residual {observed!r} exceeds "
                    f"{AUXILIARY_RESIDUAL_CEILINGS[name]!r}"
                ),
                "AUXILIARY_FORMULA_RESIDUAL_INVALID",
            )

    metadata = {
        "artifact_role": "auxiliary_axes",
        "artifact_schema_version": AUXILIARY_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "auxiliary_axis_set_id": set_identifier,
        "core_axis_id": core_identifier,
        "point_count": point_count,
        "record_coverage": len(members),
        "source_excitation_nm": excitation_nm,
        "formula": AUXILIARY_FORMULA,
        **residuals,
        "axes": {
            "pixel": {
                "axis_id": pixel_identifier,
                "coordinate": "pixel",
                "unit": "detector_pixel_index",
                "source_dtype": "float64",
                "source_value_sha256": source_hashes[
                    "pixel_float64_sha256"
                ],
                "stored_dtype": "uint16",
                "stored_value_sha256": hashlib.sha256(
                    pixel_bytes
                ).hexdigest(),
                "source_to_stored_max_abs_error": float(
                    np.max(
                        np.abs(
                            source_pixel
                            - pixel.astype(np.float64)
                        )
                    )
                ),
            },
            "wavelength": {
                "axis_id": wavelength_identifier,
                "coordinate": "wavelength",
                "unit": "nm",
                "source_dtype": "float64",
                "source_value_sha256": source_hashes[
                    "wavelength_float64_sha256"
                ],
                "stored_dtype": "float32",
                "stored_value_sha256": hashlib.sha256(
                    wavelength_bytes
                ).hexdigest(),
                "source_to_stored_max_abs_error": wavelength_error,
            },
        },
    }
    metadata_bytes = _canonical_json_bytes(metadata, newline=True)
    logical_digest = hashlib.sha256(AUXILIARY_LOGICAL_DOMAIN)
    for payload in (
        metadata_bytes,
        pixel_bytes,
        wavelength_bytes,
    ):
        logical_digest.update(struct.pack("<Q", len(payload)))
        logical_digest.update(payload)
    return {
        "metadata": metadata,
        "metadata_bytes": metadata_bytes,
        "pixel": pixel,
        "wavelength": wavelength,
        "logical_content_sha256": logical_digest.hexdigest(),
    }


def _validate_auxiliary_axes(
    path: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    path = Path(path)
    expected = _independent_auxiliary_expectations(inspection)
    expected_metadata = expected["metadata"]
    expected_metadata_bytes = expected["metadata_bytes"]
    expected_pixel = expected["pixel"]
    expected_wavelength = expected["wavelength"]
    expected_logical = expected["logical_content_sha256"]
    with h5py.File(path, "r") as artifact:
        _require_group_creation_order_disabled(
            artifact["/"],
            "auxiliary",
        )
        if set(artifact) != {"metadata_json", "axes"}:
            raise _fatal(
                "auxiliary",
                "root object set differs from contract",
                "AUXILIARY_STRUCTURE_INVALID",
            )
        for name in ("metadata_json", "axes"):
            _require_hard_link(artifact, name, f"auxiliary.{name}")
        if set(artifact.attrs) != _AUXILIARY_ROOT_ATTRS:
            raise _fatal(
                "auxiliary.attrs",
                "root attribute set differs from contract",
                "AUXILIARY_STRUCTURE_INVALID",
            )
        fixed_attrs = {
            "artifact_role": "auxiliary_axes",
            "artifact_schema_version": AUXILIARY_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
        }
        for name, expected_value in fixed_attrs.items():
            if artifact.attrs[name] != expected_value:
                raise _fatal(
                    f"auxiliary.attrs.{name}",
                    "root attribute differs from contract",
                    "AUXILIARY_STRUCTURE_INVALID",
                )

        metadata_dataset = artifact["metadata_json"]
        axes = artifact["axes"]
        _require_group_creation_order_disabled(
            axes,
            "auxiliary.axes",
        )
        if (
            not isinstance(metadata_dataset, h5py.Dataset)
            or not isinstance(axes, h5py.Group)
            or set(axes) != {"pixel", "wavelength_nm"}
            or set(axes.attrs)
        ):
            raise _fatal(
                "auxiliary",
                "group or dataset set differs from contract",
                "AUXILIARY_STRUCTURE_INVALID",
            )
        for name in ("pixel", "wavelength_nm"):
            _require_hard_link(
                axes,
                name,
                f"auxiliary.axes.{name}",
            )
            if not isinstance(axes[name], h5py.Dataset):
                raise _fatal(
                    f"auxiliary.axes.{name}",
                    "axis object must be a dataset",
                    "AUXILIARY_STRUCTURE_INVALID",
                )
        _require_auxiliary_dataset(
            metadata_dataset,
            path="auxiliary.metadata_json",
            dtype="|u1",
            shape=(len(expected_metadata_bytes),),
            chunks=(len(expected_metadata_bytes),),
        )
        _require_auxiliary_dataset(
            axes["pixel"],
            path="auxiliary.axes.pixel",
            dtype="<u2",
            shape=expected_pixel.shape,
            chunks=expected_pixel.shape,
        )
        _require_auxiliary_dataset(
            axes["wavelength_nm"],
            path="auxiliary.axes.wavelength_nm",
            dtype="<f4",
            shape=expected_wavelength.shape,
            chunks=expected_wavelength.shape,
        )

        observed_metadata_bytes = metadata_dataset[...].tobytes(order="C")
        try:
            observed_metadata = json.loads(
                observed_metadata_bytes,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise _fatal(
                "auxiliary.metadata_json",
                str(error),
                "AUXILIARY_METADATA_INVALID",
            ) from error
        _validate_auxiliary_metadata_shape(observed_metadata)
        if (
            observed_metadata_bytes
            != _canonical_json_bytes(
                observed_metadata,
                newline=True,
            )
            or observed_metadata != expected_metadata
        ):
            raise _fatal(
                "auxiliary.metadata_json",
                "metadata differs from independently derived source contract",
                "AUXILIARY_METADATA_INVALID",
            )

        observed_pixel = axes["pixel"][...]
        observed_wavelength = axes["wavelength_nm"][...]
        if (
            not np.array_equal(observed_pixel, expected_pixel)
            or not np.array_equal(
                observed_wavelength,
                expected_wavelength,
            )
        ):
            raise _fatal(
                "auxiliary.axes",
                "stored axis values differ from source-derived arrays",
                "AUXILIARY_AXIS_INVALID",
            )
        observed_digest = hashlib.sha256(AUXILIARY_LOGICAL_DOMAIN)
        for payload in (
            observed_metadata_bytes,
            np.ascontiguousarray(
                observed_pixel,
                dtype="<u2",
            ).tobytes(order="C"),
            np.ascontiguousarray(
                observed_wavelength,
                dtype="<f4",
            ).tobytes(order="C"),
        ):
            observed_digest.update(struct.pack("<Q", len(payload)))
            observed_digest.update(payload)
        observed_logical = observed_digest.hexdigest()
        if (
            observed_logical != expected_logical
            or artifact.attrs["logical_content_sha256"]
            != observed_logical
        ):
            raise _fatal(
                "auxiliary.logical_content_sha256",
                "logical digest differs from canonical content",
                "AUXILIARY_LOGICAL_DIGEST_INVALID",
            )
    return {
        "artifact_role": "auxiliary_axes",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "logical_content_sha256": expected_logical,
        "auxiliary_axis_set_id": expected_metadata[
            "auxiliary_axis_set_id"
        ],
        "core_axis_id": expected_metadata["core_axis_id"],
        "pixel_axis_id": expected_metadata["axes"]["pixel"]["axis_id"],
        "wavelength_axis_id": expected_metadata["axes"]["wavelength"][
            "axis_id"
        ],
        "point_count": expected_metadata["point_count"],
        "record_coverage": expected_metadata["record_coverage"],
        "metadata_json_bytes": len(expected_metadata_bytes),
    }


def _endmember_constituent_groups(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, tuple[_SugarAcceptedMember, ...]]:
    _require_matching_inspection(Path(raw_root).resolve(), inspection)
    members = tuple(inspection.accepted_members)
    if (
        _identity_map_sha256(members)
        != inspection.source_statistics.get("identity_map_sha256")
        or sum(member.condition == "high_snr" for member in members)
        != inspection.source_statistics.get("high_records")
        or sum(member.condition == "low_snr" for member in members)
        != inspection.source_statistics.get("low_records")
        or len(inspection.recipes)
        != inspection.source_statistics.get("recipe_count")
    ):
        raise _fatal(
            "inspection.accepted_members",
            "identity map differs from inspection contract",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    groups: dict[str, list[_SugarAcceptedMember]] = {
        f"{condition}/{component}": []
        for condition in ENDMEMBER_CONDITIONS
        for component in COMPONENTS
    }
    for member in members:
        try:
            _validate_member_identity(member)
            recipe = inspection.recipes[member.well_key]
        except (SugarMixturesValidationError, KeyError) as error:
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                "member identity or recipe binding is invalid",
                "ENDMEMBER_INSPECTION_MISMATCH",
            ) from error
        pure_components = tuple(
            component
            for component, volume in recipe.component_volumes_ul.items()
            if volume == recipe.total_volume_ul
        )
        expected_role = "pure_reference" if pure_components else "mixture"
        expected_component = pure_components[0] if pure_components else None
        if (
            member.record_role != expected_role
            or member.pure_component != expected_component
        ):
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                "record role or component differs from source recipe",
                "ENDMEMBER_INSPECTION_MISMATCH",
            )
        if expected_component is not None:
            groups[f"{member.condition}/{expected_component}"].append(member)

    expected_keys = set(groups)
    if (
        set(inspection.endmember_constituent_record_ids) != expected_keys
        or set(inspection.endmember_constituent_source_members)
        != expected_keys
    ):
        raise _fatal(
            "inspection.endmember_constituents",
            "constituent cache keys differ from fixed condition/component set",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    result = {}
    for condition in ENDMEMBER_CONDITIONS:
        condition_counts = []
        for component in COMPONENTS:
            key = f"{condition}/{component}"
            ordered = tuple(
                sorted(
                    groups[key],
                    key=lambda member: member.record_id.encode("utf-8"),
                )
            )
            if not ordered:
                raise _fatal(
                    f"inspection.endmember_constituents.{key}",
                    "constituent group is empty",
                    "ENDMEMBER_INSPECTION_MISMATCH",
                )
            record_ids = tuple(member.record_id for member in ordered)
            source_members = tuple(
                sorted(
                    (member.source_member for member in ordered),
                    key=lambda value: value.encode("utf-8"),
                )
            )
            if (
                tuple(inspection.endmember_constituent_record_ids[key])
                != record_ids
                or tuple(
                    inspection.endmember_constituent_source_members[key]
                )
                != source_members
            ):
                raise _fatal(
                    f"inspection.endmember_constituents.{key}",
                    "cached constituents differ from source-derived grouping",
                    "ENDMEMBER_INSPECTION_MISMATCH",
                )
            condition_counts.append(len(ordered))
            result[key] = ordered
        if len(set(condition_counts)) != 1:
            raise _fatal(
                f"inspection.endmember_constituents.{condition}",
                "component constituent counts differ within condition",
                "ENDMEMBER_INSPECTION_MISMATCH",
            )
    if sum(len(group) for group in result.values()) != int(
        inspection.source_statistics["pure_reference_records"]
    ) or (
        len(members) - sum(len(group) for group in result.values())
        != inspection.source_statistics.get("mixture_records")
    ):
        raise _fatal(
            "inspection.endmember_constituents",
            "constituent total differs from pure-reference record count",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    return result


def _read_prepared_endmember_evidence(
    archive: zipfile.ZipFile,
    inspection: SugarMixturesInspection,
    expected: Mapping[str, np.ndarray],
) -> Mapping[str, list[Mapping[str, str]]]:
    evidence = {
        member.source_member: member
        for member in inspection.relevant_evidence_members
        if member.role == "prepared"
    }
    infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        infos_by_name.setdefault(info.filename, []).append(info)
    prepared_artifacts = {}
    for condition in ENDMEMBER_CONDITIONS:
        artifacts = []
        hashes = []
        for source_member in ENDMEMBER_PREPARED_MEMBERS[condition]:
            member = evidence.get(source_member)
            matches = infos_by_name.get(source_member, ())
            if member is None or len(matches) != 1:
                raise _fatal(
                    f"endmembers.prepared_artifacts.{source_member}",
                    "prepared evidence binding is missing or duplicated",
                    "ENDMEMBER_PREPARED_INVALID",
                )
            info = matches[0]
            try:
                payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
                raise _fatal(
                    f"endmembers.prepared_artifacts.{source_member}",
                    str(error),
                    "ENDMEMBER_PREPARED_INVALID",
                ) from error
            observed_hash = hashlib.sha256(payload).hexdigest()
            if (
                len(payload) != member.bytes
                or info.CRC != member.crc32
                or observed_hash != member.sha256
            ):
                raise _fatal(
                    f"endmembers.prepared_artifacts.{source_member}",
                    "prepared member differs from verified evidence binding",
                    "ENDMEMBER_PREPARED_INVALID",
                )
            try:
                prepared = pd.read_pickle(io.BytesIO(payload))
                prepared_values = np.asarray(prepared)
            except (
                AttributeError,
                EOFError,
                ImportError,
                ModuleNotFoundError,
                pickle.UnpicklingError,
                ValueError,
            ) as error:
                raise _fatal(
                    f"endmembers.prepared_artifacts.{source_member}",
                    str(error),
                    "ENDMEMBER_PREPARED_INVALID",
                ) from error
            if (
                prepared_values.dtype.kind not in {"f", "i", "u"}
                or prepared_values.shape != expected[condition].shape
                or not np.all(np.isfinite(prepared_values))
                or not np.array_equal(
                    prepared_values.astype(np.float64),
                    expected[condition],
                )
            ):
                raise _fatal(
                    f"endmembers.prepared_artifacts.{source_member}",
                    "prepared values differ from direct-ZIP recomputation",
                    "ENDMEMBER_PREPARED_INVALID",
                )
            hashes.append(observed_hash)
            artifacts.append(
                {
                    "source_member": source_member,
                    "sha256": observed_hash,
                }
            )
        if len(set(hashes)) != 1:
            raise _fatal(
                f"endmembers.prepared_artifacts.{condition}",
                "full and no-refs prepared hashes differ",
                "ENDMEMBER_PREPARED_INVALID",
            )
        prepared_artifacts[condition] = artifacts
    return prepared_artifacts


def _derive_endmember_content(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    groups = _endmember_constituent_groups(raw_root, inspection)
    point_count = int(inspection.source_statistics["points_per_record"])
    archive_path = (
        Path(raw_root).resolve()
        / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    entries = []
    source_arrays = []
    stored_arrays = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for condition in ENDMEMBER_CONDITIONS:
                for component in COMPONENTS:
                    key = f"{condition}/{component}"
                    constituents = groups[key]
                    intensities = []
                    for member in constituents:
                        matches = infos_by_name.get(member.source_member, ())
                        if len(matches) != 1:
                            raise _fatal(
                                f"endmembers.constituents.{member.source_member}",
                                "constituent binding is missing or duplicated",
                                "ENDMEMBER_INSPECTION_MISMATCH",
                            )
                        parsed = _reparse_verified_member(
                            raw_root,
                            member,
                            archive=archive,
                            info=matches[0],
                            expected_points=point_count,
                        )
                        intensities.append(parsed.intensity)
                    source = np.ascontiguousarray(
                        np.median(
                            np.stack(intensities, axis=0),
                            axis=0,
                        ),
                        dtype="<f8",
                    )
                    stored = np.ascontiguousarray(source, dtype="<f4")
                    cast_error = _max_abs_error(source, stored)
                    if cast_error != 0.0:
                        raise _fatal(
                            f"endmembers.values.{key}",
                            f"float32 cast error is {cast_error!r}",
                            "ENDMEMBER_VALUE_INVALID",
                        )
                    record_ids = tuple(
                        member.record_id for member in constituents
                    )
                    source_members = tuple(
                        member.source_member for member in constituents
                    )
                    entries.append(
                        {
                            "acquisition_condition": condition,
                            "component": component,
                            "semantic_role": (
                                "derived_reference_endmember"
                            ),
                            "derivation": "pointwise_median",
                            "constituent_count": len(constituents),
                            "constituent_record_ids": list(record_ids),
                            "constituent_record_id_sha256": (
                                _length_prefixed_digest(
                                    (
                                        b"rpe-sugar-endmember-constituent-"
                                        b"record-ids-v1\0"
                                    ),
                                    record_ids,
                                )
                            ),
                            "constituent_source_member_path_sha256": (
                                _length_prefixed_digest(
                                    (
                                        b"rpe-sugar-endmember-constituent-"
                                        b"source-members-v1\0"
                                    ),
                                    source_members,
                                )
                            ),
                            "source_float64_value_sha256": (
                                _axis_value_sha256(source, "<f8")
                            ),
                            "stored_dtype": "float32",
                            "stored_value_sha256": (
                                _axis_value_sha256(stored, "<f4")
                            ),
                            "source_to_stored_max_abs_error": cast_error,
                        }
                    )
                    source_arrays.append(source)
                    stored_arrays.append(stored)
            prepared_expected = {
                condition: np.stack(
                    source_arrays[index * len(COMPONENTS) : (
                        index + 1
                    ) * len(COMPONENTS)],
                    axis=0,
                )
                for index, condition in enumerate(ENDMEMBER_CONDITIONS)
            }
            prepared_artifacts = _read_prepared_endmember_evidence(
                archive,
                inspection,
                prepared_expected,
            )
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "endmembers.source",
            str(error),
            "COMPANION_READ_FAILED",
        ) from error

    metadata = {
        "artifact_role": "derived_reference_endmembers",
        "artifact_schema_version": ENDMEMBER_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "core_axis_id": inspection.core_axis_id,
        "point_count": point_count,
        "condition_order": list(ENDMEMBER_CONDITIONS),
        "component_order": list(COMPONENTS),
        "endmember_count": len(stored_arrays),
        "entries": entries,
        "source_semantic_term": "gt_endmembers",
        "stored_semantic_term": "derived_reference_endmember",
        "prepared_artifacts": prepared_artifacts,
        "direct_zip_recomputation_equals_prepared": True,
    }
    metadata_bytes = _canonical_json_bytes(metadata, newline=True)
    intensity = np.ascontiguousarray(
        np.stack(stored_arrays, axis=0).reshape(
            len(ENDMEMBER_CONDITIONS),
            len(COMPONENTS),
            point_count,
        ),
        dtype="<f4",
    )
    return {
        "metadata": metadata,
        "metadata_bytes": metadata_bytes,
        "source_arrays": tuple(source_arrays),
        "stored_arrays": tuple(stored_arrays),
        "intensity": intensity,
    }


def _endmember_logical_sha256(
    metadata_bytes: bytes,
    stored_arrays: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256(ENDMEMBER_LOGICAL_DOMAIN)
    digest.update(struct.pack("<Q", len(metadata_bytes)))
    digest.update(metadata_bytes)
    for array in stored_arrays:
        payload = np.ascontiguousarray(
            array,
            dtype="<f4",
        ).tobytes(order="C")
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _write_derived_endmembers(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    path = Path(path)
    content = _derive_endmember_content(raw_root, inspection)
    metadata_bytes = content["metadata_bytes"]
    intensity = content["intensity"]
    logical_sha256 = _endmember_logical_sha256(
        metadata_bytes,
        content["stored_arrays"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with h5py.File(
            temporary_path,
            "w",
            libver="earliest",
            track_order=False,
            track_times=False,
        ) as artifact:
            artifact.attrs[
                "artifact_role"
            ] = "derived_reference_endmembers"
            artifact.attrs[
                "artifact_schema_version"
            ] = ENDMEMBER_SCHEMA_VERSION
            artifact.attrs["dataset_id"] = DATASET_ID
            artifact.attrs[
                "logical_content_sha256"
            ] = logical_sha256
            metadata_array = np.frombuffer(metadata_bytes, dtype=np.uint8)
            artifact.create_dataset(
                "metadata_json",
                data=metadata_array,
                **_auxiliary_dataset_options(metadata_array.size),
            )
            artifact.create_dataset(
                "intensity",
                data=intensity,
                chunks=(1, 1, intensity.shape[-1]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )
        with temporary_path.open("rb") as completed:
            os.fsync(completed.fileno())
        os.replace(temporary_path, path)
    except SugarMixturesValidationError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise _fatal(
            str(path),
            str(error),
            "COMPANION_WRITE_FAILED",
        ) from error
    return SugarMixturesCompanionSummary(
        path=path,
        artifact_role="derived_reference_endmembers",
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        logical_content_sha256=logical_sha256,
        item_count=len(ENDMEMBER_CONDITIONS) * len(COMPONENTS),
    )


def _require_endmember_dataset(
    dataset: h5py.Dataset,
    *,
    path: str,
    dtype: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> None:
    creation = dataset.id.get_create_plist()
    filters = tuple(
        creation.get_filter(index)[0]
        for index in range(creation.get_nfilters())
    )
    if (
        dataset.dtype.str != dtype
        or dataset.shape != shape
        or dataset.chunks != chunks
        or dataset.maxshape != shape
        or dataset.compression != "gzip"
        or dataset.compression_opts != 4
        or dataset.shuffle is not True
        or dataset.fletcher32 is not True
        or dataset.scaleoffset is not None
        or dataset.is_virtual
        or creation.get_layout() != h5py.h5d.CHUNKED
        or creation.get_external_count() != 0
        or set(filters)
        != {
            h5py.h5z.FILTER_DEFLATE,
            h5py.h5z.FILTER_SHUFFLE,
            h5py.h5z.FILTER_FLETCHER32,
        }
        or len(filters) != 3
        or set(dataset.attrs)
        or creation.get_attr_creation_order() != 0
    ):
        raise _fatal(
            path,
            "dataset physical layout differs from contract",
            "ENDMEMBER_STRUCTURE_INVALID",
        )


def _validate_endmember_metadata_shape(metadata: object) -> None:
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != _ENDMEMBER_METADATA_KEYS
        or not isinstance(metadata.get("entries"), list)
        or not isinstance(metadata.get("prepared_artifacts"), Mapping)
        or set(metadata["prepared_artifacts"])
        != set(ENDMEMBER_CONDITIONS)
    ):
        raise _fatal(
            "endmembers.metadata_json",
            "metadata key set differs from contract",
            "ENDMEMBER_METADATA_INVALID",
        )
    for index, entry in enumerate(metadata["entries"]):
        if not isinstance(entry, Mapping) or set(entry) != _ENDMEMBER_ENTRY_KEYS:
            raise _fatal(
                f"endmembers.metadata_json.entries[{index}]",
                "entry key set differs from contract",
                "ENDMEMBER_METADATA_INVALID",
            )
    for condition in ENDMEMBER_CONDITIONS:
        artifacts = metadata["prepared_artifacts"][condition]
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != 2
            or any(
                not isinstance(artifact, Mapping)
                or set(artifact) != _ENDMEMBER_PREPARED_KEYS
                for artifact in artifacts
            )
        ):
            raise _fatal(
                f"endmembers.metadata_json.prepared_artifacts.{condition}",
                "prepared artifact schema differs from contract",
                "ENDMEMBER_METADATA_INVALID",
            )


def _endmember_metadata_error_code(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> str:
    if (
        observed.get("prepared_artifacts")
        != expected.get("prepared_artifacts")
        or observed.get("direct_zip_recomputation_equals_prepared")
        is not True
    ):
        return "ENDMEMBER_PREPARED_INVALID"
    observed_entries = observed.get("entries")
    expected_entries = expected.get("entries")
    if isinstance(observed_entries, list) and isinstance(expected_entries, list):
        for observed_entry, expected_entry in zip(
            observed_entries,
            expected_entries,
        ):
            if not isinstance(observed_entry, Mapping):
                continue
            for key in (
                "constituent_count",
                "constituent_record_ids",
                "constituent_record_id_sha256",
                "constituent_source_member_path_sha256",
            ):
                if observed_entry.get(key) != expected_entry.get(key):
                    return "ENDMEMBER_LINEAGE_INVALID"
            for key in (
                "source_float64_value_sha256",
                "stored_value_sha256",
                "source_to_stored_max_abs_error",
            ):
                if observed_entry.get(key) != expected_entry.get(key):
                    return "ENDMEMBER_VALUE_INVALID"
    return "ENDMEMBER_METADATA_INVALID"


def _independent_endmember_expectations(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    resolved_root = Path(raw_root).resolve()
    statistics = inspection.source_statistics
    members = tuple(inspection.accepted_members)
    if inspection.raw_root != resolved_root:
        raise _fatal(
            "inspection.raw_root",
            "inspection belongs to a different raw root",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    archive_path = resolved_root / _PRODUCTION_SOURCE_CONTRACT.archive_name
    try:
        archive_bytes = archive_path.stat().st_size
    except OSError as error:
        raise _fatal(
            "inspection.archive",
            str(error),
            "ENDMEMBER_INSPECTION_MISMATCH",
        ) from error
    if (
        archive_bytes != statistics.get("archive_bytes")
        or statistics.get("record_count") != len(members)
        or statistics.get("axis_groups") != 1
        or sum(member.condition == "high_snr" for member in members)
        != statistics.get("high_records")
        or sum(member.condition == "low_snr" for member in members)
        != statistics.get("low_records")
        or len(inspection.recipes) != statistics.get("recipe_count")
    ):
        raise _fatal(
            "inspection.endmembers",
            "archive or record inventory differs from inspection",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    identity_digest = hashlib.sha256(b"rpe-sugar-identity-map-v1\0")
    groups: dict[str, list[_SugarAcceptedMember]] = {
        f"{condition}/{component}": []
        for condition in ENDMEMBER_CONDITIONS
        for component in COMPONENTS
    }
    for member in members:
        try:
            _validate_member_identity(member)
            recipe = inspection.recipes[member.well_key]
        except (SugarMixturesValidationError, KeyError) as error:
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                "member identity or recipe binding is invalid",
                "ENDMEMBER_INSPECTION_MISMATCH",
            ) from error
        for value in (
            member.record_id,
            member.sample_id,
            member.acquisition_id,
            member.source_member,
        ):
            encoded = value.encode("utf-8")
            identity_digest.update(struct.pack("<Q", len(encoded)))
            identity_digest.update(encoded)
        pure_components = tuple(
            component
            for component, volume in recipe.component_volumes_ul.items()
            if volume == recipe.total_volume_ul
        )
        expected_role = "pure_reference" if pure_components else "mixture"
        expected_component = pure_components[0] if pure_components else None
        if (
            member.record_role != expected_role
            or member.pure_component != expected_component
        ):
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                "record role or component differs from source recipe",
                "ENDMEMBER_INSPECTION_MISMATCH",
            )
        if expected_component is not None:
            groups[f"{member.condition}/{expected_component}"].append(member)
    if identity_digest.hexdigest() != statistics.get("identity_map_sha256"):
        raise _fatal(
            "inspection.accepted_members",
            "identity digest differs from inspection",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )

    expected_keys = set(groups)
    if (
        set(inspection.endmember_constituent_record_ids) != expected_keys
        or set(inspection.endmember_constituent_source_members)
        != expected_keys
    ):
        raise _fatal(
            "inspection.endmember_constituents",
            "constituent cache keys differ from contract",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )
    ordered_groups = {}
    for condition in ENDMEMBER_CONDITIONS:
        counts = []
        for component in COMPONENTS:
            key = f"{condition}/{component}"
            ordered = tuple(
                sorted(
                    groups[key],
                    key=lambda member: member.record_id.encode("utf-8"),
                )
            )
            if not ordered:
                raise _fatal(
                    f"inspection.endmember_constituents.{key}",
                    "constituent group is empty",
                    "ENDMEMBER_INSPECTION_MISMATCH",
                )
            record_ids = tuple(member.record_id for member in ordered)
            source_members = tuple(
                sorted(
                    (member.source_member for member in ordered),
                    key=lambda value: value.encode("utf-8"),
                )
            )
            if (
                tuple(inspection.endmember_constituent_record_ids[key])
                != record_ids
                or tuple(
                    inspection.endmember_constituent_source_members[key]
                )
                != source_members
            ):
                raise _fatal(
                    f"inspection.endmember_constituents.{key}",
                    "cached constituents differ from source derivation",
                    "ENDMEMBER_INSPECTION_MISMATCH",
                )
            counts.append(len(ordered))
            ordered_groups[key] = ordered
        if len(set(counts)) != 1:
            raise _fatal(
                f"inspection.endmember_constituents.{condition}",
                "component counts differ within condition",
                "ENDMEMBER_INSPECTION_MISMATCH",
            )
    constituent_total = sum(
        len(group) for group in ordered_groups.values()
    )
    if (
        constituent_total != statistics.get("pure_reference_records")
        or len(members) - constituent_total
        != statistics.get("mixture_records")
    ):
        raise _fatal(
            "inspection.endmember_constituents",
            "constituent total differs from pure-reference count",
            "ENDMEMBER_INSPECTION_MISMATCH",
        )

    point_count = int(statistics["points_per_record"])
    entries = []
    source_arrays = []
    stored_arrays = []
    evidence = {
        member.source_member: member
        for member in inspection.relevant_evidence_members
        if member.role == "prepared"
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for condition in ENDMEMBER_CONDITIONS:
                for component in COMPONENTS:
                    key = f"{condition}/{component}"
                    constituents = ordered_groups[key]
                    values = []
                    for member in constituents:
                        matches = infos_by_name.get(member.source_member, ())
                        if len(matches) != 1:
                            raise _fatal(
                                (
                                    "endmembers.constituents."
                                    f"{member.source_member}"
                                ),
                                "constituent binding is missing or duplicated",
                                "ENDMEMBER_INSPECTION_MISMATCH",
                            )
                        parsed = _reparse_verified_member(
                            raw_root,
                            member,
                            archive=archive,
                            info=matches[0],
                            expected_points=point_count,
                        )
                        values.append(parsed.intensity)
                    source = np.ascontiguousarray(
                        np.median(np.stack(values, axis=0), axis=0),
                        dtype="<f8",
                    )
                    stored = np.ascontiguousarray(source, dtype="<f4")
                    cast_error = float(
                        np.max(
                            np.abs(
                                source
                                - stored.astype(np.float64)
                            )
                        )
                    )
                    if cast_error != 0.0:
                        raise _fatal(
                            f"endmembers.values.{key}",
                            f"float32 cast error is {cast_error!r}",
                            "ENDMEMBER_VALUE_INVALID",
                        )
                    record_ids = tuple(
                        member.record_id for member in constituents
                    )
                    source_members = tuple(
                        member.source_member for member in constituents
                    )
                    record_digest = hashlib.sha256(
                        (
                            b"rpe-sugar-endmember-constituent-"
                            b"record-ids-v1\0"
                        )
                    )
                    for value in sorted(
                        record_ids,
                        key=lambda item: item.encode("utf-8"),
                    ):
                        encoded = value.encode("utf-8")
                        record_digest.update(
                            struct.pack("<Q", len(encoded))
                        )
                        record_digest.update(encoded)
                    source_member_digest = hashlib.sha256(
                        (
                            b"rpe-sugar-endmember-constituent-"
                            b"source-members-v1\0"
                        )
                    )
                    for value in sorted(
                        source_members,
                        key=lambda item: item.encode("utf-8"),
                    ):
                        encoded = value.encode("utf-8")
                        source_member_digest.update(
                            struct.pack("<Q", len(encoded))
                        )
                        source_member_digest.update(encoded)
                    source_bytes = source.tobytes(order="C")
                    stored_bytes = stored.tobytes(order="C")
                    entries.append(
                        {
                            "acquisition_condition": condition,
                            "component": component,
                            "semantic_role": (
                                "derived_reference_endmember"
                            ),
                            "derivation": "pointwise_median",
                            "constituent_count": len(constituents),
                            "constituent_record_ids": list(record_ids),
                            "constituent_record_id_sha256": (
                                record_digest.hexdigest()
                            ),
                            "constituent_source_member_path_sha256": (
                                source_member_digest.hexdigest()
                            ),
                            "source_float64_value_sha256": (
                                hashlib.sha256(source_bytes).hexdigest()
                            ),
                            "stored_dtype": "float32",
                            "stored_value_sha256": (
                                hashlib.sha256(stored_bytes).hexdigest()
                            ),
                            "source_to_stored_max_abs_error": cast_error,
                        }
                    )
                    source_arrays.append(source)
                    stored_arrays.append(stored)

            prepared_artifacts = {}
            for condition_index, condition in enumerate(
                ENDMEMBER_CONDITIONS
            ):
                expected_prepared = np.stack(
                    source_arrays[
                        condition_index * len(COMPONENTS) :
                        (condition_index + 1) * len(COMPONENTS)
                    ],
                    axis=0,
                )
                artifacts = []
                hashes = []
                for source_member in ENDMEMBER_PREPARED_MEMBERS[condition]:
                    member = evidence.get(source_member)
                    matches = infos_by_name.get(source_member, ())
                    if member is None or len(matches) != 1:
                        raise _fatal(
                            (
                                "endmembers.prepared_artifacts."
                                f"{source_member}"
                            ),
                            "prepared binding is missing or duplicated",
                            "ENDMEMBER_PREPARED_INVALID",
                        )
                    info = matches[0]
                    payload = archive.read(info)
                    observed_hash = hashlib.sha256(payload).hexdigest()
                    if (
                        len(payload) != member.bytes
                        or info.CRC != member.crc32
                        or observed_hash != member.sha256
                    ):
                        raise _fatal(
                            (
                                "endmembers.prepared_artifacts."
                                f"{source_member}"
                            ),
                            "prepared binding differs from inspection",
                            "ENDMEMBER_PREPARED_INVALID",
                        )
                    try:
                        prepared = np.asarray(
                            pd.read_pickle(io.BytesIO(payload))
                        )
                    except (
                        AttributeError,
                        EOFError,
                        ImportError,
                        ModuleNotFoundError,
                        pickle.UnpicklingError,
                        ValueError,
                    ) as error:
                        raise _fatal(
                            (
                                "endmembers.prepared_artifacts."
                                f"{source_member}"
                            ),
                            str(error),
                            "ENDMEMBER_PREPARED_INVALID",
                        ) from error
                    if (
                        prepared.dtype.kind not in {"f", "i", "u"}
                        or prepared.shape != expected_prepared.shape
                        or not np.all(np.isfinite(prepared))
                        or not np.array_equal(
                            prepared.astype(np.float64),
                            expected_prepared,
                        )
                    ):
                        raise _fatal(
                            (
                                "endmembers.prepared_artifacts."
                                f"{source_member}"
                            ),
                            "prepared values differ from source medians",
                            "ENDMEMBER_PREPARED_INVALID",
                        )
                    hashes.append(observed_hash)
                    artifacts.append(
                        {
                            "source_member": source_member,
                            "sha256": observed_hash,
                        }
                    )
                if len(set(hashes)) != 1:
                    raise _fatal(
                        f"endmembers.prepared_artifacts.{condition}",
                        "full and no-refs prepared hashes differ",
                        "ENDMEMBER_PREPARED_INVALID",
                    )
                prepared_artifacts[condition] = artifacts
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        raise _fatal(
            "endmembers.source",
            str(error),
            "COMPANION_READ_FAILED",
        ) from error

    metadata = {
        "artifact_role": "derived_reference_endmembers",
        "artifact_schema_version": ENDMEMBER_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "core_axis_id": inspection.core_axis_id,
        "point_count": point_count,
        "condition_order": list(ENDMEMBER_CONDITIONS),
        "component_order": list(COMPONENTS),
        "endmember_count": len(stored_arrays),
        "entries": entries,
        "source_semantic_term": "gt_endmembers",
        "stored_semantic_term": "derived_reference_endmember",
        "prepared_artifacts": prepared_artifacts,
        "direct_zip_recomputation_equals_prepared": True,
    }
    metadata_bytes = _canonical_json_bytes(metadata, newline=True)
    intensity = np.ascontiguousarray(
        np.stack(stored_arrays, axis=0).reshape(
            len(ENDMEMBER_CONDITIONS),
            len(COMPONENTS),
            point_count,
        ),
        dtype="<f4",
    )
    logical_digest = hashlib.sha256(ENDMEMBER_LOGICAL_DOMAIN)
    logical_digest.update(struct.pack("<Q", len(metadata_bytes)))
    logical_digest.update(metadata_bytes)
    for array in stored_arrays:
        payload = array.tobytes(order="C")
        logical_digest.update(struct.pack("<Q", len(payload)))
        logical_digest.update(payload)
    return {
        "metadata": metadata,
        "metadata_bytes": metadata_bytes,
        "source_arrays": tuple(source_arrays),
        "stored_arrays": tuple(stored_arrays),
        "intensity": intensity,
        "logical_content_sha256": logical_digest.hexdigest(),
        "constituent_records": constituent_total,
    }


def _validate_derived_endmembers(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    path = Path(path)
    expected_record_ids = {
        member.record_id for member in inspection.accepted_members
    }
    if record_ids != expected_record_ids:
        raise _fatal(
            "endmembers.record_ids",
            "core record ID set differs from inspection",
            "ENDMEMBER_RECORD_IDS_INVALID",
        )
    expected = _independent_endmember_expectations(raw_root, inspection)
    expected_metadata = expected["metadata"]
    expected_metadata_bytes = expected["metadata_bytes"]
    expected_intensity = expected["intensity"]
    expected_logical = expected["logical_content_sha256"]
    with h5py.File(path, "r") as artifact:
        _require_group_creation_order_disabled(
            artifact["/"],
            "endmembers",
            code="ENDMEMBER_STRUCTURE_INVALID",
        )
        if set(artifact) != {"metadata_json", "intensity"}:
            raise _fatal(
                "endmembers",
                "object set differs from contract",
                "ENDMEMBER_STRUCTURE_INVALID",
            )
        for name in ("metadata_json", "intensity"):
            if not isinstance(
                artifact.get(name, getlink=True),
                h5py.HardLink,
            ):
                raise _fatal(
                    f"endmembers.{name}",
                    "only direct hard links are permitted",
                    "ENDMEMBER_STRUCTURE_INVALID",
                )
        if set(artifact.attrs) != _ENDMEMBER_ROOT_ATTRS:
            raise _fatal(
                "endmembers.attrs",
                "root attributes differ from contract",
                "ENDMEMBER_STRUCTURE_INVALID",
            )
        for name, value in (
            ("artifact_role", "derived_reference_endmembers"),
            ("artifact_schema_version", ENDMEMBER_SCHEMA_VERSION),
            ("dataset_id", DATASET_ID),
        ):
            if artifact.attrs[name] != value:
                raise _fatal(
                    f"endmembers.attrs.{name}",
                    "fixed root attribute differs from contract",
                    "ENDMEMBER_STRUCTURE_INVALID",
                )
        metadata_dataset = artifact["metadata_json"]
        intensity_dataset = artifact["intensity"]
        if not isinstance(
            metadata_dataset,
            h5py.Dataset,
        ) or not isinstance(intensity_dataset, h5py.Dataset):
            raise _fatal(
                "endmembers",
                "both objects must be datasets",
                "ENDMEMBER_STRUCTURE_INVALID",
            )
        _require_endmember_dataset(
            metadata_dataset,
            path="endmembers.metadata_json",
            dtype="|u1",
            shape=(len(expected_metadata_bytes),),
            chunks=(len(expected_metadata_bytes),),
        )
        _require_endmember_dataset(
            intensity_dataset,
            path="endmembers.intensity",
            dtype="<f4",
            shape=expected_intensity.shape,
            chunks=(1, 1, expected_intensity.shape[-1]),
        )
        observed_metadata_bytes = metadata_dataset[...].tobytes(order="C")
        try:
            observed_metadata = json.loads(
                observed_metadata_bytes,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise _fatal(
                "endmembers.metadata_json",
                str(error),
                "ENDMEMBER_METADATA_INVALID",
            ) from error
        _validate_endmember_metadata_shape(observed_metadata)
        if (
            observed_metadata_bytes
            != _canonical_json_bytes(observed_metadata, newline=True)
            or observed_metadata != expected_metadata
        ):
            raise _fatal(
                "endmembers.metadata_json",
                "metadata differs from direct-ZIP derivation",
                _endmember_metadata_error_code(
                    observed_metadata,
                    expected_metadata,
                ),
            )
        observed_intensity = intensity_dataset[...]
        if not np.array_equal(observed_intensity, expected_intensity):
            raise _fatal(
                "endmembers.intensity",
                "stored values differ from direct-ZIP medians",
                "ENDMEMBER_VALUE_INVALID",
            )
        observed_digest = hashlib.sha256(ENDMEMBER_LOGICAL_DOMAIN)
        observed_digest.update(
            struct.pack("<Q", len(observed_metadata_bytes))
        )
        observed_digest.update(observed_metadata_bytes)
        for condition_index in range(len(ENDMEMBER_CONDITIONS)):
            for component_index in range(len(COMPONENTS)):
                payload = np.ascontiguousarray(
                    observed_intensity[
                        condition_index,
                        component_index,
                    ],
                    dtype="<f4",
                ).tobytes(order="C")
                observed_digest.update(struct.pack("<Q", len(payload)))
                observed_digest.update(payload)
        observed_logical = observed_digest.hexdigest()
        if (
            observed_logical != expected_logical
            or artifact.attrs["logical_content_sha256"]
            != observed_logical
        ):
            raise _fatal(
                "endmembers.logical_content_sha256",
                "logical digest differs from canonical content",
                "ENDMEMBER_LOGICAL_DIGEST_INVALID",
            )
    return {
        "artifact_role": "derived_reference_endmembers",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "logical_content_sha256": expected_logical,
        "endmember_count": len(expected["stored_arrays"]),
        "constituent_records": expected["constituent_records"],
        "equals_prepared": True,
        "high_float32_value_sha256": hashlib.sha256(
            b"".join(
                array.tobytes(order="C")
                for array in expected["stored_arrays"][: len(COMPONENTS)]
            )
        ).hexdigest(),
        "low_float32_value_sha256": hashlib.sha256(
            b"".join(
                array.tobytes(order="C")
                for array in expected["stored_arrays"][len(COMPONENTS) :]
            )
        ).hexdigest(),
    }
