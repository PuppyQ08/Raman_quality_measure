from __future__ import annotations

import base64
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from rpe.io.store import DATASET_FILES, validate_dataset
from rpe.io.sugar_mixtures_models import (
    SugarMixturesValidationError,
    _SugarFinalState,
    _SugarOutputRootLock,
    _SugarPublicationResult,
    _SugarRecoveryPlan,
    _SugarRecoveryResult,
)
from rpe.io.sugar_mixtures_receipt import (
    ADAPTER_VERSION,
    DATASET_ID,
    RECEIPT_SCHEMA_VERSION,
    SCHEMA_VERSION,
)


_PUBLICATION_TRANSACTION_POLICY_BYTES = zlib.decompress(
    base64.b85decode(
        "".join(
            (
                "c-pmCU31$u7Jc7efzh*M$C+(r$9YO@XV+8L$~w+&JFN!;k&uL%Bv=5Htx5gg_uLDBq-8a=Gu<c4__&|vo_oPB6Xk",
                "+gsv^k3c1@TZznS(+escWtm*IdZC&!b*Iz1VV$gpiSmP-|s*8w{VRor&1T;ShE)wCB^X6feAG^!eJ`);M2tj&Gso",
                "c7YGS~WQ5gIekGPY*s!cP-o)edqU>xHOgintl1+vka7c`rub@pKt5<&Qr5X?aT&3bW_&4UTF6%J|rQtM?#+GP}cA",
                "CQajxgdSuDL`n%_Q`DcQjx~{r=QyJy9Qr+usal02VO9Gzf@?*U(a`NfbJ#H1hb;g@uY$Fp;UhLQ)VwOZa)3qJlGU",
                ")rj(5qhH<X40|b=T=an>Kudm&48HIak<b12Dm?7Z%{|#fO{TD(iq-@ZH#rJ^9;3>fWl&Ou~!@6-A|0)3uZ1FO$~#",
                "P}sUQK`xc4U>j_0VWD;#EyxYiti+uOYk^~JrGpl>Dn+!?3)PguDdV;H`1$?C`N_NM^ZBJbng8&^`StbLsXVzjdw2",
                "Qyqx|sh{Nn5sHb8dLM?c`bT2y+Z+v(blG^=IMZcy!PRgr26aaR=D`{7D|DBo=r$~~}&OTTRjVY|?FP;6}gUMA5Fi",
                "bAd2Ek%12Y_E}^U<yC13dhM5OBiAAG=i`d3ya8<1}B1b+oY#ZRpnX6Zyga%Ok-QA0v-(5;<L$EnT+j4^#?{qk9Kb",
                "uV?lEVBsT)=K^<&U#55wi71;o2(XHT!>C#8vo1$|>hg(@$1QbzF<;HjtejHq<fu+tV0vi>EUIIIU%VEZf1roWe%x",
                "WD(SUcOT)_vR)yo7ZkAt{QPPHJ0{#TWCF@8#K#zw_VA_t>zlZsf93D+>2qUY*V_FMj%cE>F*|&Q7l9SM>XG{$YM`",
                "G5;@)2jy01?#Gk8!4Xv7qSU@{h_rRM&^5kmTk8-di97^&Qxg#Yb}-j^T~biU;RWg`Y$+VS+Rjn+cWs#hts2`HBv%",
                "Aox8ly3pr=t8r)nxvlBYQovy@fdh8Yn+gGB2OGA_oT+PO^tj*iZD9Zhl*-IxYvD1z+L2>8IGh-cVbKNV>{8e~oR_",
                ")h7hwAWpT8AOcqeW6>XAYa%vh9wkh@gCyN@?O3ep$XA}A{P4codPquLAYj-JFGJ+f>uPZS@s9Tva7JtBbW3WAE;m",
                "$g|euW_h3GP?*e2?pkQy!eMBo<32QGZ+@rEYeZ*s2{>&Gn8~PqyLLkdCzpX3N+@{+M<CKY5HX#@CA(84K$x;nG_l",
                "Gf4Vy*mId$CcK;bepPs0{&6)*(^6PK8Cq2aT?Vsf*5lTA6IeY>E<?|L5%LC;a&LXB5OK<-wGi=!xW^)ESE$gd$Ze",
                "YH*VZ)(Z^Mwky_1KlAk?iseW&rAhaCX@W?A&u1S>V*bNM-vR&4jB%e}=^NGhHA8cyeL!14^1(fdnmf)QJQU`xDb}",
                "i4#bQh4MMjU>WN@nS(TUt3wz;Y1gq5-a2E2q4xyxCcL3yjU^j3Hz&BOm%$<%L&L`Iv)1rmgu>1c2!6%3CXQ|vC9h",
                "IkBXRVlOU(y!I)@7`b)0ot?Lc8dz>Vr{Xv@3eZbLjt{w2^!WBQh*FCVp~~N_Ul7DQa66IHd6R_gQcr#cae4QVg$H",
                "Eb-PU53GDD3sHrMlj@I({Kup6M!*lJ1g1Yq#Qz{4~K~Y{Y$2cejNAyX~2_&Kz&rJt=5fSPZiWsM|bvD?-R)5IQ{Q",
                "LPd4Q!(I1G^*J9t}#Vzm0~LTp_~@LRAv^r>~7fd0cx=2Z}(?x5RsohNGo%K8$lX%sI@FsjzF}UJ{`1Tktp}lH$k>",
                "!D#<hL>_4DXz&=cjHn|^?+y19Ck6%n-^8geA#k)st9d*cjUPm)`EX|M5oC}`3Lc>}VLHj{iEAv^G-=$79gU-!_j)",
                "$(n%l<SQ6;nQTYAA-_#I;R(jpyTMt>{%5tl4Qjh8&9J@j?n7jR@WLO^M2WjGJ5p7A(8+o8YN=IEDSUQv^3f-}^W0",
                "B6cMCJw`F3=GDO;&~RKX8-y6!|WgU5zsA|`a9EvH-8rxLqetL<!n^68o6v%VLgri@QrCc&7#v%oXPLRr+>Wz_rz6",
                "lt)JM~8xV2DeKdBu__+n7>Sb=Rxg=wElH8pnH!M#X%pojg?j-5_BdQ$Iv80YI=sJxdCGz4!Y!ag4t?@9I28cMu_V",
                "dSdXior8<+*n^TY<dla@6ldv@%QaXf+tIRD3Xug^<CFeP$$|;6#&aU&*=O=Ol9^pO874P5PdcA6xsspFH`p@bpcW",
                "3-yM6Du@{|+!ts`yiD<)tf@}WV~9)sU>4>9<8Dp(NHj%!(bo^Kg6YheMHljP2pi~<TsobeE-DTVJ-yC)!pm~5jwa",
                "Duie0~(sH&pK4oU#}beu2rB3^SmXoT$Nz^kKF84)K}uj9XO1)g2VA<d4UQn(4N#3&TB+IUaE&T+)sJ5zAis0~ihl",
                "atbz=_ohY@f@^|;4o_rrV(-M5@V*tp7Rpx?d#uv8DegR$$<bu<Hz6_4-!M8f+1khE`Z7?6aAQN>RN+mg0Aoo=(#c",
                "Wfx|Ri9HufS^5=39u}zA;O`@_WEyMiMM|Q9PF$MrJS$eeuV`<`mC8{(;^4rJmBa5D4!jta;Lr0@d(!0!AcyW&-2r",
                "zW^&T9Xk&o%iI0bUJ=hK=(G#gV2or&nF2oa8E#k?`bbW||xyz3PW;S)k%*+~Y1Svy2tLMk9IWlLj2mgfcm1rhT-7",
                "$S|eK8pUUdEa=Hm9_DJbjV(n|=}0`X$r%L<;%0sdlk$h@*9P|E$<e{9gI9U%b4dbv3q1l&e$E4Bv$%sNwZk*4DKT",
                "e={@I#7S;Z{PR$K;^UGdNury{;LqFE7H7^Fkfgc3w9Ak&ZYPv<|PHC4gPxaTCT=_3(oq4VHw3$j=#8lA<b^M8JSe",
                "sRH{M)2BqQ0-&eN3eSo3mTIr4}bd~(cnVG",
            )
        ).encode("ascii")
    )
)
if (
    len(_PUBLICATION_TRANSACTION_POLICY_BYTES) != 6_413
    or hashlib.sha256(
        b"rpe-sugar-publication-transaction-v1\0"
        + _PUBLICATION_TRANSACTION_POLICY_BYTES
    ).hexdigest()
    != "6f31dcdded143ac7b57c48eba4e3af4cf9880e99479fb253b35814b85fabc589"
):
    raise RuntimeError("embedded Sugar publication policy is invalid")
_PUBLICATION_TRANSACTION_POLICY = json.loads(
    _PUBLICATION_TRANSACTION_POLICY_BYTES
)


@dataclass(frozen=True)
class _SugarArtifactDescriptor:
    artifact_id: str
    artifact_type: str
    staged_basename: str
    final_basename: str
    backup_basename: str


_ARTIFACT_DESCRIPTORS = tuple(
    _SugarArtifactDescriptor(
        artifact_id=document["artifact_id"],
        artifact_type=document["artifact_type"],
        staged_basename=document["staged_basename"],
        final_basename=document["final_basename"],
        backup_basename=document["backup_basename"],
    )
    for document in _PUBLICATION_TRANSACTION_POLICY["artifacts"]
)
_PUBLICATION_ORDER = tuple(
    _PUBLICATION_TRANSACTION_POLICY["publication_order"]
)
_BACKUP_ORDER = tuple(_PUBLICATION_TRANSACTION_POLICY["backup_order"])
_ROLLBACK_REMOVE_ORDER = tuple(
    _PUBLICATION_TRANSACTION_POLICY["rollback_remove_order"]
)
_RESTORE_PAYLOAD_ORDER = tuple(
    _PUBLICATION_TRANSACTION_POLICY["restore_payload_order"]
)
_DESCRIPTOR_BY_ID = MappingProxyType(
    {
        descriptor.artifact_id: descriptor
        for descriptor in _ARTIFACT_DESCRIPTORS
    }
)
_JOURNAL_NAME = "transaction_recovery.json"
_JOURNAL_TEMP_NAME = ".transaction_recovery.tmp"
_JOURNAL_KEYS = set(
    _PUBLICATION_TRANSACTION_POLICY["journal"]["exact_keys"]
)
_CONTENT_PATHS = {
    *(f"sugar_mixtures_raman/{name}" for name in DATASET_FILES),
    "sugar_mixtures_raman_views.json",
    "sugar_mixtures_raman_derived_reference_endmembers.h5",
    "sugar_mixtures_raman_auxiliary_axes.h5",
    "sugar_mixtures_raman_acquisition_json_text.jsonl",
    "sugar_mixtures_raman_conversion.json",
}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_lexists(path: Path) -> bool:
    path = Path(path)
    return path.is_symlink() or path.exists()


def _remove_created_empty_root(
    output_root: Path,
    created_identity: tuple[int, int] | None,
) -> None:
    if created_identity is None:
        return
    try:
        observed = os.lstat(output_root)
    except OSError:
        return
    if (
        stat.S_ISDIR(observed.st_mode)
        and (observed.st_dev, observed.st_ino) == created_identity
        and not any(Path(output_root).iterdir())
    ):
        Path(output_root).rmdir()


def _validate_artifact_descriptors(
    descriptors: Sequence[_SugarArtifactDescriptor],
) -> tuple[_SugarArtifactDescriptor, ...]:
    descriptors = tuple(descriptors)
    if len(descriptors) != 6:
        raise _fatal(
            "transaction.artifacts",
            "must contain exactly six descriptors",
            "PUBLICATION_DESCRIPTOR_INVALID",
        )
    portable = []
    for descriptor in descriptors:
        values = (
            descriptor.artifact_id,
            descriptor.staged_basename,
            descriptor.final_basename,
            descriptor.backup_basename,
        )
        if descriptor.artifact_type not in {"dataset", "file"} or any(
            not value
            or Path(value).name != value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
            for value in values
        ):
            raise _fatal(
                f"transaction.artifacts.{descriptor.artifact_id}",
                "descriptor paths/types must be unique portable basenames",
                "PUBLICATION_DESCRIPTOR_INVALID",
            )
        portable.append(values)
    for index in range(4):
        values = [row[index] for row in portable]
        if len(values) != len(set(values)):
            raise _fatal(
                "transaction.artifacts",
                "descriptor identities and basenames must be unique",
                "PUBLICATION_DESCRIPTOR_INVALID",
            )
    if tuple(descriptor.artifact_id for descriptor in descriptors) != (
        _PUBLICATION_ORDER
    ):
        raise _fatal(
            "transaction.artifacts",
            "descriptors must be in publication order",
            "PUBLICATION_DESCRIPTOR_INVALID",
        )
    return descriptors


_validate_artifact_descriptors(_ARTIFACT_DESCRIPTORS)


@contextlib.contextmanager
def _open_sugar_output_root_lock(
    output_root: Path,
    *,
    shared: bool,
):
    output_root = Path(output_root)
    created = False
    created_identity = None
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise _fatal(
            "output_root",
            "must be a non-symlink directory",
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
    if not output_root.exists():
        output_root.mkdir(parents=True)
        created = True
        created_stat = os.lstat(output_root)
        created_identity = (created_stat.st_dev, created_stat.st_ino)
    flags = os.O_RDONLY | os.O_DIRECTORY
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _remove_created_empty_root(output_root, created_identity)
        raise _fatal(
            "output_root",
            "O_NOFOLLOW is unavailable",
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
    flags |= nofollow
    try:
        directory_fd = os.open(output_root, flags)
    except OSError as error:
        _remove_created_empty_root(output_root, created_identity)
        raise _fatal(
            "output_root",
            str(error),
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        ) from error
    lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    lock_acquired = False
    try:
        try:
            fcntl.flock(directory_fd, lock_mode | fcntl.LOCK_NB)
            lock_acquired = True
        except OSError as error:
            code = (
                "PUBLICATION_LOCKED"
                if error.errno in {errno.EAGAIN, errno.EACCES}
                else "PUBLICATION_PLATFORM_UNSUPPORTED"
            )
            raise _fatal(
                "output_root.lock",
                str(error),
                code,
            ) from error
        lock = _SugarOutputRootLock(
            output_root=output_root,
            directory_fd=directory_fd,
            created_output_root=created,
        )
        yield lock
    finally:
        try:
            if created and lock_acquired:
                _assert_lock_identity(lock)
        finally:
            try:
                if lock_acquired:
                    fcntl.flock(directory_fd, fcntl.LOCK_UN)
            finally:
                os.close(directory_fd)
        if (
            created
            and output_root.is_dir()
            and not any(output_root.iterdir())
        ):
            output_root.rmdir()


def _lock_sugar_output_root(output_root: Path):
    return _open_sugar_output_root_lock(output_root, shared=False)


def _lock_sugar_output_root_shared(output_root: Path):
    return _open_sugar_output_root_lock(output_root, shared=True)


def _assert_lock_identity(lock: _SugarOutputRootLock) -> None:
    try:
        locked = os.fstat(lock.directory_fd)
        observed = os.lstat(lock.output_root)
    except OSError as error:
        raise _fatal(
            "output_root.identity",
            str(error),
            "OUTPUT_ROOT_REPLACED",
        ) from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (locked.st_dev, locked.st_ino)
        != (observed.st_dev, observed.st_ino)
    ):
        raise _fatal(
            "output_root.identity",
            "locked directory identity differs from output_root",
            "OUTPUT_ROOT_REPLACED",
        )


def _artifact_identity(
    descriptor: _SugarArtifactDescriptor,
    path: Path,
) -> str:
    path = Path(path)
    if path.is_symlink():
        raise _fatal(
            f"transaction.artifacts.{descriptor.artifact_id}",
            "symlink artifacts are forbidden",
            "PUBLICATION_ARTIFACT_INVALID",
        )
    if descriptor.artifact_type == "file":
        if not path.is_file():
            raise _fatal(
                f"transaction.artifacts.{descriptor.artifact_id}",
                "expected a regular file",
                "PUBLICATION_ARTIFACT_INVALID",
            )
        return _sha256_file(path)
    if not path.is_dir() or set(child.name for child in path.iterdir()) != set(
        DATASET_FILES
    ):
        raise _fatal(
            f"transaction.artifacts.{descriptor.artifact_id}",
            "core directory differs from the five-file contract",
            "PUBLICATION_ARTIFACT_INVALID",
        )
    digest = hashlib.sha256(b"rpe-sugar-top-level-artifact-v1\0")
    for child in sorted(path.iterdir(), key=lambda value: value.name):
        if child.is_symlink() or not child.is_file():
            raise _fatal(
                f"transaction.artifacts.{descriptor.artifact_id}",
                "core children must be regular non-symlink files",
                "PUBLICATION_ARTIFACT_INVALID",
            )
        encoded = child.name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", child.stat().st_size))
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _scan_content_hashes(root: Path) -> Mapping[str, str]:
    root = Path(root)
    paths = {
        relative: root / relative
        for relative in _CONTENT_PATHS
    }
    core = root / "sugar_mixtures_raman"
    expected_top_level = {
        descriptor.final_basename
        for descriptor in _ARTIFACT_DESCRIPTORS
    }
    observed_owned = {
        path.name
        for path in root.iterdir()
        if path.name in expected_top_level
    }
    if (
        observed_owned != expected_top_level
        or core.is_symlink()
        or not core.is_dir()
        or set(path.name for path in core.iterdir()) != set(DATASET_FILES)
        or any(
            path.is_symlink() or not path.is_file()
            for path in paths.values()
        )
    ):
        raise _fatal(
            "transaction.content_files",
            "content path set differs from exact ten-file contract",
            "PUBLICATION_ARTIFACT_INVALID",
        )
    return MappingProxyType(
        {
            name: _sha256_file(paths[name])
            for name in sorted(paths)
        }
    )


def _output_snapshot_sha256_from_hashes(
    hashes: Mapping[str, str],
    *,
    staging_parent: Path | None,
) -> str:
    if staging_parent is None:
        raise ValueError("snapshot byte sizes require a content root")
    root = Path(staging_parent)
    digest = hashlib.sha256(b"rpe-sugar-output-snapshot-v1\0")
    for name in sorted(hashes, key=lambda value: value.encode("utf-8")):
        path = root / name
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", path.stat().st_size))
        digest.update(bytes.fromhex(hashes[name]))
    return digest.hexdigest()


def _snapshot_identity(root: Path) -> tuple[Mapping[str, str], str]:
    hashes = _scan_content_hashes(root)
    return (
        hashes,
        _output_snapshot_sha256_from_hashes(
            hashes,
            staging_parent=root,
        ),
    )


def _read_canonical_receipt(path: Path) -> tuple[bytes, Mapping[str, object]]:
    path = Path(path)
    try:
        payload = path.read_bytes()
        document = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _fatal(
            "receipt",
            str(error),
            "SNAPSHOT_UNAVAILABLE",
        ) from error
    if (
        not isinstance(document, Mapping)
        or payload != _canonical_json_bytes(document)
    ):
        raise _fatal(
            "receipt",
            "receipt is not canonical finite JSON",
            "SNAPSHOT_UNAVAILABLE",
        )
    return payload, document


def _validate_complete_snapshot(root: Path) -> str:
    root = Path(root)
    receipt_path = root / _DESCRIPTOR_BY_ID["receipt"].final_basename
    _, receipt = _read_canonical_receipt(receipt_path)
    fixed_scalars = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
    }
    for name, expected in fixed_scalars.items():
        if receipt.get(name) != expected:
            raise _fatal(
                f"receipt.{name}",
                f"expected fixed value {expected!r}",
                "SNAPSHOT_UNAVAILABLE",
            )
    validate_dataset(root / _DESCRIPTOR_BY_ID["core"].final_basename)
    semantic = receipt.get("semantic_contract")
    if not isinstance(semantic, Mapping):
        raise _fatal(
            "receipt.semantic_contract",
            "semantic contract must be an object",
            "SNAPSHOT_UNAVAILABLE",
        )
    source_contract = receipt.get("source_contract")
    limitations = receipt.get("limitations")
    license_document = receipt.get("license")
    preprocessing_evidence = receipt.get("preprocessing_evidence")
    if (
        not isinstance(source_contract, Mapping)
        or not isinstance(limitations, Mapping)
        or not isinstance(license_document, Mapping)
        or not isinstance(preprocessing_evidence, list)
    ):
        raise _fatal(
            "receipt.contracts",
            "source/policy contract sections have invalid types",
            "SNAPSHOT_UNAVAILABLE",
        )
    identities = (
        (
            "source_contract_sha256",
            b"rpe-sugar-receipt-source-contract-v1\0",
            source_contract,
        ),
        (
            "semantic_contract_sha256",
            b"rpe-sugar-receipt-semantic-contract-v1\0",
            semantic,
        ),
        (
            "policy_contract_sha256",
            b"rpe-sugar-receipt-policy-contract-v1\0",
            {
                "license": license_document,
                "limitations": limitations,
                "preprocessing_evidence": preprocessing_evidence,
            },
        ),
    )
    for name, domain, document in identities:
        observed = hashlib.sha256(
            domain + _canonical_json_bytes(document)
        ).hexdigest()
        if receipt.get(name) != observed:
            raise _fatal(
                f"receipt.{name}",
                "contract digest differs from canonical section",
                "SNAPSHOT_UNAVAILABLE",
            )
    from rpe.io.sugar_mixtures import (
        _measure_acquisition_json_access,
        _measure_auxiliary_axes_access,
        _measure_derived_endmembers_access,
        _measure_views_access,
    )

    canonical_members = receipt.get("source_contract", {}).get(
        "canonical_members"
    )
    if not isinstance(canonical_members, list):
        raise _fatal(
            "receipt.source_contract.canonical_members",
            "canonical members must be an array",
            "SNAPSHOT_UNAVAILABLE",
        )
    source_bindings = {}
    for index, member in enumerate(canonical_members):
        path = f"receipt.source_contract.canonical_members[{index}]"
        if not isinstance(member, Mapping):
            raise _fatal(
                path,
                "canonical member binding must be an object",
                "SNAPSHOT_UNAVAILABLE",
            )
        record_id = member.get("record_id")
        source_member = member.get("source_member")
        source_sha256 = member.get("sha256")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not isinstance(source_member, str)
            or not source_member
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_sha256
            )
            or record_id in source_bindings
        ):
            raise _fatal(
                path,
                "canonical source binding is invalid or duplicated",
                "SNAPSHOT_UNAVAILABLE",
            )
        source_bindings[record_id] = (
            source_member,
            source_sha256,
        )
    _measure_views_access(
        root / _DESCRIPTOR_BY_ID["views"].final_basename,
        source_member_by_record_id={
            record_id: binding[0]
            for record_id, binding in source_bindings.items()
        },
    )
    _measure_acquisition_json_access(
        root / _DESCRIPTOR_BY_ID["acquisition_json"].final_basename,
        record_ids=set(source_bindings),
        source_bindings=source_bindings,
    )
    logical_digests = {}
    for name in ("auxiliary_axes", "derived_endmembers"):
        document = semantic.get(name)
        digest = (
            document.get("logical_content_sha256")
            if isinstance(document, Mapping)
            else None
        )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise _fatal(
                f"receipt.semantic_contract.{name}.logical_content_sha256",
                "logical content digest must be lowercase SHA256",
                "SNAPSHOT_UNAVAILABLE",
            )
        logical_digests[name] = digest
    _measure_auxiliary_axes_access(
        root / _DESCRIPTOR_BY_ID["auxiliary_axes"].final_basename,
        expected_logical_content_sha256=logical_digests["auxiliary_axes"],
    )
    _measure_derived_endmembers_access(
        root / _DESCRIPTOR_BY_ID["derived_endmembers"].final_basename,
        expected_logical_content_sha256=logical_digests[
            "derived_endmembers"
        ],
    )
    expected = {}
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise _fatal(
            "receipt.outputs",
            "outputs must be an object",
            "SNAPSHOT_UNAVAILABLE",
        )
    for artifact_name, artifact in outputs.items():
        if not isinstance(artifact, Mapping):
            raise _fatal(
                "receipt.outputs",
                "artifact attribution must be an object",
                "SNAPSHOT_UNAVAILABLE",
            )
        files = artifact.get("files")
        if not isinstance(files, Mapping):
            raise _fatal(
                "receipt.outputs",
                "file attribution must be an object",
                "SNAPSHOT_UNAVAILABLE",
            )
        for basename, details in files.items():
            relative = (
                f"{artifact_name}/{basename}"
                if artifact.get("artifact_type") == "dataset"
                else basename
            )
            path = root / relative
            if (
                not isinstance(details, Mapping)
                or path.is_symlink()
                or not path.is_file()
                or details.get("bytes") != path.stat().st_size
                or details.get("sha256") != _sha256_file(path)
            ):
                raise _fatal(
                    f"receipt.outputs.{relative}",
                    "physical attribution differs",
                    "SNAPSHOT_UNAVAILABLE",
                )
            expected[relative] = details["sha256"]
    expected["sugar_mixtures_raman_conversion.json"] = _sha256_file(
        receipt_path
    )
    observed, snapshot = _snapshot_identity(root)
    if dict(observed) != expected:
        raise _fatal(
            "receipt.outputs",
            "ten-file snapshot differs from receipt",
            "SNAPSHOT_UNAVAILABLE",
        )
    return snapshot


def _preflight_sugar_final_state(
    lock: _SugarOutputRootLock,
    *,
    overwrite: bool,
) -> _SugarFinalState:
    _assert_lock_identity(lock)
    preserved = [
        path
        for path in lock.output_root.iterdir()
        if path.name.startswith(".sugar-mixtures.staging-")
    ]
    if preserved:
        raise _fatal(
            "output_root.staging",
            "preserved recovery staging requires explicit recovery",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    finals = [
        lock.output_root / descriptor.final_basename
        for descriptor in _ARTIFACT_DESCRIPTORS
    ]
    present = [_path_lexists(path) for path in finals]
    if not any(present):
        return _SugarFinalState("absent", None)
    if not overwrite:
        raise _fatal(
            "output_root",
            "one or more owned final paths already exist",
            "OUTPUT_EXISTS",
        )
    if not all(present):
        raise _fatal(
            "output_root",
            "final artifact set is partial",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    try:
        snapshot = _validate_complete_snapshot(lock.output_root)
    except (SugarMixturesValidationError, OSError, ValueError) as error:
        raise _fatal(
            "output_root",
            f"complete final snapshot validation failed: {error}",
            "PUBLICATION_RECOVERY_REQUIRED",
        ) from error
    return _SugarFinalState("complete", snapshot)


def _actual_final_state_locked(
    lock: _SugarOutputRootLock,
) -> _SugarFinalState:
    _assert_lock_identity(lock)
    finals = [
        lock.output_root / descriptor.final_basename
        for descriptor in _ARTIFACT_DESCRIPTORS
    ]
    present = [_path_lexists(path) for path in finals]
    if not any(present):
        return _SugarFinalState("absent", None)
    if not all(present):
        return _SugarFinalState("partial_or_invalid", None)
    try:
        return _SugarFinalState(
            "complete",
            _validate_complete_snapshot(lock.output_root),
        )
    except (SugarMixturesValidationError, OSError, ValueError):
        return _SugarFinalState("partial_or_invalid", None)


def _fsync_directory(path: Path) -> None:
    fd = os.open(
        Path(path),
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_transaction_path(path: Path, *, label: str) -> None:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if path.is_dir():
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durability_gate(staging_parent: Path) -> tuple[Mapping[str, str], str]:
    staging_parent = Path(staging_parent)
    hashes = _scan_content_hashes(staging_parent)
    for relative in sorted(_CONTENT_PATHS):
        _fsync_transaction_path(
            staging_parent / relative,
            label=relative,
        )
    _fsync_transaction_path(
        staging_parent / "sugar_mixtures_raman",
        label="sugar_mixtures_raman",
    )
    _fsync_transaction_path(
        staging_parent,
        label="staging_parent",
    )
    return (
        hashes,
        _output_snapshot_sha256_from_hashes(
            hashes,
            staging_parent=staging_parent,
        ),
    )


def _write_journal_temp(path: Path, payload: bytes) -> None:
    with Path(path).open("wb") as output:
        output.write(payload)
        output.flush()


def _fsync_journal_temp(path: Path) -> None:
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_journal(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _fsync_journal_directory(staging_parent: Path) -> None:
    _fsync_directory(staging_parent)


def _write_journal(
    staging_parent: Path,
    journal: Mapping[str, object],
    *,
    lock: _SugarOutputRootLock | None = None,
) -> None:
    if set(journal) != _JOURNAL_KEYS:
        raise _fatal(
            "transaction.journal",
            "journal key set differs from contract",
            "PUBLICATION_JOURNAL_INVALID",
        )
    temporary = Path(staging_parent) / _JOURNAL_TEMP_NAME
    final = Path(staging_parent) / _JOURNAL_NAME
    if lock is not None:
        _assert_lock_identity(lock)
    _write_journal_temp(temporary, _canonical_json_bytes(journal))
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_journal_temp(temporary)
    if lock is not None:
        _assert_lock_identity(lock)
    _replace_journal(temporary, final)
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_journal_directory(staging_parent)


def _update_journal(
    staging_parent: Path,
    journal: dict[str, object],
    *,
    lock: _SugarOutputRootLock | None = None,
    **updates: object,
) -> None:
    journal.update(updates)
    _write_journal(staging_parent, journal, lock=lock)


def _fsync_rename_directories(
    staging_parent: Path,
    output_root: Path,
    *,
    lock: _SugarOutputRootLock | None = None,
) -> None:
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_directory(staging_parent)
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_directory(output_root)
    if lock is not None:
        _assert_lock_identity(lock)


def _commit_directories(
    staging_parent: Path,
    output_root: Path,
    *,
    lock: _SugarOutputRootLock | None = None,
) -> None:
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_directory(staging_parent)
    if lock is not None:
        _assert_lock_identity(lock)
    _fsync_directory(output_root)
    if lock is not None:
        _assert_lock_identity(lock)


def _cleanup_staging(
    staging_parent: Path,
    output_root: Path,
) -> None:
    shutil.rmtree(staging_parent)
    _fsync_directory(output_root)


def _cleanup_staging_locked(
    lock: _SugarOutputRootLock,
    staging_parent: Path,
) -> None:
    _assert_lock_identity(lock)
    _cleanup_staging(staging_parent, lock.output_root)


def _journal_artifacts(
    staging_parent: Path,
    output_root: Path,
    *,
    old_identities: Mapping[str, str | None],
    new_identities: Mapping[str, str],
) -> list[dict[str, object]]:
    return [
        {
            "artifact_id": descriptor.artifact_id,
            "artifact_type": descriptor.artifact_type,
            "staged_basename": descriptor.staged_basename,
            "final_basename": descriptor.final_basename,
            "backup_basename": descriptor.backup_basename,
            "old_identity_sha256": old_identities.get(
                descriptor.artifact_id
            ),
            "new_identity_sha256": new_identities[
                descriptor.artifact_id
            ],
        }
        for descriptor in _ARTIFACT_DESCRIPTORS
    ]


def _initial_journal(
    lock: _SugarOutputRootLock,
    staging_parent: Path,
    *,
    previous_state: _SugarFinalState,
    new_snapshot_sha256: str,
    new_identities: Mapping[str, str],
) -> dict[str, object]:
    old_identities = {}
    for descriptor in _ARTIFACT_DESCRIPTORS:
        path = lock.output_root / descriptor.final_basename
        old_identities[descriptor.artifact_id] = (
            _artifact_identity(descriptor, path)
            if _path_lexists(path)
            else None
        )
    locked = os.fstat(lock.directory_fd)
    return {
        "transaction_schema_version": "1.0.0",
        "transaction_id": uuid.uuid4().hex,
        "phase": "prepared",
        "output_root_device": locked.st_dev,
        "output_root_inode": locked.st_ino,
        "old_snapshot_sha256": previous_state.output_snapshot_sha256,
        "new_snapshot_sha256": new_snapshot_sha256,
        "publication_order": list(_PUBLICATION_ORDER),
        "backup_order": list(_BACKUP_ORDER),
        "rollback_remove_order": list(_ROLLBACK_REMOVE_ORDER),
        "restore_payload_order": list(_RESTORE_PAYLOAD_ORDER),
        "completed_backups": [],
        "completed_publications": [],
        "completed_removals": [],
        "completed_restores": [],
        "current_operation": None,
        "receipt_installed": False,
        "receipt_commit_fsynced": False,
        "artifacts": _journal_artifacts(
            staging_parent,
            lock.output_root,
            old_identities=old_identities,
            new_identities=new_identities,
        ),
    }


def _remove_owned_artifact(
    descriptor: _SugarArtifactDescriptor,
    final_path: Path,
    expected_identity: str,
) -> None:
    if not _path_lexists(final_path):
        return
    if _artifact_identity(descriptor, final_path) != expected_identity:
        raise _fatal(
            f"publication.final.{descriptor.final_basename}",
            "final identity is unknown",
            "PUBLICATION_RESTORE_FAILED",
        )
    if descriptor.artifact_type == "dataset":
        shutil.rmtree(final_path)
    else:
        final_path.unlink()


def _rollback_transaction(
    *,
    lock: _SugarOutputRootLock,
    staging_parent: Path,
    journal: dict[str, object],
    published: Sequence[str],
    backed_up: Sequence[str],
    new_identities: Mapping[str, str],
) -> None:
    errors = []
    artifact_map = _journal_artifact_map(journal)
    completed_removals = []
    for artifact_id in _ROLLBACK_REMOVE_ORDER:
        if artifact_id not in published:
            continue
        descriptor = _DESCRIPTOR_BY_ID[artifact_id]
        final = lock.output_root / descriptor.final_basename
        _assert_lock_identity(lock)
        try:
            _remove_owned_artifact(
                descriptor,
                final,
                new_identities[artifact_id],
            )
            completed_removals.append(artifact_id)
        except BaseException as error:
            errors.append(
                f"remove {descriptor.final_basename}: "
                f"{type(error).__name__}: {error}"
            )
            if artifact_id == "receipt":
                break
    if errors and "receipt" in published and "receipt" not in completed_removals:
        _update_journal(
            staging_parent,
            journal,
            lock=lock,
            phase="recovery_required",
            completed_removals=completed_removals,
            current_operation="manual_recovery_required",
        )
        raise _fatal(
            "publication.restore",
            (
                f"manual recovery required from {staging_parent}: "
                + "; ".join(errors)
            ),
            "PUBLICATION_RESTORE_FAILED",
        )
    if journal["old_snapshot_sha256"] is None:
        for descriptor in _ARTIFACT_DESCRIPTORS:
            final = lock.output_root / descriptor.final_basename
            if _path_lexists(final):
                errors.append(
                    f"remove {descriptor.final_basename}: "
                    "unexpected final identity remains"
                )
    completed_restores = []
    payload_restore_failed = False
    for artifact_id in _RESTORE_PAYLOAD_ORDER:
        if artifact_id not in backed_up:
            continue
        descriptor = _DESCRIPTOR_BY_ID[artifact_id]
        backup = staging_parent / descriptor.backup_basename
        final = lock.output_root / descriptor.final_basename
        try:
            _assert_lock_identity(lock)
            if _path_lexists(final):
                payload_restore_failed = True
                errors.append(
                    f"restore {descriptor.final_basename}: "
                    "destination still exists"
                )
                continue
            os.replace(backup, final)
            completed_restores.append(artifact_id)
            _fsync_rename_directories(
                staging_parent,
                lock.output_root,
                lock=lock,
            )
        except BaseException as error:
            payload_restore_failed = True
            errors.append(
                f"restore {descriptor.final_basename}: "
                f"{type(error).__name__}: {error}"
            )
    if journal["old_snapshot_sha256"] is not None:
        for artifact_id in _RESTORE_PAYLOAD_ORDER:
            descriptor = _DESCRIPTOR_BY_ID[artifact_id]
            final = lock.output_root / descriptor.final_basename
            expected_old_identity = artifact_map[artifact_id][
                "old_identity_sha256"
            ]
            try:
                if (
                    not isinstance(expected_old_identity, str)
                    or not _path_lexists(final)
                    or _artifact_identity(descriptor, final)
                    != expected_old_identity
                ):
                    payload_restore_failed = True
                    errors.append(
                        f"validate restored {descriptor.final_basename}: "
                        "old payload identity is unavailable"
                    )
            except BaseException as error:
                payload_restore_failed = True
                errors.append(
                    f"validate restored {descriptor.final_basename}: "
                    f"{type(error).__name__}: {error}"
                )
    if "receipt" in backed_up and not payload_restore_failed:
        descriptor = _DESCRIPTOR_BY_ID["receipt"]
        try:
            _assert_lock_identity(lock)
            final = lock.output_root / descriptor.final_basename
            if _path_lexists(final):
                raise _fatal(
                    f"publication.final.{descriptor.final_basename}",
                    "old receipt restore destination appeared before rename",
                    "PUBLICATION_RESTORE_FAILED",
                )
            os.replace(
                staging_parent / descriptor.backup_basename,
                final,
            )
            completed_restores.append("receipt")
            _fsync_rename_directories(
                staging_parent,
                lock.output_root,
                lock=lock,
            )
        except BaseException as error:
            errors.append(
                f"restore {descriptor.final_basename}: "
                f"{type(error).__name__}: {error}"
            )
    if errors:
        _update_journal(
            staging_parent,
            journal,
            lock=lock,
            phase="recovery_required",
            completed_removals=completed_removals,
            completed_restores=completed_restores,
            current_operation="manual_recovery_required",
        )
        raise _fatal(
            "publication.restore",
            (
                f"manual recovery required from {staging_parent}: "
                + "; ".join(errors)
            ),
            "PUBLICATION_RESTORE_FAILED",
        )
    _cleanup_staging_locked(lock, staging_parent)


def _publish_sugar_artifacts(
    *,
    lock: _SugarOutputRootLock,
    staging_parent: Path,
    previous_state: _SugarFinalState,
) -> _SugarPublicationResult:
    _assert_lock_identity(lock)
    staging_parent = Path(staging_parent)
    if (
        staging_parent.parent != lock.output_root
        or not staging_parent.name.startswith(".sugar-mixtures.staging-")
        or staging_parent.is_symlink()
        or not staging_parent.is_dir()
        or staging_parent.stat().st_dev
        != os.fstat(lock.directory_fd).st_dev
    ):
        raise _fatal(
            "staging_parent",
            "must be a direct same-filesystem non-symlink child",
            "STAGING_PARENT_INVALID",
        )
    if previous_state.classification not in {"absent", "complete"}:
        raise _fatal(
            "publication.previous_state",
            "normal publication requires absent or complete state",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    actual_state = _actual_final_state_locked(lock)
    if actual_state != previous_state:
        raise _fatal(
            "publication.previous_state",
            (
                f"supplied {previous_state.classification}/"
                f"{previous_state.output_snapshot_sha256} differs from "
                f"physical {actual_state.classification}/"
                f"{actual_state.output_snapshot_sha256}"
            ),
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    new_hashes, new_snapshot = _durability_gate(staging_parent)
    validated_staged_snapshot = _validate_complete_snapshot(staging_parent)
    if validated_staged_snapshot != new_snapshot:
        raise _fatal(
            "publication.staging_snapshot",
            "validated staged snapshot differs from durability gate",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    actual_state = _actual_final_state_locked(lock)
    if actual_state != previous_state:
        raise _fatal(
            "publication.previous_state",
            (
                "physical final state changed during staged durability "
                f"validation: supplied {previous_state.classification}/"
                f"{previous_state.output_snapshot_sha256}, observed "
                f"{actual_state.classification}/"
                f"{actual_state.output_snapshot_sha256}"
            ),
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    if (
        previous_state.classification == "complete"
        and previous_state.output_snapshot_sha256 == new_snapshot
    ):
        _cleanup_staging_locked(lock, staging_parent)
        return _SugarPublicationResult(
            mode="identical_noop",
            committed=True,
            output_snapshot_sha256=new_snapshot,
            staging_parent=None,
        )
    new_identities = {
        descriptor.artifact_id: _artifact_identity(
            descriptor,
            staging_parent / descriptor.staged_basename,
        )
        for descriptor in _ARTIFACT_DESCRIPTORS
    }
    journal = _initial_journal(
        lock,
        staging_parent,
        previous_state=previous_state,
        new_snapshot_sha256=new_snapshot,
        new_identities=new_identities,
    )
    try:
        _write_journal(staging_parent, journal, lock=lock)
    except BaseException:
        journal_exists = (staging_parent / _JOURNAL_NAME).is_file()
        if not journal_exists:
            _cleanup_staging_locked(lock, staging_parent)
        raise
    backed_up = []
    published = []
    committed = False
    try:
        actual_state = _actual_final_state_locked(lock)
        if actual_state != previous_state:
            raise _fatal(
                "publication.previous_state",
                (
                    "physical final state changed before first rename: "
                    f"supplied {previous_state.classification}/"
                    f"{previous_state.output_snapshot_sha256}, observed "
                    f"{actual_state.classification}/"
                    f"{actual_state.output_snapshot_sha256}"
                ),
                "PUBLICATION_RECOVERY_REQUIRED",
            )
        if previous_state.classification == "complete":
            artifact_map = _journal_artifact_map(journal)
            for artifact_id in _BACKUP_ORDER:
                descriptor = _DESCRIPTOR_BY_ID[artifact_id]
                _assert_lock_identity(lock)
                _update_journal(
                    staging_parent,
                    journal,
                    lock=lock,
                    phase="backup",
                    current_operation=f"backup:{artifact_id}",
                )
                _assert_lock_identity(lock)
                final_path = (
                    lock.output_root / descriptor.final_basename
                )
                expected_old_identity = artifact_map[artifact_id][
                    "old_identity_sha256"
                ]
                if (
                    not isinstance(expected_old_identity, str)
                    or not _path_lexists(final_path)
                    or _artifact_identity(descriptor, final_path)
                    != expected_old_identity
                ):
                    raise _fatal(
                        f"publication.final.{descriptor.final_basename}",
                        "backup source identity changed before rename",
                        "PUBLICATION_RECOVERY_REQUIRED",
                    )
                os.replace(
                    final_path,
                    staging_parent / descriptor.backup_basename,
                )
                backed_up.append(artifact_id)
                _update_journal(
                    staging_parent,
                    journal,
                    lock=lock,
                    completed_backups=list(backed_up),
                )
                _fsync_rename_directories(
                    staging_parent,
                    lock.output_root,
                    lock=lock,
                )
        for artifact_id in _PUBLICATION_ORDER:
            descriptor = _DESCRIPTOR_BY_ID[artifact_id]
            _assert_lock_identity(lock)
            _update_journal(
                staging_parent,
                journal,
                lock=lock,
                phase="publication",
                current_operation=f"publish:{artifact_id}",
            )
            _assert_lock_identity(lock)
            final_path = (
                lock.output_root / descriptor.final_basename
            )
            if _path_lexists(final_path):
                raise _fatal(
                    f"publication.final.{descriptor.final_basename}",
                    "publication destination appeared before rename",
                    "PUBLICATION_RECOVERY_REQUIRED",
                )
            os.replace(
                staging_parent / descriptor.staged_basename,
                final_path,
            )
            published.append(artifact_id)
            _update_journal(
                staging_parent,
                journal,
                lock=lock,
                completed_publications=list(published),
                receipt_installed=("receipt" in published),
            )
            _fsync_rename_directories(
                staging_parent,
                lock.output_root,
                lock=lock,
            )
        _commit_directories(
            staging_parent,
            lock.output_root,
            lock=lock,
        )
        committed = True
        try:
            _update_journal(
                staging_parent,
                journal,
                lock=lock,
                phase="committed",
                current_operation=None,
                receipt_commit_fsynced=True,
            )
        except BaseException as error:
            raise _fatal(
                "publication.cleanup",
                (
                    "receipt commit is durable but committed journal update "
                    f"failed; recovery evidence at {staging_parent}: "
                    f"{type(error).__name__}: {error}"
                ),
                "PUBLICATION_COMMITTED_CLEANUP_FAILED",
            ) from error
    except BaseException as publication_error:
        if committed:
            raise
        if (
            isinstance(publication_error, SugarMixturesValidationError)
            and publication_error.code == "OUTPUT_ROOT_REPLACED"
        ):
            raise
        try:
            _rollback_transaction(
                lock=lock,
                staging_parent=staging_parent,
                journal=journal,
                published=published,
                backed_up=backed_up,
                new_identities=new_identities,
            )
        except SugarMixturesValidationError:
            raise
        raise publication_error
    try:
        _cleanup_staging_locked(lock, staging_parent)
    except BaseException as error:
        raise _fatal(
            "publication.cleanup",
            (
                f"committed snapshot cleanup failed; recovery evidence at "
                f"{staging_parent}: {type(error).__name__}: {error}"
            ),
            "PUBLICATION_COMMITTED_CLEANUP_FAILED",
        ) from error
    return _SugarPublicationResult(
        mode=(
            "first_publication"
            if previous_state.classification == "absent"
            else "overwrite"
        ),
        committed=True,
        output_snapshot_sha256=new_snapshot,
        staging_parent=None,
    )


def _journal_document(staging_parent: Path) -> Mapping[str, object]:
    path = Path(staging_parent) / _JOURNAL_NAME
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fatal(
            "recovery.journal",
            str(error),
            "PUBLICATION_RECOVERY_REQUIRED",
        ) from error
    if (
        not isinstance(document, Mapping)
        or set(document) != _JOURNAL_KEYS
        or payload != _canonical_json_bytes(document)
    ):
        raise _fatal(
            "recovery.journal",
            "journal is not canonical or has the wrong keys",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    return document


def _journal_artifact_map(
    journal: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    artifacts = journal.get("artifacts")
    if not isinstance(artifacts, list):
        raise _fatal(
            "recovery.journal.artifacts",
            "artifacts must be an array",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    result = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise _fatal(
                "recovery.journal.artifacts",
                "artifact entries must be objects",
                "PUBLICATION_RECOVERY_REQUIRED",
            )
        result[artifact["artifact_id"]] = artifact
    if set(result) != set(_DESCRIPTOR_BY_ID):
        raise _fatal(
            "recovery.journal.artifacts",
            "artifact ID set differs",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    return MappingProxyType(result)


def _lexical_type(path: Path) -> str:
    path = Path(path)
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "absent"


def _path_state(
    *,
    descriptor: _SugarArtifactDescriptor,
    relative_path: str,
    path: Path,
    expected_identity: str | None,
) -> dict[str, object]:
    lexical_type = _lexical_type(path)
    observed = None
    if lexical_type in {"file", "directory"}:
        try:
            observed = _artifact_identity(descriptor, path)
        except SugarMixturesValidationError:
            observed = None
    relation = (
        "absent"
        if lexical_type == "absent"
        else (
            "expected"
            if expected_identity is not None
            and observed == expected_identity
            else "unknown"
        )
    )
    return {
        "artifact_id": descriptor.artifact_id,
        "relative_path": relative_path,
        "lexical_type": lexical_type,
        "expected_identity_sha256": expected_identity,
        "observed_identity_sha256": observed,
        "relation": relation,
    }


def _recovery_plan_digest(document: Mapping[str, object]) -> str:
    projected = {
        key: value
        for key, value in document.items()
        if key != "recovery_plan_sha256"
    }
    projected["recovery_path"] = (
        Path(str(document["recovery_path"])).name
        if document.get("recovery_path")
        else None
    )
    return hashlib.sha256(
        b"rpe-sugar-recovery-plan-v1\0"
        + _canonical_json_bytes(projected)
    ).hexdigest()


def _inspect_recovery_locked(
    lock: _SugarOutputRootLock,
    staging_parent: Path,
) -> _SugarRecoveryPlan:
    _assert_lock_identity(lock)
    staging_parent = Path(staging_parent)
    if (
        staging_parent.parent != lock.output_root
        or staging_parent.is_symlink()
        or not staging_parent.is_dir()
        or staging_parent.stat().st_dev
        != os.fstat(lock.directory_fd).st_dev
    ):
        raise _fatal(
            "staging_parent",
            "recovery staging must be a direct same-device child",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    journal = _journal_document(staging_parent)
    locked_stat = os.fstat(lock.directory_fd)
    if (
        journal.get("output_root_device") != locked_stat.st_dev
        or journal.get("output_root_inode") != locked_stat.st_ino
    ):
        raise _fatal(
            "recovery.journal.output_root_identity",
            "journal output-root device/inode differs from held lock",
            "PUBLICATION_RECOVERY_REQUIRED",
        )
    artifact_map = _journal_artifact_map(journal)
    finals = []
    backups = []
    unresolved = []
    for descriptor in _ARTIFACT_DESCRIPTORS:
        artifact = artifact_map[descriptor.artifact_id]
        final_state = _path_state(
            descriptor=descriptor,
            relative_path=descriptor.final_basename,
            path=lock.output_root / descriptor.final_basename,
            expected_identity=artifact.get("new_identity_sha256"),
        )
        backup_state = _path_state(
            descriptor=descriptor,
            relative_path=descriptor.backup_basename,
            path=staging_parent / descriptor.backup_basename,
            expected_identity=artifact.get("old_identity_sha256"),
        )
        finals.append(final_state)
        backups.append(backup_state)
        for state in (final_state, backup_state):
            if state["relation"] == "unknown":
                unresolved.append(
                    {
                        "artifact_id": descriptor.artifact_id,
                        "relative_path": state["relative_path"],
                        "reason": "unknown or symlink identity",
                        "safe_automatic_action": None,
                    }
                )
    physical_new_complete = False
    try:
        physical_new_complete = (
            _validate_complete_snapshot(lock.output_root)
            == journal["new_snapshot_sha256"]
        )
    except (SugarMixturesValidationError, OSError, ValueError):
        physical_new_complete = False
    committed = bool(journal["receipt_commit_fsynced"]) or physical_new_complete
    if committed:
        status = "committed_cleanup_required"
        action = "cleanup_committed_snapshot"
        committed_snapshot = "new"
    else:
        status = "recovery_required"
        action = (
            "remove_failed_first_publication"
            if journal["old_snapshot_sha256"] is None
            else "restore_old_snapshot"
        )
        committed_snapshot = None
    if action == "restore_old_snapshot":
        final_by_id = {
            state["artifact_id"]: state
            for state in finals
        }
        backup_by_id = {
            state["artifact_id"]: state
            for state in backups
        }
        for descriptor in _ARTIFACT_DESCRIPTORS:
            artifact = artifact_map[descriptor.artifact_id]
            expected_old = artifact.get("old_identity_sha256")
            final_observed = final_by_id[descriptor.artifact_id][
                "observed_identity_sha256"
            ]
            backup_observed = backup_by_id[descriptor.artifact_id][
                "observed_identity_sha256"
            ]
            if (
                not isinstance(expected_old, str)
                or (
                    final_observed != expected_old
                    and backup_observed != expected_old
                )
            ):
                unresolved.append(
                    {
                        "artifact_id": descriptor.artifact_id,
                        "relative_path": descriptor.backup_basename,
                        "reason": (
                            "required old artifact identity is unavailable"
                        ),
                        "safe_automatic_action": None,
                    }
                )
    document = {
        "status": status,
        "transaction_id": journal["transaction_id"],
        "phase": journal["phase"],
        "committed_snapshot": committed_snapshot,
        "recommended_action": action,
        "recovery_plan_sha256": "",
        "recovery_required": True,
        "recovery_path": str(staging_parent.resolve()),
        "finals": finals,
        "backups": backups,
        "unresolved": unresolved,
    }
    digest = _recovery_plan_digest(document)
    document["recovery_plan_sha256"] = digest
    return _SugarRecoveryPlan(
        document=MappingProxyType(document),
        recovery_plan_sha256=digest,
    )


def _inspect_sugar_recovery(
    *,
    output_root: Path,
    staging_parent: Path,
) -> _SugarRecoveryPlan:
    with _lock_sugar_output_root(output_root) as lock:
        return _inspect_recovery_locked(lock, staging_parent)


def _apply_sugar_recovery(
    *,
    output_root: Path,
    staging_parent: Path,
    action: str,
    transaction_id: str,
    expected_plan_sha256: str,
) -> _SugarRecoveryResult:
    with _lock_sugar_output_root(output_root) as lock:
        plan = _inspect_recovery_locked(lock, staging_parent)
        document = plan.document
        if (
            action
            not in {
                "restore_old_snapshot",
                "remove_failed_first_publication",
                "cleanup_committed_snapshot",
            }
            or action != document["recommended_action"]
            or transaction_id != document["transaction_id"]
            or expected_plan_sha256 != plan.recovery_plan_sha256
            or document["unresolved"]
        ):
            raise _fatal(
                "recovery.apply",
                "action, transaction, plan, or physical identities differ",
                "PUBLICATION_RECOVERY_REQUIRED",
            )
        staging_parent = Path(staging_parent)
        journal = _journal_document(staging_parent)
        artifact_map = _journal_artifact_map(journal)
        removed = []
        restored = []
        if action == "cleanup_committed_snapshot":
            _cleanup_staging_locked(lock, staging_parent)
            final_state = "new_complete"
            receipt_state = "new"
        elif action == "remove_failed_first_publication":
            for artifact_id in _ROLLBACK_REMOVE_ORDER:
                descriptor = _DESCRIPTOR_BY_ID[artifact_id]
                path = lock.output_root / descriptor.final_basename
                if _path_lexists(path):
                    _remove_owned_artifact(
                        descriptor,
                        path,
                        artifact_map[artifact_id]["new_identity_sha256"],
                    )
                    removed.append(descriptor.final_basename)
            _cleanup_staging_locked(lock, staging_parent)
            final_state = "absent"
            receipt_state = "absent"
        else:
            for artifact_id in _ROLLBACK_REMOVE_ORDER:
                descriptor = _DESCRIPTOR_BY_ID[artifact_id]
                path = lock.output_root / descriptor.final_basename
                if _path_lexists(path):
                    _remove_owned_artifact(
                        descriptor,
                        path,
                        artifact_map[artifact_id]["new_identity_sha256"],
                    )
                    removed.append(descriptor.final_basename)
            for artifact_id in _RESTORE_PAYLOAD_ORDER:
                descriptor = _DESCRIPTOR_BY_ID[artifact_id]
                backup = staging_parent / descriptor.backup_basename
                if _path_lexists(backup):
                    os.replace(
                        backup,
                        lock.output_root / descriptor.final_basename,
                    )
                    restored.append(descriptor.final_basename)
            descriptor = _DESCRIPTOR_BY_ID["receipt"]
            backup = staging_parent / descriptor.backup_basename
            if _path_lexists(backup):
                os.replace(
                    backup,
                    lock.output_root / descriptor.final_basename,
                )
                restored.append(descriptor.final_basename)
            _cleanup_staging_locked(lock, staging_parent)
            final_state = "old_complete"
            receipt_state = "old"
        result = {
            "status": "recovered",
            "transaction_id": transaction_id,
            "action": action,
            "recovery_plan_sha256": expected_plan_sha256,
            "final_state": final_state,
            "receipt_state": receipt_state,
            "removed_paths": removed,
            "restored_paths": restored,
            "remaining_recovery_paths": [],
        }
        return _SugarRecoveryResult(document=MappingProxyType(result))


def _read_sugar_snapshot_locked(
    output_root: Path,
    operation: Callable[[Mapping[str, object]], object],
):
    with _lock_sugar_output_root_shared(output_root):
        receipt_path = (
            Path(output_root)
            / _DESCRIPTOR_BY_ID["receipt"].final_basename
        )
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise _fatal(
                "receipt",
                "committed receipt is absent",
                "SNAPSHOT_UNAVAILABLE",
            )
        _, document = _read_canonical_receipt(receipt_path)
        _validate_complete_snapshot(Path(output_root))
        return operation(document)


def _read_sugar_snapshot_lock_free(
    output_root: Path,
    operation: Callable[[Mapping[str, object]], object],
):
    receipt_path = (
        Path(output_root)
        / _DESCRIPTOR_BY_ID["receipt"].final_basename
    )
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise _fatal(
            "receipt",
            "committed receipt is absent",
            "SNAPSHOT_UNAVAILABLE",
        )
    first, document = _read_canonical_receipt(receipt_path)
    _validate_complete_snapshot(Path(output_root))
    result = operation(document)
    try:
        second = receipt_path.read_bytes()
    except OSError as error:
        raise _fatal(
            "receipt",
            str(error),
            "READER_SNAPSHOT_CHANGED",
        ) from error
    if first != second:
        raise _fatal(
            "receipt",
            "receipt changed between reader checks",
            "READER_SNAPSHOT_CHANGED",
        )
    return result
