"""Pinned identities for every immutable QVR grammar migration asset.

The manifest makes release aliases explicit and prevents an identity hop from
silently accepting a changed parser snapshot or panproto schema. Updating a
grammar therefore requires updating its migration edge and this manifest in
the same change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import didactic.api as dx


class GrammarAsset(dx.Model):
    """Content identities of one immutable grammar snapshot.

    Parameters
    ----------
    grammar_json_sha256 : str
        Digest of the grammar JSON.
    node_types_sha256 : str
        Digest of the node-type declarations.
    parser_c_sha256 : str
        Digest of the generated C parser.
    compiled_inputs_sha256 : str
        Digest of the complete parser compilation inputs.
    """

    grammar_json_sha256: str
    node_types_sha256: str
    parser_c_sha256: str
    compiled_inputs_sha256: str


GRAMMAR_ASSETS: dict[str, GrammarAsset] = {
    "v0.2.0": GrammarAsset(
        grammar_json_sha256="a3589f11b2b4b4845f0a8130649607355cd1bd63505c3ba72040666876bbca03",
        node_types_sha256="530eb0dd61b8963c6294ce34265336db6520dc927ad706aceed679b331eb7b57",
        parser_c_sha256="9932756c394deb0a71f3e7718e9894fcc9d9396ccfd5e6771168d65c59d9a02d",
        compiled_inputs_sha256="2b451235635bfe315e0640b37c31291ac6955f6aced6c989436ee038f5ba8d06",
    ),
    "v0.3.0": GrammarAsset(
        grammar_json_sha256="2166403700088b266a46ab2d5fd288dcac16805cf6644f064ac7edd189bd4be2",
        node_types_sha256="17bea32a0e86f484e1400fc2e0861cd974abffe8951a9736de38fe6ea5eb14c7",
        parser_c_sha256="0f0ceec9965c09f3dbf4083869ce9146529a54ccef3428f7dd8a872427e7941e",
        compiled_inputs_sha256="1d46811c1a344bda8f5d502c625554d1ea2a4ead60b666107b43a4f8f75e0b7e",
    ),
    "v0.4.0": GrammarAsset(
        grammar_json_sha256="2f9e6286a72d5526fbd6fcbcc72a620f75b9804ae1060180e039694e24b0bdd6",
        node_types_sha256="360196e3408913c63db36c5f8200ccd98afb9722128949159a486e3c0430e469",
        parser_c_sha256="adba4b436fc4816f46cded9513b2a3609b9b82cda22b773d8027ed19b9539d4c",
        compiled_inputs_sha256="1b04bc7166cfc2e0cbee2ce8fb0cc27be830d411b1ce95f99ef607b41a74a129",
    ),
    "v0.5.0": GrammarAsset(
        grammar_json_sha256="16a2d2ab335edf25160bbeac0fa93f5491139c9ba4fd091ad172769fb62e058f",
        node_types_sha256="ba9e5ca906cacc8e8afdcb534f4f064ee81781b2b3b556c40eea5f50258e6830",
        parser_c_sha256="c6520602f1933b3cb3217fae9b83786d3469d5db3857edbd3e3c5e215bc8b20e",
        compiled_inputs_sha256="0019176286e374d80471c3930d9c1baaa9b00b62467df17afefb006be2289f4e",
    ),
    "v0.6.0": GrammarAsset(
        grammar_json_sha256="e384519a654880e10c32f2eef871597513eafbf911761433f78c31c616516de0",
        node_types_sha256="9ba270b4b303acd22a1d6fb136b42ceb46f23efef3089647357af786d117d02c",
        parser_c_sha256="200813845596dc3185fb3a4977bec130c8cdbb181103e8c247fb1394ddbb1059",
        compiled_inputs_sha256="acf49f3be2399429661c149b679ca690d99b0a2dc94f463dd1f5936e36a53707",
    ),
    "v0.7.0": GrammarAsset(
        grammar_json_sha256="5249be1f1fe6cb2253e712028a2e2013b2c0afc96ccdb94fe225b575e84303f0",
        node_types_sha256="ce638af8faefab881a90428b033eefa3b2bbd766e2d755a00b35c4280451916c",
        parser_c_sha256="d713b001d009386a285c47b94fcae779ab5e16f5370570b7d445c0f952011c47",
        compiled_inputs_sha256="8948662ca1773562d563421cf5574c42c2cd7ed9bc7beef238adfa369181e52b",
    ),
    "v0.9.0": GrammarAsset(
        grammar_json_sha256="fe9188c219012f83fe3d7a1b06d8f2b0e189fbbe56e38ce6df4ae0a156851e01",
        node_types_sha256="48d938e234f479061c3f4e8b5547d3a427018c709276dcb7ccaeeb5d2ff88cac",
        parser_c_sha256="2a2cac1958a8254d62e62605fc35487ac0b6c67560449687adccfb38f6f6176f",
        compiled_inputs_sha256="deccd74d60aa513689f1ea4b47e0dd86ab32719f935685458a0a47bd6d8cf6c1",
    ),
    "v0.11.0": GrammarAsset(
        grammar_json_sha256="7a2b858cbe3171e10faa85d7d48903b750ed3aa391c69de62e0cebe2e4af3068",
        node_types_sha256="9114383417ac38cf861f32f091f8c47826744c556aff9663cdcbedc826fcb6dc",
        parser_c_sha256="7d5e2e81195073ca8bd419e90672e180f68cd8264934ae00da39b8d61c592a3c",
        compiled_inputs_sha256="c1b6e9465e442a9fc0452351dc3008c69e7f858dcdba1ebdbfdb271710f0869a",
    ),
    "v0.14.0": GrammarAsset(
        grammar_json_sha256="029b1612017ce9214d46433e7fe6247adc31cd031a74bae546cb2f1b0770be9d",
        node_types_sha256="e47c2b2530a48a09e6266216d022c115abfe9300240b387ace75067fedfa1fa9",
        parser_c_sha256="014799053caecabba63dcd1603ccf160b065717b5019c5aca279c83fd8ae7b3d",
        compiled_inputs_sha256="efa947844bd465fe5ea53ea235e98a3cc07e68a86c82a94ca63ac568de854acd",
    ),
    "v0.15.0": GrammarAsset(
        grammar_json_sha256="f76a0d8c71f1d3a13c15c61b8594e6c679624f67d2e6534e1ba70d4c55f07260",
        node_types_sha256="47cb59716ba6abb5ab1d621e8a949854b6f61b2da676a60de2b594efb3f6dd9f",
        parser_c_sha256="814b5b2dc75fbd7ade46fa02e5772299f51ecfa76a9cfc3e93cc0efc68771eba",
        compiled_inputs_sha256="ffae70e999fecc8ed33058ac2d680bc6d71c8bf7888bd8db0a33404693f4386b",
    ),
    "v0.19.0": GrammarAsset(
        grammar_json_sha256="7a2463bb21935f683d7f977ec23ac91f5b84eda13826cc95c4e90bb76a32afa9",
        node_types_sha256="62f396e779b6ff7b19a51d17a20ce71886580defbdf5be87cd7c77b9bd831f84",
        parser_c_sha256="48aaab9589adb57c8c1c517adfa4eeb206ac2192b795e92825ea36209d412e20",
        compiled_inputs_sha256="7c273ca205f8e27dc55ce5167f2f3e2ece947753fef9a52a7b02257eb23733bc",
    ),
    "v0.62.0": GrammarAsset(
        grammar_json_sha256="cd7b2f1015bd9c4acbd0c943de6a2b255449fc0dfde686a60c02393349b78dd7",
        node_types_sha256="a3372ef2c0ba3c5aada4a70bb833723d038b6a31f2a52e425e0f55e9d7fff55f",
        parser_c_sha256="22c4deecb689048a9cb510b89c0af708da4f03d326d4f2bd7843856a70b3bef2",
        compiled_inputs_sha256="8f3a10111b43a4740361b664edccbcbd0fee30915e48d7c13586d7513f07300e",
    ),
    "HEAD": GrammarAsset(
        grammar_json_sha256="cd7b2f1015bd9c4acbd0c943de6a2b255449fc0dfde686a60c02393349b78dd7",
        node_types_sha256="a3372ef2c0ba3c5aada4a70bb833723d038b6a31f2a52e425e0f55e9d7fff55f",
        parser_c_sha256="22c4deecb689048a9cb510b89c0af708da4f03d326d4f2bd7843856a70b3bef2",
        compiled_inputs_sha256="8f3a10111b43a4740361b664edccbcbd0fee30915e48d7c13586d7513f07300e",
    ),
}


REVISION_SNAPSHOTS: dict[str, str] = {
    "v0.2.0": "v0.2.0",
    "v0.3.0": "v0.3.0",
    "v0.4.0": "v0.4.0",
    "v0.5.0": "v0.5.0",
    "v0.6.0": "v0.6.0",
    "v0.7.0": "v0.7.0",
    "v0.9.0": "v0.9.0",
    "v0.10.0": "v0.9.0",
    "v0.11.0": "v0.11.0",
    "v0.14.0": "v0.14.0",
    "v0.15.0": "v0.15.0",
    "v0.16.0": "v0.15.0",
    "v0.17.0": "v0.15.0",
    "v0.18.0": "v0.15.0",
    "v0.19.0": "v0.19.0",
    "v0.62.0": "v0.62.0",
    "HEAD": "HEAD",
}


SCHEMA_COMMITS: dict[str, str] = {
    "v0.2.0": "6f196541a2ab2aa2da5c7b15092b830223182c560906c4e8cc6bb48ef2ed800b",
    "v0.3.0": "2a07e8c196d5940d1c28f214483226d5a81f0c786e33762c2589d83aecdf0e7c",
    "v0.4.0": "64d16c54715f2e17c3a2fee3e99937385a0d2725223c386bcdd3fab0928e5f86",
    "v0.5.0": "eee9a69337072ce17cd87d7f55a13f7a1962bc9ef481d420ea645629421891e7",
    "v0.6.0": "e7f3c52f5f620e257916f3309b491f2ba28bdc99f61e65c01fc09160fccb56fc",
    "v0.7.0": "4dc96fd3f603a33beccd12ebe98714255aa5054c17b6eb6d82359dc3d217e2eb",
    "v0.9.0": "3634493012b19a7d633cd83077c6641b04038b486da27a3c9994de74bbf2d41b",
    "v0.10.0": "3634493012b19a7d633cd83077c6641b04038b486da27a3c9994de74bbf2d41b",
    "v0.11.0": "ab721b802bdefea5151d1f1ff79df38c582f29d71e9eef66bd73e362c8ad12ab",
    "v0.14.0": "0987657b46c47c394d23f2d24ba4c75046e70538dbb7583a99c3dabdf0f803de",
    "v0.15.0": "56fde386d73bc4667726dc29f4a17ceeae070ed2a3effd10695a69c41c0a3079",
    "v0.16.0": "56fde386d73bc4667726dc29f4a17ceeae070ed2a3effd10695a69c41c0a3079",
    "v0.17.0": "56fde386d73bc4667726dc29f4a17ceeae070ed2a3effd10695a69c41c0a3079",
    "v0.18.0": "56fde386d73bc4667726dc29f4a17ceeae070ed2a3effd10695a69c41c0a3079",
    "v0.19.0": "b4b20dd709fb0466384f12e5c9fbfaf959013a426bbb82ba0806a8d75ac2877e",
    "v0.62.0": "2378d1a97cf516892cb9b4484300fb9cb91e40fc5a91b9131821c9dac43c65f4",
    "HEAD": "2378d1a97cf516892cb9b4484300fb9cb91e40fc5a91b9131821c9dac43c65f4",
}


def snapshot_for(revision: str) -> str:
    try:
        return REVISION_SNAPSHOTS[revision]
    except KeyError as error:
        raise ValueError(
            f"revision {revision!r} is absent from the grammar manifest"
        ) from error


def revisions_are_identical(source_revision: str, target_revision: str) -> bool:
    """Whether the manifest asserts one grammar and schema identity."""
    source_snapshot = snapshot_for(source_revision)
    target_snapshot = snapshot_for(target_revision)
    return (
        GRAMMAR_ASSETS[source_snapshot] == GRAMMAR_ASSETS[target_snapshot]
        and SCHEMA_COMMITS[source_revision] == SCHEMA_COMMITS[target_revision]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiled_inputs_sha256(src_dir: Path) -> str:
    """Hash every source/header byte that contributes to the shared library."""
    paths = [src_dir / "parser.c"]
    scanner = src_dir / "scanner.c"
    if scanner.is_file():
        paths.append(scanner)
    paths.extend(sorted((src_dir / "tree_sitter").rglob("*.h")))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(src_dir).as_posix()):
        relative = path.relative_to(src_dir).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_snapshot(snapshot: str, src_dir: Path) -> None:
    """Reject a parser source tree whose pinned files have drifted."""
    try:
        expected = GRAMMAR_ASSETS[snapshot]
    except KeyError as error:
        raise ValueError(
            f"snapshot {snapshot!r} is absent from the grammar manifest"
        ) from error
    actual = GrammarAsset(
        grammar_json_sha256=_sha256(src_dir / "grammar.json"),
        node_types_sha256=_sha256(src_dir / "node-types.json"),
        parser_c_sha256=_sha256(src_dir / "parser.c"),
        compiled_inputs_sha256=compiled_inputs_sha256(src_dir),
    )
    if actual != expected:
        raise ValueError(
            f"immutable parser snapshot {snapshot!r} does not match its hash manifest: "
            f"expected {expected!r}, got {actual!r}"
        )


__all__ = [
    "GRAMMAR_ASSETS",
    "REVISION_SNAPSHOTS",
    "SCHEMA_COMMITS",
    "GrammarAsset",
    "compiled_inputs_sha256",
    "revisions_are_identical",
    "snapshot_for",
    "verify_snapshot",
]
