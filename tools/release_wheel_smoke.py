"""Exercise an installed Quivers wheel without parser compiler fallbacks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import tempfile

import torch

import quivers
from quivers.cli.migrations import CHAIN, compose_migration, vcs_coverage_report
from quivers.cli.migrations import _grammar as migration_grammar
from quivers.dsl import parse
from quivers.dsl import _grammar_build as grammar_build
from quivers.dsl.pygments_lexer import QvrLexer
from quivers.cli import check_file as check_one
from quivers.dsl import Compiler
from quivers.qiec import invoke_entry
from quivers.modeling import Model
from quivers.transpile import available_targets, transpile


_EXPECTED_DEPENDENCIES = {
    "didactic": "0.18.2",
    "panproto": "0.77.0",
    "panproto-grammars-all": "0.77.0",
}


def _check_dependency_versions() -> None:
    """Require the release smoke's intentionally pinned dependency set."""

    for distribution, expected in _EXPECTED_DEPENDENCIES.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError as error:
            raise SystemExit(
                f"installed-wheel smoke is missing {distribution}=={expected}"
            ) from error
        if actual != expected:
            raise SystemExit(
                "installed-wheel smoke requires "
                f"{distribution}=={expected}, found {actual}"
            )


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    package = Path(quivers.__file__).resolve()
    if package.is_relative_to(project / "src"):
        raise SystemExit(f"smoke test imported the source checkout: {package}")
    _check_dependency_versions()

    def no_current_compiler() -> str:
        raise AssertionError("current parser attempted a source build")

    def no_migration_compiler(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("migration parser attempted a source build")

    grammar_build._compiler = no_current_compiler
    migration_grammar._compile_parser = no_migration_compiler

    module = parse("index Nat = Z | S(Nat)\n")
    if len(module.statements) != 1:
        raise SystemExit("installed-wheel parser returned the wrong module")
    if not list(QvrLexer().get_tokens("index Nat = Z | S(Nat)\n")):
        raise SystemExit("installed-wheel highlighter returned no tokens")

    qiec_source = parse("define answer() : Int !{} =\n    return 42\n")
    rendered: dict[str, bytes] = {}
    for target in available_targets():
        rendered[target] = transpile(qiec_source, target=target)
        if b"qiec_answer" not in rendered[target]:
            raise SystemExit(
                f"installed-wheel {target} transpiler omitted QIEC computation"
            )
    namespace: dict[str, object] = {}
    exec(rendered["pyro"], namespace)
    if namespace["qiec_answer"]() != 42:  # type: ignore[operator]
        raise SystemExit("installed-wheel Python QIEC runtime returned wrong value")

    checked = Compiler(qiec_source, module_name="smoke", file_path="smoke.qvr")
    module = checked.qiec_module
    if module is None:
        raise SystemExit("installed-wheel compiler produced no checked module")
    if invoke_entry(module, "answer").value != 42:
        raise SystemExit("installed-wheel reference machine returned wrong value")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "model.qvr"
        path.write_text(
            "object Obs : FinSet 4\n\n"
            "program prog : Obs -> Obs\n"
            "    sample a <- Normal(loc=0.0, scale=1.0)\n"
            "    observe y : Obs <- Normal(loc=a, scale=0.5)\n"
            "    return a\n"
            "export prog\n"
        )
        if check_one(path):
            raise SystemExit("installed-wheel check reported diagnostics on a program")
        program = Compiler(
            parse(path.read_bytes(), file_path=str(path)),
            module_name="model",
            file_path=str(path),
        ).qiec_module
        if program is None:
            raise SystemExit("installed-wheel compiler produced no program entry")
        run = invoke_entry(
            program, "prog", ((0.1, 0.2, 0.3, 0.4),), sites={"a": 0.5}, seed=0
        )
        if run.value != 0.5 or run.log_joint is None:
            raise SystemExit("installed-wheel program run returned the wrong value")
        symbolic = (
            path.read_text()
            .replace("FinSet 4", "FinSet _")
            .replace("program prog :", "program prog(y : Tensor[Real]([Obs])) :")
        )
        model = Model(symbolic)
        bound = model.bind({"y": torch.tensor([0.1, 0.2, 0.3, 0.4])})
        if bound.schema.describe()["y"] != {
            "role": "observation",
            "type": "Tensor[Real]([4])",
        }:
            raise SystemExit("installed-wheel model lost its specialized data contract")
        recorded = bound.trace({"a": torch.tensor([0.5])})
        if (
            recorded.output.item() != 0.5
            or not torch.isfinite(recorded.log_joint).all()
        ):
            raise SystemExit("installed-wheel model trace returned an invalid result")
        dependent = Model(
            "object Out : Real 1\n"
            "program dependent[n : Nat](count : Nat, y : Tensor[Real]([n])) : n -> Out\n"
            "    sample a <- Normal(loc=0.0, scale=1.0)\n"
            "    let mean = a + count\n"
            "    observe y : n <- Normal(loc=mean, scale=1.0)\n"
            "    return a\n"
            "export dependent\n"
        )
        dependent_data = {"count": torch.tensor(2), "y": torch.ones(3)}
        bound = dependent.bind(dependent_data)
        if bound.schema.describe()["count"].get("constraint") != "nonnegative":
            raise SystemExit("installed-wheel model lost its runtime Nat constraint")
        if bound.schema.describe()["y"]["type"] != "Tensor[Real]([3])":
            raise SystemExit("installed-wheel model did not infer its static extent")
        if not torch.isfinite(bound.trace({"a": torch.zeros(1)}).log_joint).all():
            raise SystemExit(
                "installed-wheel dependent program returned invalid density"
            )
        try:
            dependent.bind({**dependent_data, "count": torch.tensor(-1)})
        except TypeError, ValueError:
            pass
        else:
            raise SystemExit("installed-wheel runtime accepted a negative Nat input")

    count_source = """object Row : FinSet _
program counts(trials: Tensor[Int]([Row]), p: Real, y: Tensor[Int]([Row])) : Row -> Row
    observe y : Row <- ZeroInflatedBinomial(zero_prob=0.25, total_count=trials, probs=p)
    return y
export counts
"""
    probability = torch.tensor(0.5, requires_grad=True)
    count_data = {
        "trials": torch.tensor([0, 3]),
        "p": probability,
        "y": torch.tensor([0, 1]),
    }
    count_model = Model(count_source)
    count_trace = count_model.bind(count_data).trace()
    expected = torch.tensor(9 / 32, dtype=torch.float64).log()
    torch.testing.assert_close(count_trace.log_joint.reshape(()), expected)
    torch.testing.assert_close(
        torch.autograd.grad(count_trace.log_joint.sum(), probability)[0],
        torch.tensor(-2.0),
    )
    checked_counts = Compiler(
        parse(count_source.replace("FinSet _", "FinSet 2"))
    ).qiec_module
    if checked_counts is None:
        raise SystemExit("installed-wheel count program has no checked entry")
    count_run = invoke_entry(checked_counts, "counts", ((0, 3), 0.5, (0, 1)))
    if (
        count_run.log_joint is None
        or abs(count_run.log_joint - expected.item()) > 1e-12
    ):
        raise SystemExit("installed-wheel native/reference count scores disagree")
    try:
        count_model.bind({**count_data, "trials": torch.tensor([0, 2**53 + 1])}).trace()
    except ValueError as error:
        if "2**53" not in str(error):
            raise
    else:
        raise SystemExit("installed-wheel count kernel rounded an unsupported integer")
    exact_module = Compiler(
        parse("define exact_integer() : Int !{} =\n    return 9007199254740993\n")
    ).qiec_module
    if (
        exact_module is None
        or invoke_entry(exact_module, "exact_integer").value != 2**53 + 1
    ):
        raise SystemExit("installed-wheel compiler rounded an exact integer literal")

    available = set(migration_grammar.available_revisions())
    missing = set(CHAIN) - available
    if missing:
        raise SystemExit(f"wheel is missing parser assets for: {sorted(missing)}")
    for revision in CHAIN:
        migration_grammar.registry_for(revision)

    source = b"bundle installed_wheel = [first, second]\n"
    expected = b"bundle installed_wheel : [first, second]\n"
    migrated = compose_migration("v0.14.0", "HEAD")(source)
    if migrated != expected:
        raise SystemExit(f"installed-wheel migration mismatch: {migrated!r}")

    uncovered = [report for report in vcs_coverage_report() if report.uncovered_removed]
    if uncovered:
        raise SystemExit(f"migration coverage has uncovered rules: {uncovered!r}")


if __name__ == "__main__":
    main()
