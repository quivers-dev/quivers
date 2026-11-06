# Public API and Exports

This page states which names Quivers exports, where each is imported
from, and what a contributor must do when adding, moving, or removing
one. The rules are enforced: `tests/test_public_api.py` checks the
import-time rules on every test run, and
`tools/check_api_reference.py` checks the rendered API reference after
`mkdocs build` in the docs job. A change that breaks a rule fails CI.

For a user, this policy identifies imports that should survive internal file
reorganization. For a contributor, it identifies the objects that must remain
constructible, documented, and testable. If a [step record](../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram) appears in a compiled
program, a Python user should be able to import that record and inspect it
without reaching into an internal module.

A re-export makes one object available through another module name; it does
not create a second implementation. Thus `from quivers.continuous import
PlateDraw` stays supported if the defining leaf module moves within that
package. The rules below make this practical promise mechanically checkable.

## Terms

A module is *public* when no component of its dotted path begins with
an underscore: `quivers.continuous.inline` is public, while
`quivers.continuous._ordered` and `quivers.dsl._grammar_data` are
private. A *package* is a module with an `__init__.py`; a *leaf module*
is any other module. The *public API* is the union of the `__all__`
lists of the public packages, and a name is public exactly when it
appears in one of them.

## Rules

1. **Import from the package.** The supported import path for a public
   name is the public package whose `__all__` lists it, as in
   `from quivers.continuous import PlateDraw`. A leaf-module path such
   as `quivers.continuous.plate` records where the name is defined;
   it is not part of the contract, so a name may move between the
   leaf modules of its package without a breaking change.

2. **Every public module declares `__all__`.** The list names the
   module's public interface, whether defined in the module or
   re-exported from elsewhere. A module that holds implementation only
   declares `__all__ = []`, which states that it contributes nothing to
   the public API. Names absent from `__all__` are internal however
   they are spelled; other Quivers modules may import them, and user
   code should not.

3. **Packages re-export their leaf modules.** For every public leaf
   module `M` directly inside a public package `P`, each name in
   `M.__all__` is in `P.__all__` and is bound to the same object. A
   subpackage is a package in its own right, so `P` may re-export a
   subpackage's names but need not. Two leaf modules of one package
   thus cannot export different objects under one name. A package
   may bind a re-export lazily through a module-level `__getattr__`
   when an eager import would create a cycle.

4. **No private names in `__all__`.** An underscore-prefixed name is
   never listed, and a list holds no duplicates.

5. **Public signatures name public types.** When a public function,
   or a public class's constructor or public method, annotates a
   parameter or return value with a Quivers type, that type is public.
   Every such annotation resolves at runtime under
   `typing.get_type_hints`. A caller can thus construct every argument
   a public call accepts and name every value it returns without
   importing from an internal location.

6. **Anything the compiler builds, Python can build.** Every
   [`Morphism`][quivers.core.Morphism] and
   [`ContinuousMorphism`][quivers.continuous.ContinuousMorphism]
   subclass the DSL compiler places in a compiled program is public,
   as are the step records a
   [`MonadicProgram`][quivers.continuous.MonadicProgram] is built
   from. Python code can thus assemble any program the compiler
   produces, and the objects the compiler produces are a contract
   rather than an implementation detail. The test compiles every
   gallery example and checks the classes of the morphisms it finds.
   The `nn.Module` parameter containers that
   [`as_torch_module`][quivers.core.as_torch_module] wraps morphisms in
   are not morphisms and fall outside this rule.

7. **Every public name is documented.** Each public name carries a
   docstring (for a constant or type alias, a string literal on the
   line after its assignment, which is where the documentation
   generator reads attribute docstrings) and renders in the API reference under `docs/api/`,
   either through a `:::` directive on a package that exports it or on
   the module that defines it. `tools/check_api_reference.py` reads the
   built site and fails on any public name without an anchor at one of
   those paths.

## Deciding what is public

The rules fix how exports are spelled; this rubric decides which names
a module lists. Export a name when a user of the Python API needs it to
use a capability without the DSL:

- classes a user constructs, subclasses, or receives and inspects;
- functions a user calls;
- protocols and abstract base classes a user implements;
- registries and registration functions a user extends, such as
  family specifications and renderer hooks;
- exceptions a public call raises;
- constants that parameterize a public call or name one of its
  results, such as algebra singletons.

Keep internal the passes and helpers whose contract is "what another
Quivers module needs today": parser walkers, compiler mixins, lowering
and elaboration helpers, renderer internals, and generated-code
scaffolding. These belong in a module whose `__all__` omits them, or in
an underscore-prefixed module.

## Exceptions

The target runtime sources `quivers/transpile/runtime_*` are files
grafted into generated programs, read by path next to their `.jl`,
`.scm`, `.js`, and `.stan` siblings. They are data rather than modules
of the library, nothing imports them, and the rules above do not apply
to them. The test checks that no Quivers module imports one.

## Changing the public API

Adding a public name is an addition and goes under **Added** in the
changelog. Renaming or removing one, or changing a public signature
incompatibly, goes under **Changed** or **Removed**, and while Quivers
is below 1.0 it takes a minor version bump. Moving a name between leaf
modules of one package changes nothing a user imports and needs no
entry.

## Building programs in Python

Rule 6 is what makes the DSL optional. The
[programs guide](../guides/programs.md#building-programs-in-python)
shows one program assembled from the exported builders, using the same
step records and inline distributions the compiler emits.
