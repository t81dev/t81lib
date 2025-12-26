# Python API Reference

This page is the landing spot for the auto-generated Python reference. It is produced by **MkDocs** + **mkdocstrings** using the `t81lib` bindings that live in `python/bindings.cpp`.

## Python entry points

| Entry point | Use it when you want to... | Start here |
| --- | --- | --- |
| `t81lib` | access the low-level pybind11 bindings (BigInt, Limb, packing, GGUF helpers) | `import t81lib` |
| `t81` | use the high-level re-exported API without worrying about missing bindings | `import t81 as t8` |
| `t81.torch` | run ternary tensor ops or integrate with Torch tensors | `from t81 import torch as t81torch` |
| `t81.nn` | swap in ternary-aware layers (e.g., `Linear`) | `from t81 import nn as t81nn` |
| `t81.convert` / `t81.gguf` | call the conversion/GGUF helpers programmatically | `from t81 import convert, gguf` |
| `t81.hardware` | explore ternary hardware emulation helpers | `from t81 import hardware` |

## Generating the docs

1. Install the tooling (ideally in a virtual environment):
   ```bash
   python3 -m pip install mkdocs mkdocstrings mkdocstrings[python]
   ```
2. From the repository root, run:
   ```bash
   pip install -e .
   mkdocs build --clean
   ```
   `mkdocstrings` will import `t81lib`, introspect the pybind11-exposed symbols, and render them into this page.
3. Serve a live preview with `mkdocs serve` while editing docstrings or annotations, and check the generated API output under `site/python-api/`.

## API surface sync checklist

- Confirm `python/t81/__init__.py` `_binding_exports` still matches the public names in `t81lib`.
- Update the entry-points tables if new modules or helpers are added.
- Add or refresh mkdocstrings directives when new stable submodules should appear in the API docs.

## Reference snippets

The following mkdocstrings directives pull the public bindings into the docs:

```markdown
::: t81lib
    handler: python
```

To spotlight the higher-level helpers (when extras are installed), add more directives below:

```markdown
::: t81
    handler: python
    options:
        members: []
```

When publishing the docs, MkDocs bundles these sections into the static site so the generated API reference lives alongside the rest of the portal.

::: t81lib
    handler: python
    render_toc: true

::: t81
    handler: python
    options:
        members: []
