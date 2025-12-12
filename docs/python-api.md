# Python API Reference

This page is the landing spot for the auto-generated Python reference. It is produced by **MkDocs** + **mkdocstrings** using the `t81lib` bindings that live in `python/bindings.cpp`.

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

## Reference snippets

The following mkdocstrings directives pull the public bindings into the docs:

```markdown
::: t81lib
    handler: python
```

To spotlight the higher-level helpers e.g. the PyTorch bridge, add more directives below:

```markdown
::: t81
    handler: python
    render_toc: true
```

When publishing the docs, MkDocs bundles these sections into the static site so the generated API reference lives alongside the rest of the portal.

::: t81lib
    handler: python
    render_toc: true
