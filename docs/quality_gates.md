# Quality Gates

MicroI2I must behave like a production-quality scientific software project.
Quality gates protect scientific trust, migration safety, and teaching value.

## Required Local Gate

Run:

```bash
python scripts/check_repo.py
```

This command runs:

- unit and integration tests,
- model registry validation,
- Sphinx HTML documentation build.

## Required Direct Commands

These commands should also remain valid individually:

```bash
python -m pytest tests
python scripts/microi2i_cli.py validate-registry
python scripts/build_docs.py --html-only
```

## Feature Gates

New workflow:

- typed config contract
- unit tests
- CLI integration test
- manifest/report artifacts
- Sphinx usage documentation

New metric:

- mathematical formula
- unit test with known expected values
- interpretation and limitation notes
- inclusion in evaluation report schema

New model backend:

- backend interface implementation
- config preset
- registry metadata
- architecture documentation
- smoke path or documented reason why smoke execution is not feasible

Legacy migration:

- current behavior inventory
- regression tests
- equivalent `microi2i` command
- docs update
- removal note

## Failure Policy

If a gate fails, do not merge unless the failure is explicitly documented with:

- failing command,
- reason,
- risk,
- follow-up issue or roadmap entry.
