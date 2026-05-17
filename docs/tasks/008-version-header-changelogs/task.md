# Task: Extract per-file `# VERSION` headers and in-docstring `Modification history:` blocks into `changelogs/`

> **Status:** Draft
> **Owner:** kouui
> **Created:** 2026-05-17
> **Last Updated:** 2026-05-17

## Objective

Move all change-history metadata that currently lives inside `src/spectra/**/*.py` source files into per-date markdown files under `./changelogs/`. The history blocks waste tokens every time a coding agent loads a source file. Git already records authoritative version history; the in-file blocks duplicate that and rot.

This task delivers the extracted changelog files. **Source files are NOT touched in this task**; the deletion of in-file history blocks is a follow-up task gated on user confirmation that the changelogs are complete.

## Background & Context

GitHub issue [#8](../../../../issues/8): _"want to remove `history` in *.py files — purpose: waste of tokens when file loaded by coding agent."_ The issue requires:

1. Duplicated logs across files may be merged.
2. Do not touch source code; handle the `history` comment only.
3. Create the changelogs first; do not remove history blocks until the changelogs are confirmed correct and complete.

Two kinds of history block exist in the codebase:

- **File-level `# VERSION` header**, in 52 of 71 source files. Format:
  ```
  # -------------------------------------------------------------------------------
  # <file purpose>
  # -------------------------------------------------------------------------------
  # VERSION
  # 0.1.1
  #    2021/06/07   u.k.
  #        - modified HI_rayleigh_cross_sec_
  #        - in _avH2p_, cubic v.s. linear : 3x difference
  # 0.1.0
  #    2021/05/18   u.k.   spectra-re
  # -------------------------------------------------------------------------------
  ```
  Each entry has `YYYY/MM/DD`, author tag (`u.k.` or `k.i.`), an optional short line, and optional `-` sub-bullets. Versions are not always chronologically ordered. Some files mix a `# WARNING` block after `# VERSION` — `# WARNING` is **not** history and is left untouched.

- **In-docstring `Modification history:` block**, in 2 files — `src/spectra/Atomic/ContinuumOpacity.py` (8 occurrences, one per function) and `src/spectra/Atomic/LTELib.py` (1 occurrence). Format (excerpt from `ContinuumOpacity.py:125`):
  ```
  Modification history:

  - k.ichimoto 15 jun.1987,    6 Jan.1992
  - k.ichimoto 19 Feb.1994
  - 2019.9.15   k.ichimoto from IDL ahic.pro
  ```
  These appear inside Sphinx docstrings of public functions.

Design decisions agreed in discussion (recorded in `plan.md` §5):

- **Two-file layout under `changelogs/`** — a live `changelogs/changelog.md` for ongoing work, plus rotated archives at `changelogs/archives/changelog_<YYYYMMDD>.md`. The archive filename uses the **most recent date inside that file** as the suffix. Initial migration packs all existing dated entries into archives only; `changelog.md` starts empty (just a header / rotation instructions).
- **Per-archive line budget** of **≤ 250 lines** (a pre-commit hook enforces a hard cap of 300 — see Functional Requirements). Within each archive, dates are ordered **newest-first**.
- **Drop version numbers entirely** — git carries version provenance; the changelog is ordered by date only.
- **Merge duplicates**: when the same `(date, author, header description)` appears in multiple files AND none of them has sub-bullets (e.g. the `2021/05/18 u.k. spectra-re` initial-migration line across ~39 files), merge into a single entry that lists the affected files.
- **Sub-bullets preserved verbatim** (Option A from discussion) — fidelity over compactness.
- **In-docstring `Modification history:` entries** are extracted, get a `func <function_name>:` prefix on their description, and are merged into the same archive date block as the file-level entries.
- **Author normalization**: `u.k.` → `YW.Huang`; `k.i.` / `k.ichimoto` → `k.ichimoto`; `j.n.` → `j.natsume`. Unknown authors are kept verbatim and flagged by the extractor.
- **Date normalization**: `YYYY/MM/DD` → `YYYYMMDD`; `YYYY.M.D` → `YYYYMMDD`; `D mon.YYYY` → `YYYYMMDD`.
- **Skip pre-port IDL provenance**: entries inside in-docstring history that describe original-code dates (e.g. `- k.ichimoto 15 jun.1987, 6 Jan.1992`, `- k.ichimoto 19 Feb.1994`) are upstream IDL metadata, not work performed in this project. They are dropped. Pattern: the line begins with an author tag and lists one-or-more `D mon.YYYY`-style dates with no `YYYY.M.D` lead-in.

## Requirements

### Functional Requirements

1. **Create `changelogs/` layout** at the repo root:
   - `changelogs/changelog.md` — live file; starts empty (header + rotation instructions only).
   - `changelogs/archives/changelog_<YYYYMMDD>.md` — frozen archives. `<YYYYMMDD>` is the most recent date inside that file. The initial migration packs all existing extracted entries into archives, ≤ 250 lines each.

2. **Pre-commit hook** `scripts/check_changelogs_size.py`, registered as `check-changelogs-size` in `.pre-commit-config.yaml`, fails the commit when any file under `changelogs/**/*.md` exceeds **300 lines**. Workflow when the hook fires: rotate `changelogs/changelog.md` into `changelogs/archives/changelog_<latest-date-inside>.md` and start a fresh `changelog.md`.

3. **Archive file format** (one `## YYYY-MM-DD` heading per date, dates newest-first inside the file):
   ```markdown
   # Changelog archive

   Covers <newest-date> (newest) down to <oldest-date> (oldest).

   ## YYYY-MM-DD

   ### <Description-or-File> — <author>

   <body — sub-bullets if any, or single-line description>

   ## YYYY-MM-DD

   ### `<Path/Relative/To/src/spectra.py>` — <author>

   - <bullet 1 verbatim>
   - <bullet 2 verbatim>
   ```

4. **Live `changelog.md` format**: just an H1 heading and a short comment explaining the rotation workflow. No date sections at initial migration time.

5. **In-docstring extractions** prepend `func <function_name>: ` to the description text so the function context — lost when the docstring is removed — is preserved.

6. **Author renames** applied uniformly: `u.k.` → `YW.Huang`; `k.i.` / `k.ichimoto` → `k.ichimoto`; `j.n.` → `j.natsume`.

7. **Pre-port IDL entries dropped**: in-docstring lines matching `^- <author>(\s+\d+\s+\w+\.\d{4})+` (one or more `D mon.YYYY` dates after an author, no leading `YYYY.M.D`) are skipped.

8. **Phase 1 must not modify source files.** Phase 2 deletes the in-source history blocks; Phase 2 acceptance grep verifies no `# VERSION` line remains and no `Modification history:` block remains in `src/spectra/**/*.py`.

### Non-Functional Requirements

- **Faithfulness over compactness.** Sub-bullets preserved verbatim. No paraphrasing, no fixing typos (the source typos like `udring` in `SELib.py:16` carry into the changelog unchanged).
- **Deterministic ordering**: inside one changelog file, group by file path (alphabetical), then by author within the file. Cross-file merges sort by path.
- **No new dependencies.** Extraction is a one-off script run locally, not committed as project tooling.

## Scope

### In Scope

- [ ] New directory `changelogs/` at repo root with `changelog.md` and `archives/`.
- [ ] One or more `archives/changelog_<YYYYMMDD>.md` files packing the extracted history; each ≤ 250 lines.
- [ ] Extraction of file-level `# VERSION` blocks from all 52 source files matching `grep -rln "^# VERSION" src/spectra --include="*.py"`.
- [ ] Extraction of in-docstring `Modification history:` blocks from `src/spectra/Atomic/ContinuumOpacity.py` (×8) and `src/spectra/Atomic/LTELib.py` (×1).
- [ ] `scripts/check_changelogs_size.py` + `.pre-commit-config.yaml` registration.
- [ ] This task spec (`docs/tasks/008-version-header-changelogs/`).

### Out of Scope (Boundaries)

> Items explicitly excluded from this task. A separate follow-up task removes the in-source history blocks once the changelogs are confirmed.

- **Source-code edits.** The whole point of phase 1 is to land the changelogs without touching `src/`. Removal of `# VERSION` blocks and docstring `Modification history:` blocks is task #009 (or a phase-2 commit on this same branch), gated on user confirmation.
- **`# WARNING` blocks** in source headers — they are not history.
- **Docstring text other than the `Modification history:` block** — `Parameters`, `Returns`, `References`, prose, etc. are untouched.
- **Notebook history** in `notebooks/**/*.ipynb` — out of scope.
- **`pyproject.toml` version field**, README, and other top-level metadata files.
- **Git history rewriting** — we are not annotating commits; we are creating documentation files alongside the git history.

## Acceptance Criteria

- [ ] `changelogs/changelog.md` exists and is a short stub (header + rotation instructions, no date sections at initial migration).
- [ ] `changelogs/archives/` contains one or more `changelog_<YYYYMMDD>.md` files; each is ≤ 250 lines and named by the most recent date it contains.
- [ ] Every unique normalized date with at least one in-scope entry appears in one of the archive files.
- [ ] `scripts/check_changelogs_size.py` exists and is registered in `.pre-commit-config.yaml`; running `uv run python scripts/check_changelogs_size.py` exits 0 on the current tree.
- [ ] No file under `src/spectra/` is modified by Phase 1: `git diff src/` is empty after the Phase 1 commit.
- [ ] `2021/05/18 u.k. spectra-re` (the bulk initial-migration entry) appears as a single merged entry inside one archive file, listing all affected files, not as one entry per file.
- [ ] `u.k.` does not appear anywhere under `changelogs/`; `k.i.` does not appear as a stand-alone author tag.
- [ ] In-docstring entries under `changelogs/` are prefixed with `func <function_name>: ` so the function context is preserved.
- [ ] No pre-port IDL entry (matching the IDL-skip regex: `- <author> <D mon.YYYY>(, <D mon.YYYY>)*`) appears in `changelogs/`. Single-date `YYYY.M.D` entries that predate the project (e.g. `2006.5.23`, `2015.7.5` from `LTELib.py:Ufunc_`) are real edits and ARE kept.
- [ ] User has reviewed and confirmed the changelogs before any source removal happens.

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| Design decisions (file-per-date, drop versions, author renames, etc.) | kouui | ✅ Done | Captured in this conversation, 2026-05-17 |
| User confirmation that changelogs are complete | kouui | ⏳ Pending | Gate for follow-up source-removal task |

## Risks & Open Questions

- [ ] **Parser ambiguity.** The header block is delimited by `# ---...---` lines but the `# VERSION` block may be followed by other sections (e.g. `# WARNING`). Extraction must stop at the next `# WORD` heading or at the closing `# ---...---`, whichever comes first.
- [ ] **Date format edge cases.** `LTELib.py` and `ContinuumOpacity.py` use mixed `YYYY.M.D` / `D mon.YYYY` formats. Pre-port IDL entries (the `D mon.YYYY` style without a leading `YYYY.M.D`) are dropped by rule; single-`YYYY.M.D` entries are normalized. Anything that does not match either pattern is flagged for manual review rather than silently dropped.
- [ ] **Merging granularity.** When two files have a `(date, author)` entry where one file has sub-bullets and another does not (e.g. `2021/05/18 u.k. spectra-re` vs `2021/05/18 u.k.` with later bullets), the merge rule is: keep them as separate per-file headings under the same date file. See plan §1.
- [ ] **`# WARNING` sibling block.** When `# WARNING` appears immediately after `# VERSION`, the extractor must not capture it. Verified by grep that no `# WARNING` text leaks into any changelog file.
- [ ] **Author rename collision.** If a non-u.k./non-k.i. author appears (e.g. an external contributor), it stays verbatim. Spot-check during extraction.

## References

- GitHub issue: [#8 want to remove `history` in *.py files](../../../../issues/8)
- Closest precedent (audit-and-extract pattern): `docs/tasks/007-se-params-container/`
