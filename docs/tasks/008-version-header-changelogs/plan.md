# Plan: Extract `# VERSION` headers and `Modification history:` docstring blocks into `changelogs/`

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-05-17
> **Target Completion:** 2026-05-17

---

## 0. Context

> **Objective:** Move all change-history metadata from `src/spectra/**/*.py` into a rotating-changelog layout under `./changelogs/` (a live `changelog.md` plus archive files of ≤ 250 lines each, guarded by a pre-commit hook that blocks any file > 300 lines). Phase 1 lands the changelogs and the hook without touching source. Phase 2 — gated on user confirmation — removes the in-source blocks.
> **Full spec:** [task.md](./task.md)

---

## 1. Overall Architecture

### Layout

```
changelogs/
├── changelog.md                          # live, starts empty (header + rotation note)
└── archives/
    ├── changelog_20210607.md             # ≤ 250 lines; latest date inside = 2021-06-07
    └── changelog_20260515.md             # ≤ 250 lines; latest date inside = 2026-05-15
```

`changelog.md` is the active file new entries are appended to. When the pre-commit hook
(`scripts/check_changelogs_size.py`, hard cap 300 lines) would block a commit, the user
rotates: move `changelog.md` to `archives/changelog_<latest_date_inside>.md` and create
a fresh empty `changelog.md`.

Inside any archive file, dates are ordered **newest-first**, each as an H2 heading:

```markdown
# Changelog archive

Covers 2026-05-15 (newest) down to 2021-06-07 (oldest).

## 2026-05-15

### `Function/SEquil/SELib.py` — YW.Huang

se_params: SE_Params_Container threaded through cal_SE_* wrappers

### `Struct/Atmosphere.py` — YW.Huang

- removed Tr / use_Tr / doppler_shift_continuum (moved to SE_Params_Container)

## 2026-05-09

### ...
```

### Merging rule

The `2021/05/18 u.k. spectra-re` line appears in nearly every file with no sub-bullets.
That entry collapses into **one** combined H3 heading inside the archive that holds 2021-05-18:

```markdown
## 2021-05-18

### spectra-re — YW.Huang

Files:
- `Atomic/BasicP.py`
- `Atomic/Collision.py`
- ...
```

Entries with sub-bullets stay as their own per-file H3 heading even if they share a
`(date, author)` with another file. Formally: **merge iff `(normalized_date,
normalized_author, header_description)` matches AND none of the entries has sub-bullets**.
Otherwise keep separate.

### Key components

| Component | Path | New / Modified |
|-----------|------|----------------|
| Task spec | `docs/tasks/008-version-header-changelogs/{task,plan}.md` | New |
| Extraction script | `tmp/extract_changelogs.py` (one-off, not committed) | New (ephemeral) |
| Removal script | `tmp/remove_history_blocks.py` (one-off, not committed) | New (ephemeral) |
| Live changelog | `changelogs/changelog.md` | New |
| Archive files | `changelogs/archives/changelog_<YYYYMMDD>.md` (≤ 250 lines each) | New |
| Size guard | `scripts/check_changelogs_size.py` | New |
| Pre-commit registration | `.pre-commit-config.yaml` | Modified (one hook added) |
| Source files | `src/spectra/**/*.py` | Untouched in Phase 1; deletions only in Phase 2 |

### Data flow

1. Script walks `src/spectra/**/*.py`.
2. For each file: detect a `# VERSION` block bounded by `# -----...` (open) and the next `# WORD` heading or `# -----...` line (close). Parse entries.
3. For `Atomic/ContinuumOpacity.py` and `Atomic/LTELib.py`: also scan docstrings for `Modification history:` blocks. For each block, capture the function name from the surrounding `def …(…):` line.
4. Normalize dates, normalize authors, drop pre-port IDL entries, apply the merge rule.
5. Sort all dates newest-first; pack date blocks into archive files of ≤ 250 lines each; each archive's filename is `changelog_<latest-date-inside>.md`.
6. Write `changelog.md` as an empty stub.
7. Phase 2 (gated): preserving each source file's existing line-ending style (LF or CRLF), strip the in-source `# VERSION` and `Modification history:` blocks.

---

## 2. Implementation Phases

### Phase 1: Extract & write changelogs (this task delivers when confirmed)

> **Goal:** Land `changelogs/` populated with per-date markdown files. No source changes.
> **Estimated Effort:** 0.5 day

- [ ] Step 1.1 — Commit this task spec on a feature branch (`docs/008-changelog-extraction`).
- [ ] Step 1.2 — Write a throwaway extraction script under `tmp/extract_changelogs.py` (not committed). The script:
  - walks `src/spectra/**/*.py`
  - parses the `# VERSION` block via a simple state machine (in_header → on_version → on_version_number → on_entry → on_subbullet)
  - parses in-docstring `Modification history:` blocks for the two known files; captures the enclosing function name
  - normalizes dates and authors (table below)
  - applies the skip rule for pre-port IDL entries
  - applies the merge rule (above)
  - sorts dates newest-first; packs them into archive files of ≤ 250 lines each; each archive's filename is `changelog_<latest-date-inside>.md`
  - writes `changelogs/changelog.md` as an empty stub (header + rotation instructions)
- [ ] Step 1.3 — Date normalization table:
  | Input | Normalized |
  |-------|-----------|
  | `2021/05/18` | `20210518` |
  | `2019.9.15` | `20190915` |
  | `15 jun.1987` | `19870615` (dropped by skip rule, but normalization handles it for any survivors) |
  | `6 Jan.1992` | `19920106` (dropped by skip rule) |
  | `2-21/06/15` (typo) | `20210615` via explicit fixup table |
- [ ] Step 1.4 — Author normalization table:
  | Input | Normalized |
  |-------|-----------|
  | `u.k.` | `YW.Huang` |
  | `u.k`  | `YW.Huang` (trailing-period variant seen in `SELib.py:36`) |
  | `k.i.` | `k.ichimoto` |
  | `k.ichimoto` / `K.Ichimoto` | `k.ichimoto` |
  | `j.n.` | `j.natsume` |
  | compound (`k.i., u.k.`) | `k.ichimoto, YW.Huang` (split on comma, normalize each) |
  | anything else | verbatim, flagged in the script's stderr summary for manual review |
- [ ] Step 1.5 — Skip rule for pre-port IDL entries:
  - Pattern: line in an in-docstring history block matching `^- (\S+)((?:\s+\d{1,2}\s+\w+\.\d{4},?)+)\s*$` — author followed by one or more `D mon.YYYY` dates and nothing else.
  - Action: silently skip. Log to stderr summary for audit.
- [ ] Step 1.6 — Add `scripts/check_changelogs_size.py` (a self-contained Python script using only stdlib) that walks `changelogs/**/*.md` and exits non-zero with a rotation hint if any file exceeds 300 lines. Register it in `.pre-commit-config.yaml` as `check-changelogs-size`, `always_run: true`, `pass_filenames: false`.
- [ ] Step 1.7 — Run the script. Review stderr summary: number of files processed, number of entries extracted, number of merges, number of skipped IDL entries, list of unrecognized author tags (should be empty).
- [ ] Step 1.8 — Spot-check the produced layout manually:
  - `changelogs/changelog.md` — empty stub.
  - At least one archive file under `changelogs/archives/` with the `spectra-re` merge inside.
  - Verify the `2026-05-15` `SE_Params_Container` entry appears under its date in the newest archive.
- [ ] Step 1.9 — Commit the new files together (extraction outputs, the size-guard script, and the pre-commit config update). The task spec stays in a separate commit for a clean "plan-only" diff.
- [ ] Step 1.10 — **Stop.** Present the user with the layout and spot-checks. Wait for explicit "approved" before any source removal.

**Phase 1 Exit Criteria:**
- [ ] `changelogs/changelog.md` + `changelogs/archives/changelog_<YYYYMMDD>.md` populated; every archive ≤ 250 lines.
- [ ] `uv run python scripts/check_changelogs_size.py` exits 0.
- [ ] `git diff src/` is empty.
- [ ] User has reviewed and confirmed.

---

### Phase 2 (follow-up, gated): remove the in-source blocks

> **Goal:** Delete `# VERSION` header blocks and in-docstring `Modification history:` blocks from `src/spectra/**/*.py`. Pre-commit + regression must stay green.
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 1 user confirmation

- [ ] Step 2.1 — For each file with a `# VERSION` block: delete from the `# VERSION` line through the next `# WORD` heading or the closing `# ---...---` (whichever comes first). Preserve the file's top description block and any `# WARNING` block. **Preserve the file's original line-ending style (LF or CRLF)** — read as bytes, detect, restore on write. One file in the tree (`Util/AtomicDataUtils/MakeTheoreticalHydrogenLevels.py`) is CRLF; the deletion must not normalize it.
- [ ] Step 2.2 — For `ContinuumOpacity.py` and `LTELib.py`: delete the `Modification history:` heading and its bulleted entries from each docstring, leaving the surrounding `Parameters` / `Returns` / `References` / `Original doc-string` (upstream IDL prose) sections intact. Block end-detection: stop at the first non-blank, non-bullet, non-continuation line.
- [ ] Step 2.3 — Run `pre-commit run --all-files` (ruff format / ruff check / pyright / pytest-regression / check-changelogs-size). All green expected — these are comment-only and docstring-only edits, no behavior change.
- [ ] Step 2.4 — Run `uv run pytest tests/` — green expected.
- [ ] Step 2.5 — Commit. Push branch. Open PR (only if explicitly requested by the user).

**Phase 2 Exit Criteria:**
- [ ] `grep -rn "^# VERSION" src/spectra` returns zero matches.
- [ ] `grep -rni "modification\s*history" src/spectra` returns zero matches.
- [ ] `git diff --ignore-cr-at-eol` on each modified source file shows only deletion-lines from history blocks (no formatting churn).
- [ ] `pre-commit run --all-files` clean.
- [ ] `pytest tests/` green.

---

## 3. Boundaries — Do NOT Touch

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| Source code (phase 1) | `src/spectra/**/*.py` | Phase 1 is extraction-only. Source changes are phase 2, gated on user confirmation. |
| `# WARNING` blocks | e.g. `Atomic/ContinuumOpacity.py:16-18` | Not history. Distinct category of file-header comment. |
| Docstring text other than `Modification history:` | All docstrings | `Parameters`, `Returns`, `References`, prose body — out of scope. |
| Notebooks | `notebooks/**/*.ipynb` | Out of scope; the issue scopes the request to `src/spectra/*.py`. |
| `pyproject.toml` version field | `pyproject.toml` | Not a history block; project versioning continues via git tags + pyproject. |
| Existing task docs | `docs/tasks/00[1-7]-*/` | Historic audit trail. Not edited. |
| Reference values | `tests/regression/reference_values.json` | Phase 2 is comment-only; any numeric diff would be a bug. |

---

## 4. Test Coverage

### Phase 1

| Level | Scope | Tool |
|-------|-------|------|
| Manual | Spot-check the layout (`changelog.md` stub + 2 archives, spectra-re merge inside one of them, recent date in newest archive) | Eyeball |
| Grep gate | `git diff src/` empty after commit | shell |
| Grep gate | `grep -r "u\.k\." changelogs/` empty | shell |
| Grep gate | `grep -r "^\- k\.i\. " changelogs/` empty | shell |
| Size guard | `uv run python scripts/check_changelogs_size.py` exits 0 | script |
| Pre-commit | `pre-commit run check-changelogs-size --all-files` passes | hook |

### Phase 2

| Level | Scope | Tool |
|-------|-------|------|
| Regression | All e2e regression tests | `pytest tests/regression/` |
| Unit | All unit tests | `pytest tests/unittest/` |
| Static | Pyright + ruff | `pre-commit run --all-files` |
| Grep gate | No `# VERSION` heading remains | `grep -rn "^# VERSION" src/spectra` |
| Grep gate | No `Modification history:` block remains | `grep -rni "modification\s*history" src/spectra` |

No new test code needed. Phase 2 is comment / docstring edits; behavior is invariant.

---

## 5. Key Decisions

### Decision 1: One file per unique date — `changelogs/YYYYMMDD.md`

- **Context:** Issue says `changelogs/{YYYYmmdd}.md`. Options for grouping: one file per date, one per year, or one mega-file.
- **Options considered:**
  1. One file per unique normalized date (probably ~30–50 files).
  2. One file per year (~6 files).
  3. Single file ordered chronologically.
- **Decision:** Option 1.
- **Rationale:** User-stated preference. Mirrors the issue's literal `{YYYYmmdd}` template. Grep by date is precise. Small files load fast.
- **Consequences:** A directory with several dozen files. Acceptable; this is documentation, not loaded by code.

### Decision 2: Drop version numbers (`0.1.0`, `0.1.1`, …) entirely

- **Context:** The issue explicitly permits ignoring version info. The `# VERSION` blocks group entries under version headers that are often out of chronological order.
- **Options considered:**
  1. Drop version numbers; key by date only.
  2. Preserve version numbers as section subheaders inside each date file.
- **Decision:** Option 1.
- **Rationale:** Git already records authoritative versioning. The in-file version numbers were a manual, error-prone reflection of git; keeping them in the changelog perpetuates the rot.
- **Consequences:** Some loss of "which release this shipped in" context. Recoverable from git tags if ever needed.

### Decision 3: Merge `(date, author)` duplicates only when both sides are single-line with no sub-bullets

- **Context:** The `2021/05/18 u.k. spectra-re` line appears in nearly every file. Merging is the obvious win. But `(date, author)` matches can also collide where one side has sub-bullets — merging those would conflate distinct work.
- **Options considered:**
  1. Always merge on `(date, author)`, concatenating sub-bullets.
  2. Never merge; one entry per file.
  3. Merge only when both sides are bullet-free single lines; otherwise keep separate.
- **Decision:** Option 3.
- **Rationale:** Captures the "initial migration" case (the noisy duplicate) without conflating different work that happened on the same day. User-stated preference framed it as "merge initialization, keep distinct work distinct".
- **Consequences:** Slightly more code in the extraction script (merge predicate); minor.

### Decision 4: Sub-bullets verbatim — fidelity over compactness

- **Context:** Discussion offered two formatting options for sub-bullets (preserve as separate list items vs fold into one description line).
- **Options considered:**
  1. Preserve each sub-bullet verbatim under a per-file/per-author heading.
  2. Fold into one densified line per entry.
- **Decision:** Option 1.
- **Rationale:** User preference. The history blocks already carry semantic structure as bullets; collapsing loses information. Token budget is no longer a concern once the blocks live in `changelogs/` and not in code.
- **Consequences:** Slightly larger changelog files. Trivial.

### Decision 5: In-docstring entries get a `func <name>:` prefix; merged with file-level entries

- **Context:** In-docstring `Modification history:` blocks live inside specific function docstrings. Once extracted, the function context is lost unless restored.
- **Options considered:**
  1. Keep in-docstring entries separate from file-level entries in the changelog.
  2. Merge into the same date file; prefix description with `func <name>:`.
- **Decision:** Option 2.
- **Rationale:** User-stated preference. Single chronological per-date file is easier to scan than parallel structures.
- **Consequences:** All changelog entries share one heading style; the `func ...:` prefix conveys function context.

### Decision 6: Author rename — `u.k.` → `YW.Huang`; `k.i.` and `k.ichimoto` → `k.ichimoto`; `j.n.` → `j.natsume`

- **Context:** Author tags `u.k.`, `k.i.`, `j.n.` are abbreviations. Spelling them out improves grep-ability of changelogs and matches the user's preferred form.
- **Options considered:**
  1. Keep verbatim.
  2. Normalize to a canonical form per author.
- **Decision:** Option 2.
- **Rationale:** User-stated preference. `u.k.` → `YW.Huang`; both `k.i.` and `k.ichimoto` collapse to `k.ichimoto`; `j.n.` → `j.natsume` (added during Phase 1 after the script flagged the unknown tag; user supplied the mapping). The extractor still emits a WARN for any future unknown author so they can be added explicitly rather than silently passed through.
- **Consequences:** Loses the original abbreviation, which is fine — git history still has it.

### Decision 7: Skip pre-port IDL entries in in-docstring blocks

- **Context:** `ContinuumOpacity.py` and `LTELib.py` docstrings list dates like `15 jun.1987`, `6 Jan.1992`, `19 Feb.1994` from the original IDL source. Those dates predate this project by decades.
- **Options considered:**
  1. Include them as historical entries.
  2. Skip them entirely (upstream provenance, not project history).
  3. Move them to a separate `changelogs/upstream-idl.md` file.
- **Decision:** Option 2.
- **Rationale:** User clarification: "this is different... these dates is just referencing the date of original code, not when we implement the code in this project". The changelog is for this project. Upstream provenance lives elsewhere if needed.
- **Consequences:** A few lines lost. Detection rule: leading author + one or more `D mon.YYYY` dates with no `YYYY.M.D`. The actual port entry — e.g. `2019.9.15 k.ichimoto from IDL ahic.pro` — is kept because it carries a `YYYY.M.D` date describing when the port happened in this project's lifetime.

### Decision 8: Two-phase, gated execution

- **Context:** Issue requirement 3: "create changelogs first, do not remove `history` comment until confirming the changelogs is correct and completed".
- **Options considered:**
  1. One big commit with extraction + source removal.
  2. Two phases: extract → user review → remove.
- **Decision:** Option 2.
- **Rationale:** Directly required by the issue. Lets the user audit the extraction before destructive edits.
- **Consequences:** Two commits (or two PRs). Trivial cost.

### Decision 9: Two-file layout (`changelog.md` + `archives/`) instead of per-date files

- **Context:** The initial design produced one file per unique date (~39 files for the existing history). User observed that pattern produces too many files; preferred a single live `changelog.md` with overflow archived in rotation snapshots.
- **Options considered:**
  1. One file per date.
  2. One file per year.
  3. Single monolithic `changelog.md`.
  4. Live `changelog.md` + rotation into `archives/changelog_<YYYYMMDD>.md`, with a pre-commit hook enforcing a line-count cap.
- **Decision:** Option 4.
- **Rationale:** User preference; mirrors common rotation conventions (logrotate-style). The pre-commit cap (300 lines hard, 250 soft target for archives) ensures no file becomes unwieldy. Per-archive filename = newest date inside → file is grep-friendly by date and self-describing.
- **Consequences:** Initial migration produces 2 archive files plus an empty `changelog.md`. New entries land in `changelog.md`; the rotation discipline is manual, gated by the pre-commit hook.

### Decision 10: Phase 2 must preserve original line endings

- **Context:** The first pass of the Phase 2 removal script used `Path.read_text` / `write_text`, which silently normalized CRLF → LF on the single CRLF file in the tree (`Util/AtomicDataUtils/MakeTheoreticalHydrogenLevels.py`). Codex review flagged this as out-of-scope formatting churn.
- **Options considered:**
  1. Normalize all files to LF (add a `.gitattributes` rule, treat as cleanup).
  2. Preserve each file's existing line-ending style.
- **Decision:** Option 2.
- **Rationale:** This task scope is history-block removal only. Line-ending normalization is a separate concern and would muddy the diff for reviewers.
- **Consequences:** The Phase 2 script reads/writes bytes and re-encodes CRLF when the source used it. `git diff --ignore-cr-at-eol` on each modified file shows only history-block deletions.

---

## 6. Precautions

### Technical risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Parser misidentifies the close of a `# VERSION` block (e.g. captures `# WARNING` content) | Med | Low | Close on next `# WORD` heading OR closing `# ---...---`, whichever first. Validate with `grep -r "WARNING" changelogs/` post-extraction. |
| Date format outside the known three patterns | Low | Low | Extraction script logs unknown date formats to stderr and exits non-zero. Triage manually. |
| Unknown author tag | Low | Low | Script logs unknown authors to stderr summary; entries are kept verbatim, not silently renamed. |
| Phase 2 accidentally drops a `# WARNING` or non-history comment | Med | Med | Phase 2 deletion bounds: from `# VERSION` line through next `# WORD` heading OR closing `# ---...---`. Pre-commit + regression catch behavior change; manual diff review catches comment over-deletion. |
| In-docstring extraction strips too much (e.g. trailing `References` block) | Med | Med | The block is bounded by the next blank-line-then-non-bullet, or the next docstring heading (`References`, `Parameters`, `Returns`, `Notes`). Test on the 9 known sites before generalizing. |

### Rollback plan

- **Phase 1:** changelogs are pure additions — `git revert` undoes the commit, leaves source untouched.
- **Phase 2:** revert the phase-2 commit; source comments restored. Changelogs remain (no longer authoritative until re-confirmed).

### Migration notes

- No data migration.
- No external consumer impact — comment-only changes in phase 2.

### Performance considerations

- N/A. Extraction is offline, one-shot.

### Security considerations

- N/A. Documentation-only changes; no secrets, no auth surface.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-17 | kouui | Initial draft after discussion on issue #8 |
| 2026-05-17 | kouui | Reworked layout to `changelog.md` + `archives/` (Decision 9) and added Phase 2 line-ending preservation (Decision 10). Updated requirements, acceptance criteria, exit criteria, and architecture diagram. |
