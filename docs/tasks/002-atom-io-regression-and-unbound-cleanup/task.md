# Task: AtomIO load regression coverage + pyright `reportPossiblyUnbound*` cleanup

> **Status:** Done
> **Owner:** kouui
> **Created:** 2026-04-19
> **Last Updated:** 2026-04-19

## Objective

分两阶段：
1. **Stage 1**：为 `AtomIO.py` 的 load 路径补 e2e 回归覆盖 —— load 同一 `*.conf` 得到**完全一致**的 `Atom` / `Wavelength_Mesh` / `path_dict`（包含所有数组内容），形成行为回归锁。
2. **Stage 2**：以 Stage 1 的安全网为前提，移除 `pyproject.toml` 中 `reportPossiblyUnboundVariable = false` / `reportOperatorIssue = false` 两处全局抑制，逐文件从根源修复 122 处错误（优先 `AtomIO.py` 62 处）。

## Background & Context

- 上一轮 PR #9 清掉了 27 处 `# type: ignore`，但 `pyproject.toml` 仍保留两处全局抑制（`reportPossiblyUnboundVariable`、`reportOperatorIssue`）。
- 同时取消抑制后 pyright 报 **122 错**，按文件分布：`AtomIO.py` 62 / `ExFAL.py` 23 / `Grotrian.py` 21 / `01makeAtom.py` 9 / `HelpUtil.py` 4 / `Plotting.py` 2 / `LTELib.py` 1。
- **核心矛盾**：`AtomIO.py` 错误最多且**没有直接单元测试**；现有 e2e (`test_reg_e2e_SE.py`) 只断言 SE 结果，不断言 `Atom` struct 本身 → 重构 parse 分支（`prefix`、`_ctj_`、`contIndex` 等 possibly-unbound 变量）时,安全网不足。

- Spec: `tmp/handoff.md`（本地工作区）
- Related Issues: 无（PR #9 的 follow-up）

## Requirements

### Functional Requirements

**Stage 1**
1. 新建 `tests/regression/test_reg_e2e_AtomLoad.py`，参数化遍历 8 个 config：
   - H 系列: `H.conf`、`H6.conf`、`H_theory.conf`
   - He 系列: `He.conf`、`He_I.conf`、`He_I_II.conf`
   - Ca 系列: `Ca_II.conf`、`Ca_I-II-III.conf`
2. 每个 config 执行 `Atom.init_Atom_()` 并与参考值**按字段全量比对**：所有标量（`Z`、`Mass`、`nLevel`、enum 等）、所有数组（`Level.erg`、`Line.AJI`、`CE.Te_table`、`PI.alpha_table`、`waveMesh.Cont_mesh` 等）、`path_dict`（**相对化到 `_ROOT_DIR`** 后比对）。
3. 新建参考文件 `tests/regression/atom_reference_values.json`（独立于现有 `reference_values.json`，避免后者膨胀）。
4. 新建生成脚本 `scripts/gen_atom_reference.py`：手动触发，从当前 `main` 分支代码 load 8 atoms 后写 JSON。脚本与 test 共享同一个序列化/比对 helper。

**Stage 2**
5. `pyproject.toml` 移除 `reportPossiblyUnboundVariable = false` 和 `reportOperatorIssue = false`。
6. `uv run --extra dev pyright` 输出 `0 errors`。
7. 所有 122 处 possibly-unbound / operator 错误从根源修复：
   - 默认手法：循环前初始化、`else: raise ValueError(...)` 补分支、拆 guard 函数
   - 禁用：任意位置加 `assert X is not None` 作为 silencer、per-line `# type: ignore`
   - 例外：`Experimental/ExFAL.py` 与 `RH2Spectra/01makeAtom.py` 可用 **file-level** pyright 指令（如 `# pyright: reportPossiblyUnboundVariable=false`），不是全局抑制。决策需在本文件 Risks 记录。
8. 回归：Stage 1 新 tests + 现有 247 tests 全绿。

### Non-Functional Requirements

- **不改变运行时行为**：Stage 2 只做类型面的根源修，不动业务逻辑；Stage 1 新 tests 跑完 atom_reference_values.json 与代码当前行为 bit-exact 一致。
- **可复现**：生成脚本产出的 JSON 在不同机器上一致（路径相对化、float 用 `tolist()` round-trip）。
- **维护性**：比对 helper 与生成 helper 是同一段代码（避免双份维护）。

## Scope

### In Scope

**Stage 1 — AtomIO 回归覆盖**
- [ ] `scripts/gen_atom_reference.py` 新建（含序列化 helper）
- [ ] `tests/regression/atom_reference_values.json` 生成
- [ ] `tests/regression/test_reg_e2e_AtomLoad.py` 新建（参数化 8 configs）
- [ ] `tests/regression/conftest.py` 加 `ref_atom` session fixture
- [ ] 序列化/比对 helper 模块（可放 `tests/regression/_atom_serde.py`，test 与脚本共享 import）

**Stage 2 — pyright 根源修**
- [ ] `pyproject.toml` 移除 `reportPossiblyUnboundVariable`、`reportOperatorIssue` 两处抑制
- [ ] `src/spectra/Util/AtomUtils/AtomIO.py` 62 处
- [ ] `src/spectra/Visual/Grotrian.py` 21 处
- [ ] `src/spectra/Util/HelpUtil.py` 4 处
- [ ] `src/spectra/Visual/Plotting.py` 2 处
- [ ] `src/spectra/Atomic/LTELib.py` 1 处
- [ ] `src/spectra/Experimental/ExFAL.py` 23 处（file-level pragma 允许）
- [ ] `src/spectra/Util/AtomicDataUtils/RH2Spectra/01makeAtom.py` 9 处（file-level pragma 允许）

### Out of Scope (Boundaries)

- **数值正确性证明**：本次 Stage 1 是"行为回归锁"，锁定的是**当前** AtomIO 的输出。若当前代码在某分支已有 bug，Stage 1 会把 bug 也一起锁。Stage 2 如果发现 pyright 提示真 bug，就地 fix + 更新参考值 + 单独说明。
- **过时的 conf 文件**：`C_III`、`O_V`、`Si_III` 等用户确认已过时，不在本次覆盖。
- **新功能 / 重构业务逻辑**：只修类型 + 加测试，不动计算实现。
- **其他文件的既有 `# type: ignore`**：本次 scope 外的历史 ignore（PR #9 之前）不动。

## Acceptance Criteria

**Stage 1**
- [x] `scripts/gen_atom_reference.py` 可独立运行，产出 `atom_reference_values.json`
- [x] `tests/regression/test_reg_e2e_AtomLoad.py` 至少 8 个 test case（每 config 一个）全绿
- [x] `uv run --extra dev python -m pytest tests/regression/ -q` 全部 passed（新 8 + 原 247）
- [x] `atom_reference_values.json` 存 relative path（不含任何机器绝对路径）
- [x] **Bonus**：reference 从 `a84128b` 的 worktree 生成，diff 显示自 `a84128b` 起 11 个 AtomIO commit 行为 bit-exact preserving

**Stage 2**
- [x] `pyproject.toml` 不再有 `reportPossiblyUnboundVariable = false` 和 `reportOperatorIssue = false`
- [x] `uv run --extra dev pyright` = `0 errors, 0 warnings`
- [x] 122 处 possibly-unbound / operator 全部从根源修（2 个例外：`ExFAL.py` 23 处 + `01makeAtom.py` 9 处用 file-level pragma，决策见 Q4）
- [x] `uv run --extra dev python -m pytest tests/regression/ -q` 仍全绿（255 passed）
- [x] `uv run --extra dev pre-commit run --all-files` 通过

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| PR #9 | 本仓库 | merged (`250cc36`) | 已在 `main` |
| `init_Atom_` 返回结构 | `Struct/Atom.py:110-259` | stable | 返回 `(Atom, Wavelength_Mesh, path_dict)` |
| `reference_values.json` 惯例 | 现有 | stable | 扁平 `{key: array|scalar}`，新文件沿用同 pattern |

## Risks & Open Questions

- [ ] **R1**：struct array 的 dtype 含混合字段（int / float / bool），JSON 序列化时每个字段单独 `tolist()`；反序列化时需要配合比对逻辑，不用重建原 dtype（直接比 list）。
- [ ] **R2**：`float` round-trip。`np.float64 → list (Python float) → json → list → np.asarray(dtype=float64)` 是 bit-exact 的（IEEE-754 17 位有效数字），但依赖 `json.dumps` 默认使用 `repr()` 而非 `str()`。Python 3 `json.dumps` 已用 `repr`，无需特殊处理。验证时 assert 用 `rtol=1e-12, atol=0`。
- [ ] **R3**：`path_dict` 含 `None` 值（没有 CEe/PI 的 config）。JSON 原生支持 `null`，无额外处理。
- [ ] **R4**：`_ctj_table` 包含嵌套 tuple of tuple of str —— JSON 会变 nested list；比对时要把 list 转回 tuple 才能 `==` 或改比对逻辑直接比 list（推荐后者）。
- [ ] **R5 (Stage 2)**：`AtomIO.make_Atom_PI_` 在 PR #9 Phase 2 刚动过（int/float cast + Hydrogenic overload），现再动 possibly-unbound 可能改动重叠。缓解：修之前先 `git log -p src/spectra/Util/AtomUtils/AtomIO.py` 看上轮 diff，避免 undo。
- [ ] **R6 (Stage 2)**：循环变量 possibly-unbound 易误伤（如 `for i, k in enumerate(items): ... items 之后用 i`）。手法：循环前 `i = -1`（或合适默认），或循环前 `assert items, "message"`（**只**在数据结构本身保证非空的情况下用）。
- [ ] **Q1 (已确认)**：`path_dict` 存相对还是绝对路径？→ **相对化到 `_ROOT_DIR`**（决策已定）。
- [ ] **Q2 (已确认)**：Ca_I-II-III.conf 与 Ca_I_II_III.conf 的关系？→ diff 后只差尾部空白，**只测 Ca_I-II-III.conf**。
- [ ] **Q3 (已确认)**：序列化粒度？→ **D1c + D1b 混合**：所有标量 + 所有数组**全量存**（非仅 fingerprint），保证敏感度。
- [ ] **Q4 (待 Stage 2 开工前决策)**：`Experimental/ExFAL.py` 和 `RH2Spectra/01makeAtom.py` 是全量根源修，还是用 file-level pragma？默认倾向 pragma（低回报 / 非核心路径）。

## References

- `tmp/handoff.md` — 本 follow-up 的 handoff 记录
- `docs/tasks/pyright-cleanup-followup/task.md` — 上一轮任务记录（PR #9 的 scope）
- PR #9 (`250cc36`) — 27 处 ignore 根源消除
- `.claude/memories/style-guides/general_en.md` — 项目 coding 原则（SRP、DRY、根源修）
