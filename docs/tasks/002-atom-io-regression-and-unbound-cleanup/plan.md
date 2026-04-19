# Plan: AtomIO load regression coverage + pyright `reportPossiblyUnbound*` cleanup

> **Task:** `docs/tasks/atom-io-regression-and-unbound-cleanup/task.md`
> **Owner:** kouui
> **Created:** 2026-04-19
> **Target Completion:** 2026-04-22 (2-3 工作日)

---

## 0. Context

> **Objective:** 先为 `AtomIO` 的 load 路径补 e2e 全字段回归覆盖（Stage 1），然后用该安全网移除 `pyproject.toml` 剩余的 `reportPossiblyUnboundVariable` / `reportOperatorIssue` 两处抑制并从根源修 122 处错误（Stage 2）。
> **Full spec:** `docs/tasks/atom-io-regression-and-unbound-cleanup/task.md`

---

## 1. Overall Architecture

### System Overview

```
 scripts/gen_atom_reference.py (one-shot)
           │
           │  load & serialize 8 atoms
           ▼
 tests/regression/atom_reference_values.json
           │
           │  read by ref_atom fixture
           ▼
 tests/regression/test_reg_e2e_AtomLoad.py
           │  for each config:
           │    atom, waveMesh, path_dict = Atom.init_Atom_(conf)
           │    compare ALL fields against reference (value-by-value)
           ▼
    ┌──────────────────────┐
    │  Stage 1 safety net  │───────┐
    └──────────────────────┘       │
                                   ▼
                      Stage 2: pyright possibly-unbound fix
                      per file (AtomIO → Grotrian → ... → Experimental)
                      每个小 commit 后 rerun pytest + pyright
```

### Key Components

| Component | Responsibility | New / Modified |
|-----------|---------------|----------------|
| `tests/regression/_atom_serde.py` | 把 `Atom` + `Wavelength_Mesh` + `path_dict` 扁平化为 `{key: list|scalar}`；提供 `dump_atom(...)` + `assert_atom_matches(...)` | New |
| `scripts/gen_atom_reference.py` | 一次性脚本：遍历 8 config，调 `dump_atom`，合并 JSON 写盘 | New |
| `tests/regression/atom_reference_values.json` | 参考值文件（~8 atoms × ~50 keys） | New |
| `tests/regression/test_reg_e2e_AtomLoad.py` | 参数化 test，每 config 一个 case | New |
| `tests/regression/conftest.py` | 加 `ref_atom` session fixture | Modified |
| `pyproject.toml` | 移除两处抑制 | Modified (Stage 2) |
| `src/spectra/Util/AtomUtils/AtomIO.py` 等 | 逐文件从根源修 possibly-unbound / operator | Modified (Stage 2) |

### Data Flow

**Stage 1 — 生成**:
1. `gen_atom_reference.py` 对每 config 调 `Atom.init_Atom_(conf_path)`
2. 调 `_atom_serde.dump_atom(atom, waveMesh, path_dict, name=<ConfigName>)` 得 `dict`
3. 合并 8 份 dict → 写 `atom_reference_values.json`

**Stage 1 — 验证**:
1. `test_reg_e2e_AtomLoad.py` 参数化 fixture 注入 `(conf_rel_path, config_name, is_hydrogen)`
2. 每 case 调 `Atom.init_Atom_(conf)` 得 `(atom, waveMesh, path_dict)`
3. 调 `_atom_serde.assert_atom_matches(atom, waveMesh, path_dict, name=<ConfigName>, ref=ref_atom)`
4. helper 内部对每个 key 取 `ref_atom[f"{name}.{field}"]` 做 per-field 比较（int/bool/str 用 `==`；float 用 `np.isclose(rtol=1e-12)`；array 用 `np.testing.assert_allclose(rtol=1e-12, atol=0)` / `array_equal` for int）

**Stage 2 — 根源修**:
1. 取消 `pyproject.toml` 两处抑制 → `pyright` 报 122
2. 按文件顺序（AtomIO → Grotrian → Plotting → LTELib → HelpUtil → ExFAL → 01makeAtom）逐个修
3. 每修完一个文件 → `pytest` + `pyright` 全绿 → 单独 commit
4. 全修完后 `pre-commit run --all-files` 验收

---

## 2. Implementation Phases

### Phase 1: Stage 1 serialization helper + generator

> **Goal:** 建立序列化/反序列化/比对的 helper 与 one-shot 生成脚本，跑出 `atom_reference_values.json`
> **Estimated Effort:** 0.5 day

- [ ] Step 1.1 — 新建 `tests/regression/_atom_serde.py`，实现：
  - `_struct_array_to_dict(arr, prefix) -> dict[str, list]` — struct array → {field: list}
  - `_path_dict_to_rel(path_dict) -> dict[str, str|None]` — 绝对路径 → 相对 `CFG._ROOT_DIR`
  - `dump_atom(atom, waveMesh, path_dict, name) -> dict[str, Any]` — 扁平化全字段，key 前缀 `<name>.`
  - `assert_atom_matches(atom, waveMesh, path_dict, name, ref)` — 对每 key 做类型分派的比对
- [ ] Step 1.2 — 新建 `scripts/gen_atom_reference.py`：
  - `CONFIGS: list[tuple[str, bool]]` = 8 条（相对路径 + `is_hydrogen`）
  - 遍历 → 调 `dump_atom` → merge → `json.dump(..., indent=2)` 写盘
  - 支持 `--out` 参数（默认 `tests/regression/atom_reference_values.json`）
- [ ] Step 1.3 — 手动跑 `uv run --extra dev python scripts/gen_atom_reference.py`
- [ ] Step 1.4 — 目视检查 JSON（尺寸合理？无绝对路径？无 NaN？）

**Phase 1 Exit Criteria:**
- [ ] `atom_reference_values.json` 成功生成
- [ ] 文件不含 `/home/kouui/` 等绝对路径片段
- [ ] 文件包含 8 个 atom，每 atom 至少 40+ keys

---

### Phase 2: Stage 1 test + fixture

> **Goal:** 加 test + fixture，让 8 个 atom 全量字段比对通过
> **Estimated Effort:** 0.5 day
> **Depends on:** Phase 1

- [ ] Step 2.1 — `tests/regression/conftest.py` 加 `ref_atom` session fixture（读 `atom_reference_values.json`）
- [ ] Step 2.2 — `tests/regression/test_reg_e2e_AtomLoad.py`：
  ```python
  CONFIGS = [
      ("data/conf/H.conf",            "H",           True ),
      ("data/conf/H6.conf",           "H6",          True ),
      ("data/conf/H_theory.conf",     "H_theory",    True ),
      ("data/conf/He.conf",           "He",          False),
      ("data/conf/He_I.conf",         "He_I",        False),
      ("data/conf/He_I_II.conf",      "He_I_II",     False),
      ("data/conf/Ca_II.conf",        "Ca_II",       False),
      ("data/conf/Ca_I-II-III.conf",  "Ca_I-II-III", False),
  ]

  @pytest.mark.parametrize("conf_rel,name,is_hydrogen", CONFIGS, ids=[c[1] for c in CONFIGS])
  def test_load_atom_matches_reference(conf_rel, name, is_hydrogen, ref_atom):
      conf_path = str(CFG._ROOT_DIR / conf_rel)
      atom, waveMesh, path_dict = Atom.init_Atom_(conf_path, is_hydrogen=is_hydrogen)
      assert_atom_matches(atom, waveMesh, path_dict, name, ref_atom)
  ```
- [ ] Step 2.3 — 运行 `uv run --extra dev python -m pytest tests/regression/test_reg_e2e_AtomLoad.py -v`
- [ ] Step 2.4 — 全部跑过后跑 full regression：`pytest tests/regression/ -q`

**Phase 2 Exit Criteria:**
- [ ] 8 个新 case 全绿
- [ ] 原 247 cases 全绿（不回归）
- [ ] Stage 1 commit（"test: add e2e regression for AtomIO load coverage across 8 configs"）

---

### Phase 3: Stage 2 AtomIO 根源修

> **Goal:** 取消两处 pyright 抑制，先修 AtomIO 62 处（最大头）
> **Estimated Effort:** 1 day
> **Depends on:** Phase 2

- [ ] Step 3.1 — `pyproject.toml` 取消 `reportPossiblyUnboundVariable = false` 和 `reportOperatorIssue = false`，确认 `pyright` 报 ~122
- [ ] Step 3.2 — 跑 `uv run --extra dev pyright src/spectra/Util/AtomUtils/AtomIO.py 2>&1 | grep -E "error|warning"` 列出 AtomIO 的 62 处
- [ ] Step 3.3 — 按"先易后难"分类处理：
  - **Case A (循环变量未初始化)**：循环前加合适默认值（如 `i = -1`、`key = ""`）
  - **Case B (分支条件不完整)**：补 `else: raise ValueError(...)`
  - **Case C (配置可选字段)**：函数入口 guard + 早返回
  - **Case D (operator issue 是 Case A/B/C 的下游)**：修完 A/B/C 会自动消失
- [ ] Step 3.4 — 每改几处就跑 `pyright src/spectra/Util/AtomUtils/AtomIO.py`（增量）+ `pytest tests/regression/test_reg_e2e_AtomLoad.py -q`
- [ ] Step 3.5 — AtomIO 归零后跑 full regression + full pyright
- [ ] Step 3.6 — commit "refactor: eliminate possibly-unbound in AtomIO via root-cause fixes"

**Phase 3 Exit Criteria:**
- [ ] `pyright src/spectra/Util/AtomUtils/AtomIO.py` = 0 errors
- [ ] 所有 regression tests passed

---

### Phase 4: Stage 2 其他文件根源修

> **Goal:** 逐文件修剩余 60 处，每文件一个 commit
> **Estimated Effort:** 0.5-1 day
> **Depends on:** Phase 3

- [ ] Step 4.1 — `Visual/Grotrian.py` 21 处 —— 无回归覆盖，修后跑 `tests/examples/example.H_Grotrian.py` 烟雾测试
- [ ] Step 4.2 — `Util/HelpUtil.py` 4 处
- [ ] Step 4.3 — `Visual/Plotting.py` 2 处
- [ ] Step 4.4 — `Atomic/LTELib.py` 1 处（被 SE e2e 间接覆盖）
- [ ] Step 4.5 — `Experimental/ExFAL.py` 23 处 —— **决策点**：若工作量过大，可保留 file-level pragma `# pyright: reportPossiblyUnboundVariable=false`（决策写入 task.md 的 Q4）
- [ ] Step 4.6 — `Util/AtomicDataUtils/RH2Spectra/01makeAtom.py` 9 处 —— 与上轮一致，倾向 file-level pragma
- [ ] Step 4.7 — 每个文件单独 commit，commit message 用 `refactor: eliminate possibly-unbound in <File>`

**Phase 4 Exit Criteria:**
- [ ] `uv run --extra dev pyright` = 0 errors
- [ ] 所有 regression tests passed
- [ ] `pre-commit run --all-files` 全绿

---

### Phase 5: 终清理与 PR

> **Goal:** 收尾 —— 更新 task 状态、写 PR 描述
> **Estimated Effort:** 0.5 day
> **Depends on:** Phase 4

- [ ] Step 5.1 — 确认 `pyproject.toml` 最终状态（两处抑制已删，`stubPath` 保留）
- [ ] Step 5.2 — task.md 与 plan.md 勾选完成项，status → Done
- [ ] Step 5.3 — 本次由用户决定是否创建 PR；默认不创建

**Phase 5 Exit Criteria:**
- [ ] task.md status = Done
- [ ] （可选）PR 创建后 CI 全绿

---

## 3. Boundaries — Do NOT Touch

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| 过时 conf | `data/conf/C_III*`、`O_V*`、`Si_III*`（若存在） | 用户确认已过时 |
| Ca_I_II_III.conf | `data/conf/Ca_I_II_III.conf` | 与 `Ca_I-II-III.conf` 仅空白差异,选一即可；不动重复文件 |
| 业务逻辑 | `AtomIO.py` 中的物理算式、`init_Atom_` 的调用次序 | 本次只修类型层面,不动计算 |
| 现有 `reference_values.json` | `tests/regression/reference_values.json` | 新参考值另起 `atom_reference_values.json`,避免历史文件膨胀 |
| PR #9 scope 外的历史 `# type: ignore` | 其他文件已有的（`a4ed0d9` 之前） | 非本次 scope |

**Rule of thumb:** 若修改要碰以上路径,停下来重新评估。

---

## 4. Test Coverage

### Testing Strategy

| Level | Scope | Tool / Framework |
|-------|-------|------------------|
| Unit / Regression | `Atom.init_Atom_` 全字段 round-trip | pytest + JSON reference |
| Integration | SE / CloudModel e2e（已存在） | pytest |
| Smoke | Grotrian plot（已存在 example 脚本） | 手动运行 |

### Required Test Cases

#### Regression Tests (新增)

- [ ] `test_load_atom_matches_reference[H]` — 完整路径 + `is_hydrogen=True`
- [ ] `test_load_atom_matches_reference[H6]` — nLevel 截断 + RadiativeLine 但无 Grotrian 行
- [ ] `test_load_atom_matches_reference[H_theory]` — 无 Aji 分支,走 `make_Atom_Line_` 的 `path is None` 分支
- [ ] `test_load_atom_matches_reference[He]` — 完整字段
- [ ] `test_load_atom_matches_reference[He_I]` — 无 CIe / PI
- [ ] `test_load_atom_matches_reference[He_I_II]` — 完整字段多 stage
- [ ] `test_load_atom_matches_reference[Ca_II]` — 单 stage + 独立 atom dir
- [ ] `test_load_atom_matches_reference[Ca_I-II-III]` — 多 stage

#### Edge Cases

- [ ] 没有 CEe 的 config（`H.conf`、`H6.conf`、`H_theory.conf`）→ `CE.Te_table.size == 0`
- [ ] 没有 PI 的 config → `PI.alpha_table.shape == (2, 0)`
- [ ] `is_hydrogen=True` 且 `Aji` path 为 None（`H_theory.conf`）→ 走 `_Hydrogen.einstein_A_coefficient_`
- [ ] `path_dict` 字段为 None（缺失 conf 行）→ JSON 存 `null`

### Coverage Target

- Stage 1: 新增 regression 覆盖 AtomIO 9 个顶层装配函数的每一条 conf 相关分支（~60% 行覆盖,100% 分支选择）
- Stage 2 不降低覆盖（`pytest` 应仍全绿）

---

## 5. Key Decisions

### Decision 1: 序列化粒度 — 全量 vs fingerprint

- **Context:** `Atom` struct 含多个 numpy struct array（Level/Line/Cont/CE.Coe/…）;全量 list 存 JSON 会让 atom_reference 膨胀;fingerprint（shape + sum + arr[0/−1]）小但会漏检中间位置变动。
- **Options Considered:**
  1. **D1c fingerprint** — 小、快、可能漏检
  2. **D1b 全量** — 大、敏感
  3. **D1d 混合（pickle per atom）** — 折中但难 diff
- **Decision:** D1b（全量），user 明确要 "must verify all values in the struct (including values in the array)"
- **Rationale:** Stage 2 要改 `possibly-unbound` 分支,正好是"条件字段赋值"这类易漏检 bug 的温床;fingerprint 命中率不够。
- **Consequences:** JSON 文件会大（估 ~500KB-1MB）;需 pretty-print 以便 diff。值得。

### Decision 2: 参考文件位置 — 合并 vs 独立

- **Context:** 已存在 `tests/regression/reference_values.json` (846 行);新增 atom 全字段 ~8×50 keys 会让其过大。
- **Options Considered:**
  1. 合并进 `reference_values.json` 并前缀 `Atom.`
  2. 独立新文件 `atom_reference_values.json`
- **Decision:** 独立（option 2）
- **Rationale:** 职责分离;regenerate 时不影响原文件;两个 fixture 独立易维护。
- **Consequences:** `conftest.py` 多一个 fixture;test 文件 import 对应 fixture 即可。

### Decision 3: `path_dict` 相对 vs 绝对

- **Context:** `read_conf_` 用 `.resolve()` 返回绝对路径,不同机器会不同。
- **Options Considered:**
  1. 存绝对路径 → 参考值机器相关
  2. 相对化到 `CFG._ROOT_DIR`（user 确认）
- **Decision:** 存相对路径（user 确认 "seems better to use relative"）
- **Rationale:** JSON 可移植性;便于 code review diff。
- **Consequences:** `_atom_serde._path_dict_to_rel(path_dict)` 在 dump 与 compare 两侧都要应用；若未来 `_ROOT_DIR` 解析逻辑变,要一起更新。

### Decision 4: Stage 2 修法 — 根源修 vs file-level pragma

- **Context:** 122 处 possibly-unbound 中,`ExFAL.py` 23 处 + `01makeAtom.py` 9 处位于 Experimental / 协作者脚本,ROI 低。
- **Options Considered:**
  1. 全部根源修
  2. 核心文件根源修 + 边缘文件 file-level pragma
- **Decision:** Core 根源修,`ExFAL.py`/`01makeAtom.py` 允许 file-level pragma（如果实际修起来成本失衡）
- **Rationale:** 与 PR #9 Decision 3 一致 —— "能根源修就不 ignore,但范围限在 non-Experimental"；Experimental 不是 load 路径,影响有限。
- **Consequences:** 未来这些文件重写时要回头把 pragma 去掉。

---

## 6. Precautions

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| JSON 浮点 round-trip 不 bit-exact | Low | Med | 比对用 `rtol=1e-12, atol=0`;Python 3 `json.dumps` 用 `repr()`,IEEE-754 已 exact |
| `numpy.dtype` object 比较（struct array）麻烦 | Med | Low | 每 field 独立 list,不保留 dtype |
| Stage 2 循环变量误伤 | High | Med | 每改完跑 `pytest` + `pyright`;小步提交 |
| `_ROOT_DIR` 解析不一致（CI vs local） | Low | Med | 统一走 `CFG._ROOT_DIR`;若 CI 失败就地 debug |
| `make_Atom_PI_` 修复撞 PR #9 上轮 diff | Med | Med | 开修前 `git log -p` 读 PR #9 上轮 diff;Coe 相关字段小心 |

### Rollback Plan

_Stage 1 失败(生成的 JSON 跟代码实际 load 不一致)_:
1. 删除 `atom_reference_values.json`、`gen_atom_reference.py`、`test_reg_e2e_AtomLoad.py`、`_atom_serde.py`
2. revert `conftest.py` 的 `ref_atom` fixture
3. 独立 commit 回滚

_Stage 2 失败(pyright 根源修引入回归)_:
1. 逐文件 revert 对应 commit
2. 针对失败文件重开子任务
3. Stage 1 test 是 safety net,能立刻发现

### Migration Notes

- 无 data migration
- `atom_reference_values.json` 是新增文件,未来 AtomIO 业务逻辑变更时需 regenerate（文档化在 `scripts/gen_atom_reference.py` 顶部 docstring）
- 无 breaking change

### Performance Considerations

- 新 regression 8 cases,每个 `init_Atom_` < 1s（主要耗在 numba JIT warm-up,但已被 SE e2e 覆盖）。额外 <10s。
- 不影响运行时性能（只改类型层面）

### Security Considerations

- 无外部输入;读本地 `data/` 文件
- 无 credential 改动
