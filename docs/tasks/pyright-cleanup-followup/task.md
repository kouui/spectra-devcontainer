# Task: pyright cleanup follow-up

> **Status:** Draft
> **Owner:** kouui
> **Created:** 2026-04-19
> **Last Updated:** 2026-04-19

## Objective

收拾上一轮 pyright 大清洗（`239 → 0`，commit `a4ed0d9`）遗留的两项技术债：
1. 用 Python 3.13 原生 `type` 语句重写 `Types.py` 中的 Union / Generic 类型别名，消除 `reportInvalidTypeForm` 的 34 处误报，并从 `pyproject.toml` 移除该规则的全局抑制。
2. 复审本次 pyright 修复新增的 27 处 `# type: ignore`：可替换的换成更干净的写法（`assert` / 显式转换 / 签名修正），掩盖真 bug 的就地修掉，库/stub 限制的保留。

## Background & Context

- `commit a4ed0d9 (fix: resolve all pyright errors 239 → 0)`：为快速收敛把部分运行时正确、pyright 抱怨的位置用 `# type: ignore` 和全局抑制规则绕过。
- `pyproject.toml` 目前全局关闭 `reportInvalidTypeForm`，是因为 `T_VEC_FA = T_FLOAT | T_ARRAY` 在 `T_FLOAT = float` 的语义下会被 pyright 当成运行时 `types.UnionType` 表达式而非类型别名。
- Python 3.13 原生 `type` 语句（PEP 695）能精确表达类型别名，pyright 可正确识别为 `TypeAliasType`。

- Spec: `tmp/handoff.md`（本地工作区）
- Related Issues: 无

## Requirements

### Functional Requirements

1. `pyproject.toml` 中 `reportInvalidTypeForm = false` 被移除后，`uv run --extra dev pyright` 仍为 0 错误。
2. 本次 pyright 修复新增的 27 处 `# type: ignore` 中，至少 20 处被替换为更干净的写法或彻底修掉真 bug；剩余保留条目必须附一行注释说明原因。
3. 回归测试 `tests/regression/` 247 passed（含 JIT 路径 `CFG._IS_JIT=True`）。
4. `pre-commit run --all-files` 全绿（ruff + pyright）。

### Non-Functional Requirements

- 性能：不改变运行时行为；numba JIT 装饰路径（`@nb_njit`、`@nb_vec`）不能退化。
- 维护性：保留的 `# type: ignore` 必须有注释说明（哪个库缺 stub / 为何不能用类型表达）。
- 向后兼容：`Types.py` 的公共名字（`T_VEC_FA` 等）保持不变，只换实现方式。

## Scope

### In Scope

- [ ] `src/spectra/Types.py` — `T_VEC_*`、`T_CTJ_*`、`T_IDX_PAIR_TABLE`、`T_E_*` 用 `type` 语句重写
- [ ] `pyproject.toml` — 移除 `reportInvalidTypeForm = false`
- [ ] `src/spectra/Visual/Grotrian.py` — 用构造器层面保证 `self.fig` 非 None，消除 7 处 `union-attr` ignore
- [ ] `src/spectra/Visual/Plotting.py` — 修复 `:69`（收窄 None）和 `:136`（`add_axes` 签名）
- [ ] `src/spectra/RadiativeTransfer/Profile.py` — `.real`/`.imag` 改 `numpy.real()`/`numpy.imag()`
- [ ] `src/spectra/Experimental/ExScatter.py` — 修 `nHI_pop_LTE` 签名（真 bug: `T_FLOAT` → `T_VEC_FA`）
- [ ] `src/spectra/Function/Icp/SELib.py:477` — 检查 `SE_Container` 构造器缺参
- [ ] `src/spectra/Util/HelpUtil.py:81` — `dtype.names is None` 防御
- [ ] `src/spectra/Util/AtomUtils/AtomIO.py:1035-1037` — 显式 `int()` 转换
- [ ] `src/spectra/Util/AtomicDataUtils/MakePhotoioniz.py:115` — 修 `_v1_fit_func_` 签名
- [ ] `src/spectra/Function/Hydrogen/DegenerateN.py:51` — 确认 `PI_cross_section_` overload 覆盖
- [ ] `src/spectra/ImportAll.py:27-32` — 若 `nb_List` 定义可简化则一起改
- [ ] `src/spectra/Atomic/PhotoIonize.py:100` — `fill_value` tuple 传 scipy，保留 ignore（stub 限制）并加注释
- [ ] `src/spectra/Configurations.py:55` — `THREADING_LAYER` 属性访问，保留 ignore 并加注释

### Out of Scope (Boundaries)

- **`src/spectra/Util/AtomicDataUtils/RH2Spectra/01makeAtom.py`**: 协作者维护的 RH → Spectra 转换脚本，文件名以数字开头不可 import；使用独立 dataclass（`CollisionRH`/`AlphaRH`/`LevelRH`）和原生类型标注。该文件内 3 处 `# type: ignore` 本次保留。
- **`src/spectra/Atomic/ContinuumOpacity.py`、`src/spectra/Atomic/Hydrogen.py`、`src/spectra/Warnings.py` 等先前已有的 `# type: ignore`**：属于 `commit a4ed0d9` 之前的历史债，不在本次 27 处范围内，不处理。
- **numba 类型表达式** (`nb_types.float64[:]` 等注释掉的历史代码)：`Types.py` 末尾的注释块保留原样。
- **新增功能或重构**：本次只做类型清理，不动任何业务逻辑。

## Acceptance Criteria

- [ ] `pyproject.toml` 不再有 `reportInvalidTypeForm = false`
- [ ] `uv run --extra dev pyright` 输出 `0 errors, 0 warnings`
- [ ] 新增的 27 处 ignore 中被消除的数量 ≥ 20
- [ ] 保留的 ignore 每处都有一行说明注释
- [ ] `uv run --extra dev python -m pytest tests/regression/ -q` 247 passed
- [ ] `uv run --extra dev pre-commit run --all-files` 通过
- [ ] JIT 路径 smoke test：`tests/regression/test_reg_e2e_SE.py` 在默认配置下通过（覆盖 `@nb_njit` / `@nb_vec` 装饰）

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| Python 3.13 `type` 语句 | cpython | stable (PEP 695) | 项目已锁 `requires-python = ">=3.13"` |
| numba JIT 不读 Python 注解 | numba | 经验证 | `@nb_njit` 用运行时类型推断；`@nb_vec` 用字符串签名。`TypeAliasType` 不影响 |

## Risks & Open Questions

- [ ] **R1**: `type T_VEC_FA = ...` 创建的是 `TypeAliasType` 实例，运行时 `isinstance(x, T_VEC_FA)` 会失败。grep 全仓库已确认**无此类用法**（`isinstance\s*\([^,]+,\s*T_(VEC|CTJ|IDX)` 零匹配），风险已消除。
- [ ] **R2**: numba 对 `type` 语句别名的解析尚未实测。缓解：Phase 1 完成后立刻跑 `test_reg_e2e_SE.py`。
- [ ] **R3**: `Visual/Grotrian.py` 没有回归测试覆盖。缓解：修改后手动跑 `notebooks/` 里一个使用 Grotrian 的 notebook，或用 `matplotlib.use('Agg')` 写烟雾测试。
- [ ] **Q1**: `Function/Icp/SELib.py:477` 的 `SE_Container` 构造器缺 `Ntotal/Nh/Ne/Te` 是真 bug 还是 dataclass 有默认值？需要读 `_Container.py` 确认。
- [ ] **Q2**: `DegenerateN.py:51` 的 `PI_cross_section_` 是否需要补第 3 个 `@OVERLOAD`？需要看 `Eratio` 实际类型。

## References

- `tmp/handoff.md` — 前一轮遗留的 27 处分类报告
- `commit a4ed0d9` — pyright 239 → 0 的修复批次
- PEP 695 — Python 3.12+ `type` 语句规范
