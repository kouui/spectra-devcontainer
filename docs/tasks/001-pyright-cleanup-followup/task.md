# Task: pyright cleanup follow-up

> **Status:** Done
> **Owner:** kouui
> **Created:** 2026-04-19
> **Last Updated:** 2026-04-19

## Objective

收拾上一轮 pyright 大清洗（`239 → 0`，commit `a4ed0d9`）遗留的两项技术债，遵循**能从根源修就不 ignore**的原则：

1. 用 Python 3.13 原生 `type` 语句重写 `Types.py` 中的 Union / Generic 类型别名，消除 `reportInvalidTypeForm` 的 34 处误报，并从 `pyproject.toml` 移除该规则的全局抑制。
2. 把本次 pyright 修复新增的 27 处 `# type: ignore` **全部**从根源消除：
   - C 类（9 处）修签名 / 显式转换 / 防御 None（改错误代码）
   - B 类（11 处）用构造器保证 / 结构重构 / numpy 函数式 API（改写法）
   - A 类（5 处）为 numba / scipy 写项目内 `.pyi` stub 文件（pyright `stubPath`）——不再保留 ignore + 注释的妥协方案

## Background & Context

- `commit a4ed0d9 (fix: resolve all pyright errors 239 → 0)`：为快速收敛把部分运行时正确、pyright 抱怨的位置用 `# type: ignore` 和全局抑制规则绕过。
- `pyproject.toml` 目前全局关闭 `reportInvalidTypeForm`，是因为 `T_VEC_FA = T_FLOAT | T_ARRAY` 在 `T_FLOAT = float` 的语义下会被 pyright 当成运行时 `types.UnionType` 表达式而非类型别名。
- Python 3.13 原生 `type` 语句（PEP 695）能精确表达类型别名，pyright 可正确识别为 `TypeAliasType`。

- Spec: `tmp/handoff.md`（本地工作区）
- Related Issues: 无

## Requirements

### Functional Requirements

1. `pyproject.toml` 中 `reportInvalidTypeForm = false` 被移除后，`uv run --extra dev pyright` 仍为 0 错误。
2. 本次 pyright 修复新增的 27 处 `# type: ignore` **全部消除**（27/27）。若某处确实无法根源修，必须停下来开 follow-up issue 讨论，**不得就地保留 ignore 了事**。
3. 为 numba（`numba.typed.List`、`numba.njit`、`numba.vectorize`、`numba.core.config.THREADING_LAYER`、`numba.set_num_threads`、`numba.objmode`）和 scipy（`scipy.interpolate.interp1d` 的 `fill_value` 参数）提供项目内 `.pyi` stub 文件，`pyproject.toml [tool.pyright]` 增加 `stubPath = "typings"`。
4. 回归测试 `tests/regression/` 247 passed（含 JIT 路径 `CFG._IS_JIT=True`）。
5. `pre-commit run --all-files` 全绿（ruff + pyright）。

### Non-Functional Requirements

- 性能：不改变运行时行为；numba JIT 装饰路径（`@nb_njit`、`@nb_vec`）不能退化。
- 维护性：保留的 `# type: ignore` 必须有注释说明（哪个库缺 stub / 为何不能用类型表达）。
- 向后兼容：`Types.py` 的公共名字（`T_VEC_FA` 等）保持不变，只换实现方式。

## Scope

### In Scope

**Types 重写**
- [ ] `src/spectra/Types.py` — `T_VEC_*`、`T_CTJ_*`、`T_IDX_PAIR_TABLE`、`T_E_*` 用 `type` 语句重写
- [ ] `pyproject.toml` — 移除 `reportInvalidTypeForm = false`，添加 `stubPath = "typings"`

**C 类真 bug（9 处，全消）**
- [ ] `src/spectra/Experimental/ExScatter.py` — 修 `nHI_pop_LTE` 签名（真 bug: `T_FLOAT` → `T_VEC_FA`）—— 2 处
- [ ] `src/spectra/Util/AtomicDataUtils/MakePhotoioniz.py:115` — 修 `_v1_fit_func_` 签名
- [ ] `src/spectra/Util/AtomUtils/AtomIO.py:1035-1037` — 显式 `int()` 转换 —— 2 处
- [ ] `src/spectra/Util/HelpUtil.py:81` — `dtype.names is None` 防御
- [ ] `src/spectra/Function/Icp/SELib.py:477` — 读 `_Container.py` 确认并补全 `SE_Container` 构造器参数
- [ ] `src/spectra/Function/Hydrogen/DegenerateN.py:51` — 补 `PI_cross_section_` overload 或显式类型转换

**B 类样式（11 处，全消，无 assert 退路）**
- [ ] `src/spectra/Visual/Grotrian.py` — **类型层面**让 `self.fig: Figure` 非 Optional（修改 `__init__` 结构）；若做不到，停下来开 follow-up —— 7 处
- [ ] `src/spectra/Visual/Plotting.py:69` — 重构 elif 结构，确保 `points` 在所有分支被赋值
- [ ] `src/spectra/Visual/Plotting.py:136` — 调用方改用元组字面量 `(0.0, 0.0, 1.0, 1.0)`，函数签名声明 `tuple[float, float, float, float]`
- [ ] `src/spectra/RadiativeTransfer/Profile.py:248-249` — `.real`/`.imag` 改 `numpy.real()`/`numpy.imag()` —— 2 处

**A 类库 stub 限制（5 处，全消，通过写 `.pyi`）**
- [ ] `typings/numba/__init__.pyi` — 新增，覆盖 `njit`、`vectorize`、`set_num_threads`、`objmode`
- [ ] `typings/numba/typed/__init__.pyi` — 新增，覆盖 `List`（供 `ImportAll.py:12, 27, 30, 32`）
- [ ] `typings/numba/core/config.pyi` — 新增，覆盖 `THREADING_LAYER` 属性（供 `Configurations.py:55`）
- [ ] `typings/scipy/interpolate/__init__.pyi` — 新增，覆盖 `interp1d` 的 `fill_value` 接受 `float | tuple[float, float]`（供 `PhotoIonize.py:100`）

**连带清理**
- [ ] `src/spectra/ImportAll.py:27-32` — stub 生效后，`nb_List` 的 `# type: ignore` 全部删除，类型注解可能也能简化

### Out of Scope (Boundaries)

- **`src/spectra/Util/AtomicDataUtils/RH2Spectra/01makeAtom.py`**: 协作者维护的 RH → Spectra 转换脚本，文件名以数字开头不可 import；使用独立 dataclass（`CollisionRH`/`AlphaRH`/`LevelRH`）和原生类型标注。该文件内 3 处 `# type: ignore` 本次保留（不属于 27 处）。
- **`src/spectra/Atomic/ContinuumOpacity.py`、`src/spectra/Atomic/Hydrogen.py`、`src/spectra/Warnings.py` 等先前已有的 `# type: ignore`**：属于 `commit a4ed0d9` 之前的历史债，不在本次 27 处范围内。未来可独立任务处理（届时 `typings/` 已有 numba stubs 可复用）。
- **完整 numba / scipy stubs**：本次只写**用到的**符号（最小化原则），不追求完整覆盖。
- **numba 类型表达式** (`nb_types.float64[:]` 等注释掉的历史代码)：`Types.py` 末尾的注释块保留原样。
- **新增功能或重构**：本次只做类型清理，不动任何业务逻辑。

## Acceptance Criteria

- [x] `pyproject.toml` 不再有 `reportInvalidTypeForm = false`，并新增 `stubPath = "typings"`
- [x] `uv run --extra dev pyright` 输出 `0 errors, 0 warnings`
- [x] 新增的 27 处 ignore **全部消除**（27/27）—— C 9/9, B 11/11（含 Grotrian 7 处通过 Path E 根源修），A 5/5
- [x] `typings/` 下至少包含 `numba/__init__.pyi`、`numba/typed/__init__.pyi`、`numba/core/config.pyi`、`scipy/interpolate/__init__.pyi`
- [x] `uv run --extra dev python -m pytest tests/regression/ -q` 247 passed
- [x] `uv run --extra dev pre-commit run --all-files` 通过
- [x] JIT 路径 smoke test：`tests/regression/test_reg_e2e_SE.py` 在默认配置下通过（覆盖 `@nb_njit` / `@nb_vec` 装饰）
- [x] Grotrian smoke test：构造 + `make_fig` + `save_fig` 无错，且 `plt.get_fignums()` 不积压（Path E 的 `_ensure_fig_` 在创建新 figure 前 `plt.close()` 旧 figure）

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
