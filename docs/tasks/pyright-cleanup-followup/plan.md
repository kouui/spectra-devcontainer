# Plan: pyright cleanup follow-up

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-04-19
> **Target Completion:** 2026-04-26

---

## 0. Context

> **Objective:** 用 Python 3.13 `type` 语句重写 `Types.py` 消除 `reportInvalidTypeForm` 全局抑制；并把上轮 pyright 修复新增的 27 处 `# type: ignore` **全部从根源消除**（C 类改代码 bug，B 类改写法，A 类写项目内 stub）。
> **Full spec:** [task.md](./task.md)
>
> **核心原则**：能从根源修就不 ignore。本 plan 五阶段执行：
> **Phase 1** Types.py 重写 → **Phase 2** C 类 bug → **Phase 3** B 类样式（无 assert 退路）→ **Phase 4** A 类 `.pyi` stub → **Phase 5** 收尾。

---

## 1. Overall Architecture

### 系统总览

```
  ┌─────────────────────┐
  │  Types.py (source)  │  ← Phase 1: type 语句重写
  └──────────┬──────────┘
             │ re-exports via ImportAll.py
             ▼
  ┌─────────────────────┐        ┌────────────────────────┐
  │ 全包 src/spectra/**  │───────▶│  pyright               │
  │ 使用 T_VEC_FA 等别名  │        │  - stubPath = typings/ │  ← Phase 4 新增
  └──────────┬──────────┘        │  - reportInvalidTypeForm│ ← Phase 1 移除
             │                   │    抑制已移除           │
             │                   └────────────────────────┘
             ▼
  ┌──────────────────────┐
  │  27 处 ignore 全消    │ ← Phase 2(C)/3(B)/4(A) 目标
  │  回归测试 247 passed  │ ← 每 Phase 出口必须绿
  └──────────────────────┘

  typings/                       ← Phase 4 新增 stub 树
  ├── numba/
  │   ├── __init__.pyi           (njit, vectorize, set_num_threads, objmode)
  │   ├── typed/__init__.pyi     (List)
  │   └── core/config.pyi        (THREADING_LAYER)
  └── scipy/interpolate/
      └── __init__.pyi           (interp1d.fill_value accepts tuple)
```

### 关键组件

| Component | Responsibility | New / Modified |
|-----------|---------------|----------------|
| `src/spectra/Types.py` | 类型别名源头 | Modified |
| `pyproject.toml [tool.pyright]` | 去掉 `reportInvalidTypeForm`；加 `stubPath = "typings"` | Modified |
| `typings/numba/__init__.pyi` | numba 顶层 API stub（njit/vectorize/objmode/set_num_threads） | **New** |
| `typings/numba/typed/__init__.pyi` | `numba.typed.List` stub | **New** |
| `typings/numba/core/config.pyi` | `THREADING_LAYER` 属性 stub | **New** |
| `typings/scipy/interpolate/__init__.pyi` | `interp1d.fill_value` 接受 tuple | **New** |
| `Visual/Grotrian.py` | 构造器保证 `self.fig: Figure` 非 Optional | Modified |
| `Experimental/ExScatter.py` | 函数签名 `T_FLOAT → T_VEC_FA` | Modified (签名 bug) |
| `Util/AtomicDataUtils/MakePhotoioniz.py` | `_v1_fit_func_` 签名修正 | Modified (签名 bug) |
| `Util/AtomUtils/AtomIO.py` | 显式 `int()` 转换 | Modified |
| `Util/HelpUtil.py` | `dtype.names is None` 防御 | Modified |
| `RadiativeTransfer/Profile.py` | `.real`/`.imag` → `numpy.real`/`imag` | Modified |
| `Visual/Plotting.py` | 重构 elif + `add_axes` 签名 | Modified |
| `Function/Icp/SELib.py` | 补全 `SE_Container` 构造参数 | Modified |
| `Function/Hydrogen/DegenerateN.py` | `PI_cross_section_` overload / 显式转换 | Modified |
| `ImportAll.py` | stub 生效后删 `nb_List` 所有 ignore | Modified |

### 数据流（主流程）

1. **Phase 1** 重写 `Types.py`（`type T_VEC_FA = ...`）→ 移除 `reportInvalidTypeForm` 抑制 → pyright 0 错误 → 回归
2. **Phase 2** C 类真 bug 9 处根源修（签名 / 显式转换 / None 防御）→ pyright 0 错误 → 回归
3. **Phase 3** B 类样式 11 处根源修（构造器非 Optional / elif 重构 / numpy 函数式 API）→ pyright 0 错误 → 回归
4. **Phase 4** 写 `typings/` stub → 配置 `stubPath` → 删 A 类 5 处 ignore → pyright 0 错误 → 回归
5. **Phase 5** 最终 `pre-commit run --all-files` → 手动 Grotrian 验证 → 状态改 Review

---

## 2. Implementation Phases

### Phase 1: Types.py 重写 + 移除全局抑制

> **Goal:** 消除 34 个 `reportInvalidTypeForm`，从 `pyproject.toml` 移除该抑制规则
> **Estimated Effort:** 0.5 day

- [ ] **Step 1.1** — 在 `Types.py` 把以下别名改为 `type` 语句：
  - `T_VEC_IFA`、`T_VEC_FA`、`T_VEC_IA`
  - `T_CTJ_TABLE`、`T_CTJ_PAIR`、`T_CTJ_PAIR_TABLE`、`T_IDX_PAIR_TABLE`
  - 所有 `T_E_*`（`T_E_ATOMIC_DATA_SOURCE`、`T_E_ATOM`、`T_E_COLLISIONAL_TRANSITION*`、`T_E_ABSORPTION_PROFILE_TYPE`、`T_E_ATMOSPHERE_COORDINATE_TYPE`）
  - **不改** `T_FLOAT = float` 等直接指向类对象的别名（pyright 本来就识别）。
- [ ] **Step 1.2** — 确认 `__all__` 的导出名完整无漏。
- [ ] **Step 1.3** — `pyproject.toml` 删除 `reportInvalidTypeForm = false`。
- [ ] **Step 1.4** — 跑 `uv run --extra dev pyright` → 应仍为 0 错误 0 警告。
- [ ] **Step 1.5** — 跑完整回归 `uv run --extra dev python -m pytest tests/regression/ -q` → 247 passed。
- [ ] **Step 1.6** — 打开 `CFG._IS_JIT=True`（或默认即是）跑 `test_reg_e2e_SE.py` 确认 numba 装饰路径无退化。

**Phase 1 Exit Criteria:**
- [ ] pyright 0 错误
- [ ] 回归 247 passed
- [ ] JIT 路径通过

---

### Phase 2: C 类真 bug 根源修复（9 处全消）

> **Goal:** 修掉 ignore 掩盖的真实类型 bug，签名 / 转换 / 防御全部走根源
> **Estimated Effort:** 1 day
> **Depends on:** Phase 1

- [ ] **Step 2.1** — `Experimental/ExScatter.py`
  - 函数 `HI_bf_emissivity_LTE_` 签名 `nHI_pop_LTE: T_FLOAT` 改为 `T_VEC_FA`
  - 删除 `:30` 和 `:38` 两处 ignore
- [ ] **Step 2.2** — `Util/AtomicDataUtils/MakePhotoioniz.py:115`
  - `_v1_fit_func_` 签名 `wave: T_FLOAT` 改为 `T_VEC_FA`
  - 删除 ignore
- [ ] **Step 2.3** — `Util/AtomUtils/AtomIO.py:1035-1037`
  - 显式 `z = int(Level["stage"][Coe["idxI"][i]])` 将 `numpy.int64` 转为 `int`
  - 删除两处 ignore
- [ ] **Step 2.4** — `Util/HelpUtil.py:81`
  - 在 `for field_name in dtype.names:` 之前加 `dtype.names is None` 防御（return / raise，视上下文）
  - 删除 ignore
- [ ] **Step 2.5** — `Function/Icp/SELib.py:477`
  - 读 `_Container.SE_Container` 定义确认 `Ntotal/Nh/Ne/Te` 语义
  - 若有默认值：删除 ignore；若无：从调用上下文补齐参数
  - **若判断为深层逻辑 bug（需改业务）**：停下来开 follow-up issue，本 Phase 不强行处理（但也**绝不保留 ignore**）
- [ ] **Step 2.6** — `Function/Hydrogen/DegenerateN.py:51`
  - 优先级：显式转换（`int(ni)` 等）> 补第 3 个 `@OVERLOAD` > 签名放宽
  - 根据读 `_Hydrogen.PI_cross_section_` 的结果决策
  - 删除 ignore
- [ ] **Step 2.7** — 每步改完跑 pyright + 回归。

**Phase 2 Exit Criteria:**
- [ ] C 类 9 处 ignore 全部消除（若 2.5 开 follow-up，则 8/9 + issue 链接）
- [ ] pyright 0 错误
- [ ] 回归 247 passed

---

### Phase 3: B 类样式根源修复（11 处全消，无 assert 退路）

> **Goal:** 从**类型层面**根源修复，不用 `assert` 贴创可贴
> **Estimated Effort:** 1 day
> **Depends on:** Phase 2

- [ ] **Step 3.1** — `Visual/Grotrian.py` 7 处 `self.fig` ignore（根源：类型非 Optional）
  - 读 `Grotrian.__init__`，找 `self.fig = None` / `self.fig: Optional[Figure] = None` 的位置
  - **根源方案**：在 `__init__` 里直接创建 `fig`（或要求调用者传入），类型声明改为 `self.fig: Figure`（非 Optional）
  - **禁止退化方案**：在每个方法入口写 `assert self.fig is not None` —— 这是对症修复，违反"根源优先"原则
  - 若 `__init__` 结构上无法保证非 Optional（例如有延迟创建的合理需求）：停下来开 follow-up 讨论重构方案，本 Phase **不就地 assert**
  - 删除 7 处 `union-attr` ignore
- [ ] **Step 3.2** — `Visual/Plotting.py:69`（根源：控制流赋值覆盖）
  - 读 `:64-69`，重构 elif 结构使 `points` 在所有可达分支被非 None 赋值：
    ```python
    elif points is None:
        if axis == "x": points = axe.get_xticks()[:-1].astype(np.int64)
        elif axis == "y": points = axe.get_yticks()[:-1].astype(np.int64)
        else: raise ValueError(f"invalid axis: {axis}")
        points = points[points >= 0]
    ```
  - **禁止** `assert points is not None`
  - 删除 ignore
- [ ] **Step 3.3** — `Visual/Plotting.py:136`（根源：签名 + 调用方）
  - 调用方 `axe_kw = {"ax1": [0, 0, 1, 1]}` 改为 `"ax1": (0.0, 0.0, 1.0, 1.0)`
  - 函数签名声明 `axe_kw: dict[str, tuple[float, float, float, float]] | None = None`
  - `add_axes(val)` 直接传元组，不 `tuple(val)`
  - 删除 ignore
- [ ] **Step 3.4** — `RadiativeTransfer/Profile.py:248-249`（根源：用 numpy 函数式 API）
  - `w4.real` → `_numpy.real(w4)`；`w4.imag` → `_numpy.imag(w4)`
  - 删除 2 处 ignore
- [ ] **Step 3.5** — 每步改完跑 pyright + 回归。
- [ ] **Step 3.6** — 手动跑 Grotrian 的 notebook（或写 `tests/regression/test_reg_grotrian.py` 冒烟用 `matplotlib.use("Agg")`）确保绘图未破坏。

**Phase 3 Exit Criteria:**
- [ ] B 类 11 处 ignore 全部消除（若 3.1 开 follow-up，则 4/11 + issue 链接；但 C 类可能相应调整）
- [ ] pyright 0 错误
- [ ] 回归 247 passed
- [ ] Grotrian 手动验证 OK

---

### Phase 4: A 类写 `.pyi` stub 根源修复（5 处全消）

> **Goal:** 通过项目内 type stubs 根治 numba / scipy 库 stub 缺失问题
> **Estimated Effort:** 0.5 day
> **Depends on:** Phase 3

- [ ] **Step 4.1** — `pyproject.toml [tool.pyright]` 添加 `stubPath = "typings"`
- [ ] **Step 4.2** — 写 `typings/numba/__init__.pyi`（只覆盖本项目用到的符号）：
  ```python
  from collections.abc import Callable
  from typing import Any, overload
  def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]: ...
  def vectorize(*args: Any, **kwargs: Any) -> Callable[..., Any]: ...
  def set_num_threads(n: int) -> None: ...
  def objmode(*args: Any, **kwargs: Any) -> Any: ...
  ```
- [ ] **Step 4.3** — 写 `typings/numba/typed/__init__.pyi`：
  ```python
  from typing import Any, Generic, TypeVar
  _T = TypeVar("_T")
  class List(Generic[_T]):
      def append(self, item: _T) -> None: ...
      # 按实际用法补充；保持最小
  ```
- [ ] **Step 4.4** — 写 `typings/numba/core/config.pyi`：
  ```python
  THREADING_LAYER: str
  ```
- [ ] **Step 4.5** — 写 `typings/scipy/interpolate/__init__.pyi`：
  - 覆盖 `interp1d` 的 `fill_value` 参数为 `float | tuple[float, float]`
  - 注意：只覆盖 `interp1d` 一个符号；其他 `scipy.interpolate` 调用仍走原生 stubs（如有）
- [ ] **Step 4.6** — 删除以下 ignore：
  - `ImportAll.py:12` (`from numba.typed import List`)
  - `ImportAll.py:15, 16` (`njit`, `vectorize`)
  - `ImportAll.py:27` (`nb_List: T_TYPE[List | T_LIST]`)
  - `ImportAll.py:30, 32` (`nb_List = List` / `nb_List = list`)
  - `Configurations.py:12, 14` (`from numba.core import config`, `set_num_threads`)
  - `Configurations.py:55` (`nb_config.THREADING_LAYER = ...`)
  - `Atomic/PhotoIonize.py:17` (`from scipy.interpolate import interp1d`)
  - `Atomic/PhotoIonize.py:100` (`fill_value=fill_value`)
  - `Warnings.py:11, 12` (`njit`, `objmode`)

  > 注：这些 import-level ignore 大多属于 `commit a4ed0d9` 之前的历史债，不在本次 27 处计数内，但 stubs 生效后顺带清掉零成本。本次 27 处 A 类目标：`ImportAll.py:27/30/32`、`Configurations.py:55`、`PhotoIonize.py:100`。
- [ ] **Step 4.7** — 跑 pyright + 回归。

**Phase 4 Exit Criteria:**
- [ ] 本次 27 处 A 类 5 处全部消除（`ImportAll.py` 3 + `Configurations.py` 1 + `PhotoIonize.py` 1）
- [ ] pyright 0 错误
- [ ] 回归 247 passed
- [ ] JIT 路径 smoke test 通过（`test_reg_e2e_SE.py`）

---

### Phase 5: 收尾

> **Goal:** 最终全仓 `pre-commit` 绿；汇总报告
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 4

- [ ] **Step 5.1** — 跑 `uv run --extra dev pre-commit run --all-files`
- [ ] **Step 5.2** — 全仓 grep 确认 27 处目标 ignore 已全消；统计剩余 ignore（应只剩 `01makeAtom.py` 的 3 处和历史债）
- [ ] **Step 5.3** — 更新 `task.md` 的 Acceptance Criteria 勾选，状态置为 Review
- [ ] **Step 5.4** — 若 Phase 2/3 有 follow-up issue，在 PR 描述里链接
- [ ] **Step 5.5** — 准备 PR 描述（若用户要开 PR）

**Phase 5 Exit Criteria:**
- [ ] `pre-commit run --all-files` 全绿
- [ ] 27 处 ignore 全消（或明确 follow-up）
- [ ] `task.md` 状态 Review

---

## 3. Boundaries — Do NOT Touch

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| RH 转换脚本 | `src/spectra/Util/AtomicDataUtils/RH2Spectra/01makeAtom.py` | 文件名以数字开头无法 import；使用独立 dataclass；协作者维护 |
| 历史 ignore（非 27 处） | `src/spectra/Atomic/ContinuumOpacity.py`、`src/spectra/Atomic/Hydrogen.py`、`src/spectra/Atomic/PhotoIonize.py:17`（import 层） 等 `commit 44a2bfa` 之前的 ignore | 不在本次 27 处范围内。注：Phase 4 写 numba stubs 后，`ImportAll.py`/`Warnings.py`/`Configurations.py` 的 import 层 ignore 可以顺带清掉（零成本），但不强制 |
| numba 类型表达式历史注释 | `src/spectra/Types.py:110-148`（所有 `# import numba.types` 等注释块） | 保留为文档 |
| 业务逻辑 | 任何非类型层面的修改 | 本次只做类型清理 |
| 完整 stubs | `typings/numba/**`、`typings/scipy/**` 未用到的符号 | 只写项目实际 import 的部分，最小化原则 |

**Rule of thumb:** 修改超出本清单范围 → 停下来回到 task.md 讨论。

---

## 4. Test Coverage

### 测试策略

| Level | Scope | Tool / Framework |
|-------|-------|------------------|
| Static | 类型检查 | pyright |
| Static | 风格检查 | ruff |
| Unit + Integration | 回归测试套件 | pytest (`tests/regression/`) |
| Manual | Grotrian 绘图 | 手动跑 notebook |

### 必须验证的场景

#### Static 检查

- [ ] `uv run --extra dev pyright` → `0 errors, 0 warnings`
- [ ] `uv run --extra dev pre-commit run --all-files` 全绿

#### 回归测试

- [ ] `tests/regression/` 247 passed
- [ ] `test_reg_e2e_SE.py` 在 JIT 默认开启配置下 passed

#### 边界场景

- [ ] numba 装饰函数（`@nb_njit`、`@nb_vec`）首次调用能正确编译（caches 命中/重建）
- [ ] `Grotrian` 生成 PNG 不报错
- [ ] 用 `Types.py` 别名做注解的函数在运行时可被 `typing.get_type_hints()` 解析（冒烟：import 不报错即可）

### 覆盖目标

- 消除率：本次新增 27 处 ignore 中 ≥ 20 处消除（~74%）
- 保留条目：每处附一行注释说明

---

## 5. Key Decisions

### Decision 1: 用 `type` 语句而非 `TypeAlias`

- **Context:** `Types.py` 的 Union 别名 pyright 识别失败，需要一种明确的类型别名写法。
- **Options Considered:**
  1. `from typing import TypeAlias; T_VEC_FA: TypeAlias = T_FLOAT | T_ARRAY` — 兼容 Python 3.10+，语义清楚。
  2. `type T_VEC_FA = T_FLOAT | T_ARRAY` (PEP 695) — Python 3.12+，lazy 求值，语法最简洁。
- **Decision:** 选 2（用户明确选择）。
- **Rationale:** 项目 `requires-python = ">=3.13"`；`type` 语句是语言层原生支持，未来 Python 工具链标配；lazy 求值避免导入期 name binding 问题。
- **Consequences:** 
  - 别名实例类型是 `typing.TypeAliasType` 而非 `types.UnionType`
  - 运行时 `isinstance(x, T_VEC_FA)` 会失败 —— 已 grep 全仓库无此用法
  - numba `@nb_njit` 不读 Python 注解，不受影响

### Decision 2: `01makeAtom.py` 保持现状

- **Context:** 该文件 `# type: ignore` 无法消除（独立类型体系），且文件名数字开头无法 import。
- **Options Considered:**
  1. 重构为标准模块 — 破坏与 RH 工具链的兼容
  2. 迁出 `src/` 到 `scripts/` — 结构改动大
  3. 保持现状 — 最低侵入
- **Decision:** 3
- **Rationale:** 不在 27 处新增 ignore 清理目标内；未来可独立任务处理。
- **Consequences:** `pyright` 仍扫描该文件，但既有 ignore 足以保持 0 错误。

### Decision 3: Grotrian.py 的 `self.fig` 用构造器保证非 Optional，**严禁**退回 per-method `assert`

- **Context:** 7 处 `union-attr` ignore 全因 `self.fig: Optional[Figure]`。
- **Options Considered:**
  1. 构造器保证 `self.fig: Figure` 非 Optional — **根源修**
  2. 每方法入口 `assert self.fig is not None` — **对症修（assert 是运行时窄化，没消除类型上的 Optional 根因）**
  3. 若 1 结构上无法做到：开 follow-up issue 重构，本 Phase 不就地 assert
- **Decision:** 1 或 3；禁止 2
- **Rationale:** 遵循"能从根源修就不 ignore"原则。`assert` 是运行时断言，把问题从"编译期 ignore"转移到"运行期 assert"，类型层面的根因（`self.fig` 可能为 None）没有被消除。
- **Consequences:** 需读 `Grotrian.__init__`；若现状有合理的延迟初始化需求，本任务不强行重构，改为 follow-up。

### Decision 4: A 类 5 处通过 `typings/` stub 根源修，**不保留 ignore + 注释**

- **Context:** A 类 ignore（`ImportAll.py:27-32`、`Configurations.py:55`、`PhotoIonize.py:100`）根因是 numba / scipy 没有 type stubs。
- **Options Considered:**
  1. 保留 `# type: ignore` + 单行注释说明原因 — **对症修**
  2. 在项目内 `typings/` 写最小 `.pyi` stub 文件 + `pyright stubPath` — **根源修**
  3. 向 numba / scipy 上游贡献 stubs — 理想但 out of scope（工作量大、周期长）
- **Decision:** 2
- **Rationale:** 根源是 pyright 找不到符号类型信息。项目内 stub 直接解决这个根源，无副作用（不影响运行时，`.pyi` 只给类型检查器看）。与方案 1 相比，清零了 ignore 数量；与方案 3 相比，立即可行。
- **Consequences:** 
  - `typings/` 成为项目新目录，后续遇到新的 numba/scipy API 可扩充
  - pyright 配置增加 `stubPath = "typings"`
  - stub 写错可能导致误报或漏报 —— 缓解：最小化原则，只写用到的符号

### Decision 5: Plotting.py:69 elif 结构重构，**严禁** `assert`

- **Context:** `elif points is None` 分支里 `points` 在 `axis == "x"/"y"` 子分支赋值，否则不赋值，外层 `points[points >= 0]` 触发 pyright Optional 窄化失败。
- **Decision:** 重构 elif 加 `else: raise ValueError(...)`，使 `points` 在所有可达子分支被赋值
- **Rationale:** 根源是控制流里有未处理的 axis 值（虽然调用方可能从来不传第三值，但类型上是未证明的）。`raise` 同时表达运行期契约和类型窄化。
- **Consequences:** 对非 "x"/"y" 的 axis 调用会抛错 —— 事实上现状就是静默漏过（`points` 未赋值，后面 `points[points >= 0]` 可能用到上一次赋值的残留），根源修反而修了一个隐藏 bug。

### Decision 6: 分阶段提交

- **Context:** 一次性大改风险大，难以 bisect。
- **Decision:** 五个 Phase 各自为独立 commit，每个 commit 过 pyright + 回归测试。
- **Consequences:** PR 最终包含 ≥ 5 个 commit，可按 phase 回滚。

---

## 6. Precautions

### 技术风险

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| numba 对 `type` 语句别名解析失败 | Low | High | Phase 1 立即跑 `test_reg_e2e_SE.py` |
| `Grotrian.__init__` 结构上无法保证 `fig` 非 None | Med | Low | **不退回 assert**；开 follow-up issue 重构，该 Phase 先保留 ignore（例外：明确标注 follow-up link） |
| `SE_Container` 构造器缺参是真逻辑 bug | Med | Med | 读 `_Container.py` 确认；若是真 bug，开 follow-up issue；不保留 ignore |
| `typings/` 自写 stub 覆盖不全导致 pyright 新报错 | Med | Low | 最小化原则，只写用到的符号；遇到新报错补 stub 或开 follow-up |
| `typings/` stub 与 numba / scipy 实际 API 不一致 | Low | Med | stub 只影响类型检查，不影响运行；若现有回归测试通过即说明运行时无退化 |
| Experimental 模块签名改动破坏某 notebook | Low | Low | 该模块无回归测试，改动后全局 grep 调用点 |
| 删 ignore 后 pyright 报新错 | Med | Low | 每步增量提交，保留可回退点 |

### 回退方案

若某 Phase 出现无法解决的 pyright / 回归失败：

1. `git revert <phase-commit>` 回到上一 Phase 出口
2. 在 task.md 的 Open Questions 里记录原因
3. 该 Phase 降级：要么保留 ignore + 加注释，要么 open follow-up issue

### 迁移注意

- [ ] 向后兼容：`Types.py` 的公共名字全部保留；只换实现方式。下游 `from spectra.Types import T_VEC_FA` 不受影响。
- [ ] Feature flag：无。
- [ ] 迁移脚本：无。

### 性能考虑

- Python 层类型别名对运行时几乎无影响（`TypeAliasType` 实例创建一次）
- `assert` 在 `-O` 优化模式下被剥离，正常运行不受影响
- numba JIT 不受影响（验证见 R2 缓解）

### 安全考虑

- 无外部输入处理变更，不涉及注入/认证

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-04-19 | kouui | Initial draft (基于 tmp/handoff.md 和 commit a4ed0d9 上下文) |
| 2026-04-19 | kouui | 按"能根源修就不 ignore"原则重构：新增 Phase 4 写 `typings/` stubs 根治 A 类 5 处；Phase 3 去除 `assert` 退路；Acceptance 从 ≥20/27 改为 27/27 全消；新增 Decision 4/5 |
