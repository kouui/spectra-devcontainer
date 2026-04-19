# Plan: pyright cleanup follow-up

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-04-19
> **Target Completion:** 2026-04-26

---

## 0. Context

> **Objective:** 用 Python 3.13 `type` 语句重写 `Types.py` 消除 `reportInvalidTypeForm` 全局抑制；并把上轮 pyright 修复新增的 27 处 `# type: ignore` 能替换则替换、是 bug 则就地修掉、库限制则保留并加注释。
> **Full spec:** [task.md](./task.md)
>
> 本 plan 描述三阶段执行顺序：**先收掉 Types.py 大头 → 再灭真 bug（C 类）→ 最后清样式（B 类）**。

---

## 1. Overall Architecture

### 系统总览

```
┌─────────────────────┐
│  Types.py (source)  │  ← Phase 1: type 语句重写
└──────────┬──────────┘
           │ re-exports via ImportAll.py
           ▼
┌─────────────────────┐     ┌──────────────────────┐
│ 全包 src/spectra/**  │────▶│  pyright --strict     │  ← Phase 1 出口：0 错误
│ 使用 T_VEC_FA 等别名  │     │  reportInvalidTypeForm│
└──────────┬──────────┘     │  抑制已移除           │
           │                └──────────────────────┘
           ▼
┌─────────────────────┐
│  回归测试 247 passed  │  ← 每 Phase 出口必须绿
│  含 JIT 路径         │
└─────────────────────┘
```

### 关键组件

| Component | Responsibility | New / Modified |
|-----------|---------------|----------------|
| `src/spectra/Types.py` | 类型别名源头 | Modified |
| `pyproject.toml [tool.pyright]` | 去掉 `reportInvalidTypeForm = false` | Modified |
| `Visual/Grotrian.py` | 构造器保证 `self.fig` 非 None | Modified (结构小调整) |
| `Experimental/ExScatter.py` | 函数签名 `T_FLOAT → T_VEC_FA` | Modified (签名 bug) |
| `Util/AtomicDataUtils/MakePhotoioniz.py` | `_v1_fit_func_` 签名修正 | Modified (签名 bug) |
| `Util/AtomUtils/AtomIO.py` | 显式 `int()` 转换 | Modified |
| `Util/HelpUtil.py` | `dtype.names is None` 防御 | Modified |
| `RadiativeTransfer/Profile.py` | `.real`/`.imag` → `numpy.real`/`imag` | Modified |
| `Visual/Plotting.py` | 修 elif 收窄 + `add_axes` 签名 | Modified |
| `Function/Icp/SELib.py` | 确认 `SE_Container` 构造 | Modified / 待确认 |
| `Function/Hydrogen/DegenerateN.py` | `PI_cross_section_` overload | Modified / 待确认 |

### 数据流（主流程：改别名 → 类型检查 → 测试）

1. 重写 `Types.py`（`type T_VEC_FA = ...`）
2. `pyright` 验证仍为 0 错误
3. 跑回归测试（含 JIT 路径）
4. 逐个消除 ignore，每次 pyright + 测试
5. 最终跑 `pre-commit run --all-files`

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

### Phase 2: C 类真 bug 修复（9 处）

> **Goal:** 修掉 ignore 掩盖的真实类型 bug
> **Estimated Effort:** 1 day
> **Depends on:** Phase 1

- [ ] **Step 2.1** — `Experimental/ExScatter.py`
  - 函数 `HI_bf_emissivity_LTE_` 签名 `nHI_pop_LTE: T_FLOAT` 改为 `T_VEC_FA`
  - 删除 `:30` 和 `:38` 两处 ignore
  - Experimental 模块优先级低，但改动极小
- [ ] **Step 2.2** — `Util/AtomicDataUtils/MakePhotoioniz.py:115`
  - `_v1_fit_func_` 签名 `wave: T_FLOAT` 改为 `T_VEC_FA`
  - 删除 ignore
- [ ] **Step 2.3** — `Util/AtomUtils/AtomIO.py:1035-1037`
  - 显式 `z = int(Level["stage"][Coe["idxI"][i]])` 将 `numpy.int64` 转为 `int`
  - 删除两处 ignore
- [ ] **Step 2.4** — `Util/HelpUtil.py:81`
  - 在 `for field_name in dtype.names:` 之前加：
    ```python
    if dtype.names is None:
        return  # 或 raise，视上下文
    ```
  - 删除 ignore
- [ ] **Step 2.5** — `Function/Icp/SELib.py:477`
  - 读 `_Container.SE_Container` 的定义，确认 `Ntotal/Nh/Ne/Te` 是否有默认值
  - 若有默认值：删除 ignore；若无：补齐参数或改签名（**此处可能触发真逻辑修改，需再讨论**）
- [ ] **Step 2.6** — `Function/Hydrogen/DegenerateN.py:51`
  - 读 `_Hydrogen.PI_cross_section_` 的 overload 列表
  - 若 `Eratio` 类型不在现有 overload 内：补第 3 个 overload
  - 若能用 `int(ni)` 等简单转换收敛：优先用转换
- [ ] **Step 2.7** — 每步改完跑 pyright + 回归测试。

**Phase 2 Exit Criteria:**
- [ ] 9 处 C 类 ignore 至少消除 7 处（SELib 的 2.5 可能需独立讨论）
- [ ] pyright 0 错误
- [ ] 回归 247 passed

---

### Phase 3: B 类样式替换（11 处）

> **Goal:** 用断言/转换/签名替换 ignore
> **Estimated Effort:** 0.5 day
> **Depends on:** Phase 2

- [ ] **Step 3.1** — `Visual/Grotrian.py` 7 处 `self.fig` ignore
  - 方案：让 `__init__` 末尾保证 `self.fig` 已创建并且非 None（可能需要调整构造顺序）
  - 退化方案：每个方法入口加 `assert self.fig is not None`
  - 删除 7 处 `union-attr` ignore
- [ ] **Step 3.2** — `Visual/Plotting.py:69`
  - `elif points is None:` 分支末尾 `points = points[points >= 0]` 前，加 `assert points is not None`（或改 elif 结构让 `points` 在两个子分支都被赋值）
  - 删除 ignore
- [ ] **Step 3.3** — `Visual/Plotting.py:136`
  - 调用方 `axe_kw = {"ax1": [0, 0, 1, 1]}` 改为 `"ax1": (0.0, 0.0, 1.0, 1.0)`，并改签名 `axe_kw: dict[str, tuple[float,float,float,float]]`
  - 或在函数内 `tuple(val)` 之前做长度断言：`assert len(val) == 4; rect = (val[0], val[1], val[2], val[3])`
  - 删除 ignore
- [ ] **Step 3.4** — `RadiativeTransfer/Profile.py:248-249`
  - `w4.real` → `_numpy.real(w4)`；`w4.imag` → `_numpy.imag(w4)`
  - 删除两处 ignore
- [ ] **Step 3.5** — 每步改完跑 pyright + 回归测试。
- [ ] **Step 3.6** — 手动跑一个用 Grotrian 的 notebook（如 `notebooks/` 下的示例）确保绘图未破坏。

**Phase 3 Exit Criteria:**
- [ ] B 类 11 处 ignore 至少消除 10 处
- [ ] Grotrian 手动验证 OK
- [ ] pyright 0 错误
- [ ] 回归 247 passed

---

### Phase 4: 保留条目注释 + 收尾

> **Goal:** A 类 5 处保留的 ignore 补注释；全仓 `pre-commit` 绿；汇总报告
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 3

- [ ] **Step 4.1** — 为保留的 ignore 加单行注释说明原因：
  - `ImportAll.py:27-32`：`# numba.typed.List 无 stubs`
  - `Configurations.py:55`：`# numba.core.config 无 stubs`
  - `Atomic/PhotoIonize.py:100`：`# scipy stubs 声明 fill_value: float，但运行时支持 tuple`
- [ ] **Step 4.2** — 跑 `uv run --extra dev pre-commit run --all-files`。
- [ ] **Step 4.3** — 统计最终消除 / 保留的 ignore 数，更新 `task.md` 的 Acceptance Criteria 勾选。
- [ ] **Step 4.4** — 准备 PR 描述（如果用户要开 PR）。

**Phase 4 Exit Criteria:**
- [ ] `pre-commit run --all-files` 全绿
- [ ] ignore 消除率 ≥ 20/27（74%）
- [ ] task.md 状态置为 Review

---

## 3. Boundaries — Do NOT Touch

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| RH 转换脚本 | `src/spectra/Util/AtomicDataUtils/RH2Spectra/01makeAtom.py` | 文件名以数字开头无法 import；使用独立 dataclass；协作者维护 |
| 历史 ignore | `src/spectra/Atomic/ContinuumOpacity.py`、`src/spectra/Atomic/Hydrogen.py`、`src/spectra/Warnings.py` 中 `commit 44a2bfa` 之前的 ignore | 不在本次 27 处范围内 |
| numba 类型表达式历史注释 | `src/spectra/Types.py:110-148`（所有 `# import numba.types` 等注释块） | 保留为文档 |
| 业务逻辑 | 任何非类型层面的修改 | 本次只做类型清理 |

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

### Decision 3: Grotrian.py 的 `self.fig` 用构造器保证非 None，不每方法 assert

- **Context:** 7 处 `union-attr` ignore 全因 `self.fig: Optional[Figure]`。
- **Options Considered:**
  1. 每方法入口 `assert self.fig is not None` — 7 处模板代码
  2. 构造器末尾统一保证 `self.fig` 初始化完成，类型声明为非 Optional — 单点修复
- **Decision:** 优先试 2；若结构上无法保证（如延迟初始化模式），退回 1。
- **Rationale:** 根治 > 对症；但不能破坏绘图流程。
- **Consequences:** 需读 `Grotrian.__init__` 确认是否有条件分支不创建 `fig`。

### Decision 4: 分阶段提交

- **Context:** 一次性大改风险大，难以 bisect。
- **Decision:** 四个 Phase 各自为独立 commit，每个 commit 过 pyright + 回归测试。
- **Consequences:** PR 最终包含 ≥ 4 个 commit，可按 phase 回滚。

---

## 6. Precautions

### 技术风险

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| numba 对 `type` 语句别名解析失败 | Low | High | Phase 1 立即跑 `test_reg_e2e_SE.py` |
| `Grotrian.__init__` 结构上无法保证 `fig` 非 None | Med | Low | 退回每方法 `assert` 方案 |
| `SE_Container` 构造器缺参是真 bug | Med | Med | 读 `_Container.py` 确认；若是真 bug 提出单独讨论 |
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
