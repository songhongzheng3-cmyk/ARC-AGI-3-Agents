# Kaggle 提交说明

核对日期：2026-10-16。新增诊断已上传为 Notebook v2，正式提交 `57251615`；提交后首次查询状态为 `PENDING`，暂无分数。

2026-10-19 本地更新：仅自有源码的 `StructuredAgent` 在动态接触和 `OptionCompiler` 都开启时，公开 ls20 同预算 700 步已 **5/7**，HTTP 复验一致；同包关闭编译器为 4/7。第五关通过，第六/七关尚未通过；不代表官方成绩改善，未上传。两个开关默认仍关闭，普通重建不会自动启用实验环境变量。已验证 notebook 单独冻结在 `sampling_runs/round12_option_compiler/frozen_online_v1/submission.ipynb`，未覆盖先前待提交 Notebook。证据与配置见 [OptionCompiler 实验报告](EXPERIMENT_REPORT_20261019_OPTION_COMPILER.md)。

随后新增默认关闭的 `STRUCTURED_ENTITY_TRACKING` 对象跟踪实验。同一新包开启/关闭仍为 5/7，开启组 HTTP 复验一致，全部 700 步动作及画面与旧版一致，没有提高通关数。新包位于 `sampling_runs/round13_level6_identity/frozen_entity_v1/`，未替换上述基线或待提交文件。详见 [第六关对象身份报告](EXPERIMENT_REPORT_20261019_LEVEL6_ENTITIES.md)。

再后新增默认关闭的被动接触效果学习 `STRUCTURED_CONTACT_EFFECTS`，仅在动态接触/实体表示开启时学习。最终 `sampling_runs/round14_contact_effects/frozen_effects_v2/` 同包开启/关闭及开启组 HTTP 仍 5/7，轨迹不变；已验证的旋转/调色局部实验尚未进入最终动作选择。未替换待提交 notebook，未上传；证据见 [接触效果实验报告](EXPERIMENT_REPORT_20261019_CONTACT_EFFECTS.md)。

本地优化最新状态：自写 `StructuredAgent` 包已更新为 17 个文件，公开 ls20 在 700 动作下完成 4/7；离线与 SDK HTTP 轨迹一致，相同包关闭联合搜索为 3/7。最新包尚未上传或正式提交；冻结 notebook、包哈希和证据见 [联合导航实验报告](EXPERIMENT_REPORT_20261016_JOINT_NAVIGATION.md)。下文旧版本的 2/7 和点击 RESET 结果属于历史诊断。

## 当前状态

- 先前源码检查点为 `7dc34b9`；此后增加了提交工具。完整单元测试和 Ruff 检查通过。
- ls20 参考后端本地验证 7/7、637 动作；ka59 实验入口本地验证 2/7、43 动作。这些不是 Kaggle 隐藏集成绩。
- 官方 Starter 已下载到 `.external/kaggle-starter`，revision `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`。
- Kaggle 用户名 `hongzhengsong` 已配置。Token 保存在本机 `~/.kaggle/access_token`（权限 600），不会打包。2026-10-13 已上传仅含自有代码的 Notebook v1；正式提交编号 `57131611`，最终状态 `COMPLETE`、公开分数 `0.00`。取得的 Notebook 日志仅为 Phase A，无法据此确定正式评测每个游戏的表现。
- 2026-10-16 10:21（北京时间）已将含持久诊断的自有代码 Notebook v2 正式提交，编号 `57251615`，首次查询 `PENDING`。v2 的 Phase A 已 `COMPLETE`，远端源码包 SHA256 与本地验证包一致；运行 Python 3.13.15、arc-agi 0.9.8、arcengine 0.9.3。已下载 Phase A 产物到 `sampling_runs/kaggle_submission_v2_phase_a/`。今日 1 次正式提交额度已使用；当前分数尚未产生。
- 离线预览包已生成到 `.external/kaggle-starter/notebooks/submission.ipynb`，以 54 个显式源码文件嵌入，CPU/无网络。比赛 SDK `arc-agi==0.9.9` 下本地 ls20 7/7、637 动作；其他游戏不具备相同成绩保证。正式 gateway 尚未验证。
- 参考后端继续由 `scripts/setup_ls20_solver_backend.sh` 固定 revision 安装。其仓库在当前 pinned revision 没有发现 LICENSE 或明确的再分发授权，因此**不可直接上传当前含该源码的 Notebook**。必须先取得权利人许可，或将其移除改用有授权的实现；`scripts/kaggle_submit.py push` 默认会阻止上传。
- 已用 `uv run python scripts/build_kaggle_submission.py --owned-only` 生成 `.external/kaggle-starter/notebooks-owned/submission.ipynb`，所有游戏均运行自有 `StructuredAgent`。包内 14 个文件，无参考后端。打包源码在 SDK 0.9.9 下对 ls20 完成 5 步短测，无报错；700 步完整预算完成 2/7 后停在第三关。不得将参考后端 7/7 当作这版提交的成绩。
- ka59 的实验依赖本地开发轨迹，尚未成为独立比赛 Agent。运行数据 `sampling_runs/` 和 `recordings/` 保留在本机，不纳入此次 Git 提交；新 checkout 不自动包含这些数据。

## 正式提交路径

来源：https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter

1. 登录 Kaggle，加入比赛并接受规则：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules 。
2. 在 Kaggle Settings 生成 API token，保存在 Starter 的 `.kaggle/access_token` 或本机 `~/.kaggle/access_token`，文件权限设为 600。不要发送到聊天或提交到 Git。
3. 修改 Starter 的 `notebooks/kernel-metadata.json`，将 `REPLACE_WITH_YOUR_USERNAME` 换成 Kaggle 用户名。
4. 取得第三方参考后端源码的再分发许可，或替换为自有且合规的实现。构建入口 `uv run python scripts/build_kaggle_submission.py`，会覆盖 Starter 的 `notebooks/submission.ipynb`。
5. 检查包内策略和依赖。比赛禁用互联网，不能在执行时访问外部 LLM API 或从 GitHub 下载后端。构建包的本地验证入口为 `scripts/kaggle_runner.py`；ls20 已用打包源码和 SDK 0.9.9 验证 7/7、637 动作。
6. 权利明确且本地校验通过后，在本项目根目录运行 `uv run python scripts/kaggle_submit.py push --acknowledge-backend-rights`；随后运行 `uv run python scripts/kaggle_submit.py status`。上传完成不等于已获得比赛成绩。**不要运行 Starter 的 `make submit` 或 `make notebook`**，那会覆盖当前 Notebook 为随机 Agent。

仅自有代码的提交路径：`uv run python scripts/build_kaggle_submission.py --owned-only`，然后 `uv run python scripts/kaggle_submit.py push --owned-only`，再用 `uv run python scripts/kaggle_submit.py status --owned-only` 检查 Phase A。它与含参考后端的本地预览包分开存放；预期策略成绩不能套用参考后端结果。Phase A 完成后仍须在 Kaggle Notebook 页面选择 **Submit to Competition** 才会触发正式 rerun 和排行榜评分。

CLI 也支持正式提交 Notebook 版本：`kaggle competitions submit arc-prize-2026-arc-agi-3 -k hongzhengsong/arc-prize-2026-arc-agi-3-starter -v 1 -f submission.parquet -m 'Owned-code StructuredAgent baseline v1'`。官方比赛规则的每日上限是 **1 次**（Starter README 所说的 5 次不适用于当前规则）。2026-10-13 已使用当天名额；检查结果使用 `kaggle competitions submissions -c arc-prize-2026-arc-agi-3`，不要重复提交相同版本。
7. 状态 complete 后，打开对应 Kaggle Notebook，选择 “Submit to Competition”，输出文件选择 `submission.parquet`。
8. 到比赛的 My Submissions 查看隐藏游戏评测状态、分数和错误日志。

Starter 在普通 Save & Run All 模式下生成占位 parquet；真正比赛重跑时，gateway 记录 Agent 动作并生成实际输出。不要把占位文件当作真实测试结果。

每日限额以比赛规则为准；2026-10-13 核实为 1 次，不采用 Starter README 中的 5 次。

Starter 虚拟环境可用 `uv venv --python 3.12 .external/kaggle-starter/.venv` 和 `uv pip install --python .external/kaggle-starter/.venv/bin/python 'arc-agi>=0.9.6' 'kaggle>=2.2' python-dotenv` 安装；不要为本地接口验证下载 pandas/pyarrow 等额外大包。Kaggle Notebook 的占位 parquet 使用 Kaggle 运行环境预装的 pandas/pyarrow。

## 评测诊断与本地网关验证

生成的 Notebook 将诊断保存到 `/kaggle/working/arc_eval/`。Phase A 只验证导入并生成占位 parquet；Phase B 等待 gateway 后运行同一个打包 Agent。逐游戏完成和总运行完成都会输出到 Notebook 日志，文件能否下载仍取决于 Kaggle 是否开放该次正式 rerun 的产物，不能将普通 Notebook 的 Phase A 日志视为隐藏评测日志。

| 文件 | 用途 |
| --- | --- |
| `installation.json`、`install.log` | Python 版本、离线依赖安装阶段和失败信息 |
| `bundle.json` | 本次运行的源码包 SHA256 |
| `runtime.json` | Phase A/B、等待网关/运行 Agent 阶段、子进程退出码和错误 |
| `report.json` | Python/SDK 版本、游戏列表、每个游戏状态、动作数、完成关数、动作分布、重复画面数、缺失响应及异常 |
| `events.jsonl` | 初始画面、每次动作及坐标、执行前后状态/完成关数/总关数/允许动作/画面摘要，包含失败重试 |
| `runner.log` | Phase B 中 SDK 与 Agent 的完整标准输出和异常栈 |
| `recordings/` | 已返回的完整逐步画面和 action_input，保留原有 Recorder 格式 |
| `scorecard.json` | SDK 关闭评分卡后的结果，移除认证字段；本地结果不代表官方分数 |

报告在 SDK 导入前创建，在游戏启动、每次请求前后及异常时更新。游戏中途失败也会保留已经完成的动作与关卡，继续后续游戏；总体返回非零退出码。缺失反馈的重试请求数与 Agent 的动作计数分别记录。进程被平台强杀时，已有报告可能仍显示 `running`，应结合最后一条事件定位。

本地验证使用 Notebook 中嵌入的源码包，不从工作区导入 Agent。离线通道直接使用 SDK；网关通道启动 SDK 官方 HTTP 服务，以 competition 模式运行公开缓存游戏，并执行 Notebook 原样的解包与 Phase B 调用。验证会核对 HTTP 错误、实际游戏动作和评分卡关闭回调，并比较两通道的动作/反馈轨迹。

```bash
# 生成自有代码版本，不上传
.venv/bin/python scripts/build_kaggle_submission.py --owned-only

# 使用已有 SDK 0.9.9 环境，验证 ls20 的离线/HTTP 两条路径
.external/kaggle-starter/.venv/bin/python scripts/verify_kaggle_submission.py --game ls20 --budget 700

# 复现此前 Kaggle 日志中的 Python 3.13 / SDK 0.9.8 主版本组合
uv venv --python 3.13 .external/kaggle-compat/.venv
uv pip install --python .external/kaggle-compat/.venv/bin/python 'arc-agi==0.9.8' 'arcengine==0.9.3' python-dotenv
.external/kaggle-compat/.venv/bin/python scripts/verify_kaggle_submission.py --budget 700
```

省略 `--game` 会运行全部本地缓存公开游戏；默认同时验证两条路径。`--transport offline` 或 `--transport gateway` 可单独运行；`--budget 5` 可短测。输出在新建的 `sampling_runs/kaggle_verify_<UTC时间>/`；入口打印绝对路径。`verification.json` 中的 `passed` 代表运行链路和诊断完整，并不代表过关，必须另看 `levels_completed`、`state` 和 `stop_reason`。

该验证覆盖 Python/SDK 兼容性与真实 HTTP 请求，不包含 Kaggle 的 Linux 镜像、私有 sidecar、数据集 wheel 安装和隐藏游戏；不能声称完全复现正式评分，也不能反推 v1 零分的确定原因。

### 2026-10-16 实测结果

- Python 3.12.13 / SDK 0.9.9：ls20 离线与 HTTP 网关各执行 700 步，均为 2/7，动作及反馈轨迹完全一致。
- Python 3.13.9 / SDK 0.9.8 / arcengine 0.9.3：本地缓存的 8 个公开游戏，两条路径各游戏 700 步，共 11,200 次 Agent 步进；无运行异常、无 HTTP 错误，网关评分卡成功关闭。两条路径全部动作与反馈轨迹一致。ls20 为 2/7；另外 7 个游戏均为 0 关。
- vc33、lp85、ft09 的两条路径均为 700 次 RESET；自有 StructuredAgent 在缺少方向动作时的 fallback 返回 RESET，当前缺少有效的点击动作探索。这是可复现的策略缺陷，不能仅凭这些公开游戏结果认定正式隐藏集零分原因。
- 完整报告：`sampling_runs/kaggle_verify_20261016T020919492585Z/verification.json`，逐步轨迹在其 `offline/`、`gateway/` 子目录。
- 最终生成包 SHA256：`2b5d05496a28a06e4c4dd048b605ed736a47361a8861780a7dd9844247d25c35`，15 文件；最终包在 Python 3.13 / SDK 0.9.8 下另完成两通道 5 步短测，报告 `sampling_runs/kaggle_verify_20261016T021106091976Z/verification.json`。完整预算对照期间只调整了等待网关的超时检查，未改变 Agent 策略。
- 单元测试 240 项通过，包含新加的 6 项诊断失败场景。新增诊断已构建并上传为 Notebook v2，2026-10-16 正式提交已消耗当日 1 次额度，初始状态 `PENDING`。

## 代码检查点与数据

Git 提交用于保存源码、配置模板、测试和报告，不会自动上传到 Kaggle。当前 origin 是官方上游仓库，未向它推送。

历史复现入口和数据位置见 `COMPETITION_EXECUTION_PLAN.md`、`EXPERIMENT_REPORT_20261008.md`、`EXPERIMENT_REPORT_20261008_KA59.md`。原始轨迹保留在本地；需要迁移机器时应单独备份，并明确哪些轨迹属于训练输入。
