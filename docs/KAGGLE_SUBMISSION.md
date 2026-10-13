# Kaggle 提交说明

核对日期：2026-10-13。当前代码检查点不是已经提交的比赛版本。

## 当前状态

- 先前源码检查点为 `7dc34b9`；此后增加了提交工具。完整单元测试和 Ruff 检查通过。
- ls20 参考后端本地验证 7/7、637 动作；ka59 实验入口本地验证 2/7、43 动作。这些不是 Kaggle 隐藏集成绩。
- 官方 Starter 已下载到 `.external/kaggle-starter`，revision `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`。
- Kaggle 用户名 `hongzhengsong` 已配置，本机尚未配置 Kaggle Token。尚未上传 Notebook，也没有正式分数。
- 离线预览包已生成到 `.external/kaggle-starter/notebooks/submission.ipynb`，以 54 个显式源码文件嵌入，CPU/无网络。比赛 SDK `arc-agi==0.9.9` 下本地 ls20 7/7、637 动作；其他游戏不具备相同成绩保证。正式 gateway 尚未验证。
- 参考后端继续由 `scripts/setup_ls20_solver_backend.sh` 固定 revision 安装。其仓库在当前 pinned revision 没有发现 LICENSE 或明确的再分发授权，因此**不可直接上传当前含该源码的 Notebook**。必须先取得权利人许可，或将其移除改用有授权的实现；`scripts/kaggle_submit.py push` 默认会阻止上传。
- ka59 的实验依赖本地开发轨迹，尚未成为独立比赛 Agent。运行数据 `sampling_runs/` 和 `recordings/` 保留在本机，不纳入此次 Git 提交；新 checkout 不自动包含这些数据。

## 正式提交路径

来源：https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter

1. 登录 Kaggle，加入比赛并接受规则：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules 。
2. 在 Kaggle Settings 生成 API token，保存在 Starter 的 `.kaggle/access_token`，文件权限设为 600。不要发送到聊天或提交到 Git。
3. 修改 Starter 的 `notebooks/kernel-metadata.json`，将 `REPLACE_WITH_YOUR_USERNAME` 换成 Kaggle 用户名。
4. 取得第三方参考后端源码的再分发许可，或替换为自有且合规的实现。构建入口 `uv run python scripts/build_kaggle_submission.py`，会覆盖 Starter 的 `notebooks/submission.ipynb`。
5. 检查包内策略和依赖。比赛禁用互联网，不能在执行时访问外部 LLM API 或从 GitHub 下载后端。构建包的本地验证入口为 `scripts/kaggle_runner.py`；ls20 已用打包源码和 SDK 0.9.9 验证 7/7、637 动作。
6. 权利明确且本地校验通过后，在本项目根目录运行 `uv run python scripts/kaggle_submit.py push --acknowledge-backend-rights`；随后运行 `uv run python scripts/kaggle_submit.py status`。上传完成不等于已获得比赛成绩。**不要运行 Starter 的 `make submit` 或 `make notebook`**，那会覆盖当前 Notebook 为随机 Agent。
7. 状态 complete 后，打开对应 Kaggle Notebook，选择 “Submit to Competition”，输出文件选择 `submission.parquet`。
8. 到比赛的 My Submissions 查看隐藏游戏评测状态、分数和错误日志。

Starter 在普通 Save & Run All 模式下生成占位 parquet；真正比赛重跑时，gateway 记录 Agent 动作并生成实际输出。不要把占位文件当作真实测试结果。

官方 Starter 当前说明每日 5 次正式提交；具体限制以 Kaggle 比赛页面为准。

Starter 虚拟环境可用 `uv venv --python 3.12 .external/kaggle-starter/.venv` 和 `uv pip install --python .external/kaggle-starter/.venv/bin/python 'arc-agi>=0.9.6' 'kaggle>=2.2' python-dotenv` 安装；不要为本地接口验证下载 pandas/pyarrow 等额外大包。Kaggle Notebook 的占位 parquet 使用 Kaggle 运行环境预装的 pandas/pyarrow。

## 代码检查点与数据

Git 提交用于保存源码、配置模板、测试和报告，不会自动上传到 Kaggle。当前 origin 是官方上游仓库，未向它推送。

历史复现入口和数据位置见 `COMPETITION_EXECUTION_PLAN.md`、`EXPERIMENT_REPORT_20261008.md`、`EXPERIMENT_REPORT_20261008_KA59.md`。原始轨迹保留在本地；需要迁移机器时应单独备份，并明确哪些轨迹属于训练输入。
