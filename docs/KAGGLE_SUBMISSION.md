# Kaggle 提交说明

核对日期：2026-10-13。当前代码检查点不是已经提交的比赛版本。

## 当前状态

- 本项目完整单元测试 229 项通过，Ruff 检查通过。
- ls20 参考后端本地验证 7/7、637 动作；ka59 实验入口本地验证 2/7、43 动作。这些不是 Kaggle 隐藏集成绩。
- 官方 Starter 已下载到 `.external/kaggle-starter`，revision `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`。
- 本机尚未配置 Kaggle 凭证。尚未上传 Notebook，也没有正式分数。
- 官方 Starter 默认包含随机 Agent，不能直接用它代表本项目效果。现有多文件策略、参考后端及依赖仍需打包接入，再验证比赛 SDK 和 gateway 生命周期。
- 参考后端继续由 `scripts/setup_ls20_solver_backend.sh` 固定 revision 安装；打包前需要核对其许可和比赛外部代码规则。
- ka59 的实验依赖本地开发轨迹，尚未成为独立比赛 Agent。运行数据 `sampling_runs/` 和 `recordings/` 保留在本机，不纳入此次 Git 提交；新 checkout 不自动包含这些数据。

## 正式提交路径

来源：https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter

1. 登录 Kaggle，加入比赛并接受规则：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules 。
2. 在 Kaggle Settings 生成 API token，保存在 Starter 的 `.kaggle/access_token`，文件权限设为 600。不要发送到聊天或提交到 Git。
3. 修改 Starter 的 `notebooks/kernel-metadata.json`，将 `REPLACE_WITH_YOUR_USERNAME` 换成 Kaggle 用户名。
4. 完成现有 Agent 的打包适配，在独立 Starter 环境中执行 `make setup`，再做本地接口验证。不要在原项目环境中盲目升级依赖，也不要默认运行留出游戏。
5. `make notebook` 生成 Notebook，检查包含的是目标策略及其完整依赖。比赛禁用互联网，不能在执行时访问外部 LLM API 或临时从 GitHub 下载后端。
6. `make submit` 上传并启动 Notebook；`make status` 查看运行状态。这一步完成不等于已获得比赛成绩。
7. 状态 complete 后，打开对应 Kaggle Notebook，选择 “Submit to Competition”，输出文件选择 `submission.parquet`。
8. 到比赛的 My Submissions 查看隐藏游戏评测状态、分数和错误日志。

Starter 在普通 Save & Run All 模式下生成占位 parquet；真正比赛重跑时，gateway 记录 Agent 动作并生成实际输出。不要把占位文件当作真实测试结果。

官方 Starter 当前说明每日 5 次正式提交；具体限制以 Kaggle 比赛页面为准。

## 代码检查点与数据

Git 提交用于保存源码、配置模板、测试和报告，不会自动上传到 Kaggle。当前 origin 是官方上游仓库，未向它推送。

历史复现入口和数据位置见 `COMPETITION_EXECUTION_PLAN.md`、`EXPERIMENT_REPORT_20261008.md`、`EXPERIMENT_REPORT_20261008_KA59.md`。原始轨迹保留在本地；需要迁移机器时应单独备份，并明确哪些轨迹属于训练输入。
