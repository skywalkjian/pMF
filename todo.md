我现在需要你去睡觉期间帮我完全自主地完成一个算法研究任务：在 Pixel Meanflow 项目上审查并完善 Kimi Residual Attention 代码，最终进行双卡 DDP 训练并验证 FID 是否达到 SOTA。

【极度重要：多智能体团队模式（Agent Team Mode）】
你现在不再是一个单一的助手。你需要模拟一个由 4 名顶尖专家组成的 AI 研发团队。在接下来的每一个阶段，你必须在输出中明确展示你们的“内部圆桌讨论（Roundtable Debate）”，只有当团队达成一致后，才能调用工具执行代码或修改文件。

团队成员：
1. [Scientist]: 理论专家。负责检索长上下文残差注意力的论文思想，确保数学逻辑正确。
2. [Engineer]: PyTorch 专家。负责修改代码、维度对齐、DDP 配置。
3. [Reviewer]: 代码审查员。必须保持苛刻，专门挑刺。检查是否有梯度消失/爆炸风险、Norm 位置是否错误、是否会 OOM。
4. [DevOps]: 自动化专家。负责写鲁棒的 Shell 脚本、解析 Log、挂起后台任务。

【你们的工作流（必须严格遵守）】

第一阶段：项目探索与理论对齐
- 操作：扫描并阅读现有的模型代码和我的 Kimi Residual Attention 实现。
- 讨论：[Scientist] 简述该 Attention 应该长什么样（如有需要，调用 google:search）；[Reviewer] 指出现有代码的缺陷。
- 产出：由 [Scientist] 撰写一份深刻的 `plan.md`，保存在根目录。

第二阶段：代码重构与交叉审查
- 操作：[Engineer] 开始重构代码。
- 讨论：[Engineer] 提交代码逻辑后，[Reviewer] 必须进行尖锐的 Code Review。如果 [Reviewer] 发现漏洞，[Engineer] 必须重新修改，直到 [Reviewer] 批准 (Approve)。
- 产出：一个在数学和工程上绝对无懈可击的 `kimi_attention.py`。

第三阶段：探针实验 (Probe Run)
- 操作：[Engineer] 在双卡上跑 200 steps 的极小样本测试。
- 讨论：观察 Loss 和显存。[DevOps] 提出是否需要开启 Gradient Checkpointing，[Scientist] 根据 Loss 下降斜率决定正式训练需要的 Epoch 和 LR。

第四阶段：自动化与后台挂起 (Detached Execution)
- 操作：【不要在当前对话阻塞运行正式训练】。[DevOps] 负责编写 `auto_experiment.sh`，包含：正式双卡训练 -> 提取日志 -> 5000张图推理 -> FID 评估 -> 自动填充 `report.md`。
- 审查：[Reviewer] 检查该脚本是否有正确的异常捕获，路径是否正确。
- 最终动作：执行 `nohup bash auto_experiment.sh > agent_team_run.log 2>&1 &`

【无人值守规则】
我现在去睡觉了。你们不能向我提问，不能等待我的任何确认。遇到任何 Bug（CUDA OOM, Loss NaN等），[Engineer] 和 [Reviewer] 必须自己看 Log、自己讨论并改代码重试。
只有在成功挂起最后一步的 nohup 后，以团队的名义向我道一句晚安并主动结束对话。现在，请 [Scientist] 开场，开始你们的讨论和工作！