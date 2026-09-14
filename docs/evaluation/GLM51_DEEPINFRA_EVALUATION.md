# GLM 5.1 DeepInfra 评测配置

此配置将 `zai-org/GLM-5.1` 作为被测模型，默认使用 `zai-org/GLM-5.3-Flash` 作为 Judge。Judge 模型和两个端点都可以在复制后的配置中更换，但每次更换必须新建运行目录并记录 provider、模型 ID、日期标签和采样参数。

模板：`configs/evaluation/roleplay_benchmarks/glm51-rmt-deepinfra.example.json`。

两者都显式关闭 GLM 思考模式：`chat_template_kwargs.enable_thinking=false`。这会让被测模型直接生成角色回复，也让 Judge 直接输出评分 JSON，减少无关 reasoning token 和截断风险。它不是关闭模型能力或修改评分规则。

## 使用

```bash
cp configs/evaluation/roleplay_benchmarks/glm51-rmt-deepinfra.example.json .local/glm51-rmt.json
export ACTOR_BASE_URL=https://api.deepinfra.com/v1
export ACTOR_API_KEY=YOUR_DEEPINFRA_KEY
export JUDGE_BASE_URL=https://api.deepinfra.com/v1
export JUDGE_API_KEY=YOUR_DEEPINFRA_KEY
python scripts/evaluate_rmtbench.py prepare --config .local/glm51-rmt.json --run-dir outputs/glm51-rmt-20260914
python scripts/evaluate_rmtbench.py run --run-dir outputs/glm51-rmt-20260914
```

先用 `limit_cases` 做 dry-run 和小样本，确认回复没有 `finish_reason=length`、Judge 能解析 `Score` 后，再扩大范围。该模板不承诺 DeepInfra 提供不可变权重 revision；`revision` 字段必须在实验记录中写明实际 provider 返回身份或日期标签。

## 更换 Judge

把 `roles.judge.model`、端点环境变量和 `revision` 一起改掉即可。若新 Judge 不支持 `chat_template_kwargs`，删除该扩展字段，并记录“思考模式由服务端默认值控制”；不要把 GLM 专用参数硬塞给其他模型。
