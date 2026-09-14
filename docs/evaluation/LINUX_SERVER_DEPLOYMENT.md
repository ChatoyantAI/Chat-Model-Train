# Linux 评测服务器部署与指定 GPU

本文是现有四套评测入口的部署说明，不改变提示词、数据版本或评分协议。各套配置与结果口径见 [运行手册](ROLEPLAY_BENCHMARK_RUNBOOK.md)。

## 1. 拉取 main，代码与环境分开

需要 Linux、Python 3.11+、Git。GPU 推理还需要匹配的 NVIDIA 驱动。以下路径是示例，可整体替换为自己的持久化磁盘路径。

仓库为私有仓库：先使用自己的 GitHub SSH key、deploy key 或凭据管理器获得读取权限。不要将 Token 拼进 clone URL。机器登录 SSH key 与 GitHub 仓库访问授权是两回事。

```bash
export EVAL_ROOT=/workspace
mkdir -p "$EVAL_ROOT/projects" "$EVAL_ROOT/envs" "$EVAL_ROOT/models" \
  "$EVAL_ROOT/runs/roleplay-evaluation" "$EVAL_ROOT/cache/pip" "$EVAL_ROOT/tmp"
git clone --branch main --single-branch \
  https://github.com/ChatoyantAI/Chat-Model-Train.git \
  "$EVAL_ROOT/projects/Chat-Model-Train"
cd "$EVAL_ROOT/projects/Chat-Model-Train"
git submodule update --init --recursive
git rev-parse HEAD
```

更新已有部署时，先确认没有评测正在使用这份 checkout，再执行 `git pull --ff-only origin main` 和子模块更新。不要在正在运行的评测目录里直接更换代码或模型。

## 2. CPU 评测环境与数据

RMTBench / MiniMax 的包装只需要 Python 标准库。CoSER 使用旧版 LangChain，单独固定兼容版本，不和训练环境混装。

```bash
python3 -m venv "$EVAL_ROOT/envs/roleplay-eval"
export PIP_CACHE_DIR="$EVAL_ROOT/cache/pip"
export TMPDIR="$EVAL_ROOT/tmp"
EVAL_PY="$EVAL_ROOT/envs/roleplay-eval/bin/python"
"$EVAL_PY" -m pip install \
  -r configs/evaluation/roleplay_benchmarks/coser.requirements.txt \
  langchain==0.2.17 langchain-community==0.2.19
# NLTK 要求私有目录；某些共享挂载的权限会被拒绝。
export NLTK_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/roleplay-nltk"
mkdir -p "$NLTK_DATA"
chmod 700 "$NLTK_DATA"
"$EVAL_PY" -m nltk.downloader -e -d "$NLTK_DATA" punkt punkt_tab

# HF_TOKEN 使用自己的账号权限；可以使用仓库说明中的本地 .env 文件。
# 401/403 表示需要访问授权，脚本不会自动接受数据许可。
"$EVAL_PY" scripts/download_roleplay_benchmarks.py
"$EVAL_PY" scripts/roleplay_benchmarks.py verify all

# 下载并验证 RMT 论文模板，随后才运行依赖完整数据的测试。
"$EVAL_PY" -c 'from src.evaluation.roleplay_benchmarks.paper_templates import load_templates; load_templates()'
"$EVAL_PY" -m unittest discover -s tests/evaluation -p 'test_benchmark_*.py' -v
```

也可迁移团队已经获授权下载的同版本数据目录及 `snapshot.json`，然后重新 `verify all`；不要复制个人凭据。公共数据位于 `data/external/roleplay_benchmarks/`，官方源码位于 `vendor/benchmarks/`，均通过版本清单校验。

## 3. 隔离的模型服务环境

以下锁定 vLLM 0.28.0 的 CUDA 12.9 / x86_64 构建，按照 [官方安装文档](https://docs.vllm.ai/en/v0.28.0/getting_started/installation/gpu/) 使用发布 wheel 和对应 PyTorch 源，避免直接 `pip install vllm` 拉到 CUDA 13 构建。这不是所有驱动和模型均已验证的承诺；旧驱动对 CUDA 小版本兼容有功能限制，实际模型的 JIT 内核仍需单独验证。不自动升级宿主机驱动。

```bash
python3 -m venv "$EVAL_ROOT/envs/roleplay-vllm"
"$EVAL_ROOT/envs/roleplay-vllm/bin/python" -m pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129 accelerate scipy
"$EVAL_ROOT/envs/roleplay-vllm/bin/python" -m pip check
"$EVAL_ROOT/envs/roleplay-vllm/bin/vllm" serve --help
```

安装不会启动模型。不要将框架或模型权重装进系统 Python、根分区或 Git 仓库。

### GPU 必须由运行者选择

先查看 GPU 的编号、UUID 和占用，确认选用的卡已分配给自己：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
```

启动示例（**手动执行才会占卡**）：

```bash
# 下面全部由运行者填写，不是服务器预留卡号。
export EVAL_GPUS=0,2,4,6
export MODEL_PATH=/workspace/models/YOUR_MODEL_SNAPSHOT
export SERVED_MODEL=candidate-your-run
export MODEL_PORT=8188
export MODEL_CONTEXT=65536
export MODEL_MAX_SEQS=8
export HF_HOME="$EVAL_ROOT/cache/huggingface"

# 必填；缺失时停止，避免误用所有卡。
: "${EVAL_GPUS:?Select GPU IDs explicitly}"
: "${MODEL_PATH:?Set a model snapshot path}"
: "${SERVED_MODEL:?Set a unique service model ID}"
: "${MODEL_PORT:?Set an unused port}"
: "${MODEL_CONTEXT:?Set a supported context length}"
: "${MODEL_MAX_SEQS:?Set inference concurrency capacity}"
IFS=',' read -ra SELECTED_GPUS <<< "$EVAL_GPUS"
CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
  "$EVAL_ROOT/envs/roleplay-vllm/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$MODEL_PORT" \
  --tensor-parallel-size "${#SELECTED_GPUS[@]}" \
  --max-model-len "$MODEL_CONTEXT" --max-num-seqs "$MODEL_MAX_SEQS" \
  --gpu-memory-utilization 0.85
```

`0,2,4,6` 表示只使用这四张卡，进程内部重新编号为 0–3；单卡填一个 ID。也可以填写完整 GPU UUID，避免依赖编号。`tensor-parallel-size` 是一个模型分布到几张卡，不是评测请求并发。不是所有模型都支持任意并行数；参数不兼容或显存不足时调整自己的配置，不自动抢占其他卡。

建议先前台确认启动成功，再放入独立 tmux 会话。不同运行者使用不同 GPU、端口、服务模型 ID 和结果目录。本说明没有实现跨用户 GPU 调度/抢占锁，选卡仍需团队协调。

默认仅监听本机，CPU 评测连接 `http://127.0.0.1:端口/v1`。需要外部访问时优先 SSH 隧道；不要直接将无鉴权推理接口暴露公网。

## 4. 连接评测，不自动开始付费任务

模型部署成功后，复制仓库内各套 `.example.json` 到自己的 `.local/` 配置目录。替换模型 ID、不可变 revision、精度、采样参数等占位项；运行目录按模型和实验区分。

```bash
export ACTOR_BASE_URL="http://127.0.0.1:$MODEL_PORT/v1"
export ACTOR_API_KEY=local-service
# 实际裁判地址和 Key 由运行者设置，不能把示例当作可用凭据。
# MiniMax 另需 SIMULATOR_BASE_URL / SIMULATOR_API_KEY。

"$EVAL_PY" scripts/evaluate_rmtbench.py prepare \
  --config .local/rmt-your-run.json \
  --run-dir "$EVAL_ROOT/runs/roleplay-evaluation/rmt-your-run"
"$EVAL_PY" scripts/evaluate_rmtbench.py run \
  --run-dir "$EVAL_ROOT/runs/roleplay-evaluation/rmt-your-run"
# 核对 dry-run 后才加 --execute --max-requests N。
```

四套入口和限制：

| 基准 | 入口 | 运行前必须明确 |
|---|---|---|
| CoSER | `scripts/evaluate_coser.py` | `BENCHMARK_BASE_URL` 必须同时路由 actor、judge、environment、next-speaker；只有 actor 的 vLLM 不够 |
| CharacterBench | `scripts/evaluate_characterbench.py` | 官方生成、提示词、专用 CharacterJudge、聚合四阶段；原始数据的 3 个英文异常会被严格预检阻止 |
| RMTBench | `scripts/evaluate_rmtbench.py` | actor / judge 端点；Claude 3.5 与 4.5 分开；数据布局和采样参数明确配置 |
| MiniMax | `scripts/evaluate_minimax_roleplay.py` | actor / judge / simulator 端点；公开子集和本仓库补充协议，不冒充完整官方榜单 |

CharacterBench 的官方本地裁判需另准备 `CHARACTERJUDGE_PATH` 和对应依赖。上述推理环境同时安装了 `accelerate`、`scipy`；安装其他依赖时也必须保留 CUDA 构建约束，避免被默认 CUDA 13 wheel 替换。此环境方案仍需使用实际 CharacterJudge 权重试跑验证；不能把只做依赖导入当作裁判验证。运行 judge 阶段同样显式加 `CUDA_VISIBLE_DEVICES=运行者选择的卡`，并通过 `--python` 传入该环境的 Python。CPU 包装本身不会替你指定或保留 GPU。

`main` 的通用入口不包含旧服务器实验目录里的全部专用路由和调度脚本。部署仓库不等于迁移正在运行的任务；四套全自动统一调度需要另行实现，不在这里暗中替换评测协议。

## 5. 运行记录与恢复

每次运行保存代码 commit、配置、模型 revision、服务命令、选用 GPU UUID、依赖 `pip freeze` 和数据校验结果。不要保存整个进程环境，避免将密钥写入记录。

RMT/MiniMax 在同一运行目录重复执行可复用成功缓存，失败继续重试；切换模型、采样参数、提示词或数据必须新建目录。`--max-requests` 是每次命令的请求次数上限，不是美元预算。查看 `progress.json`、`summary.json`、`summary.md`；失败不是 0 分。完整命令见运行手册。

首次临时授权 clone 不会给服务器永久 GitHub 权限。后续自行 `git pull` 仍需配置具有仓库只读权限的凭据；不要把用于登录服务器的私钥复制进去。
