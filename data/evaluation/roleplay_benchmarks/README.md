# 固定评测数据

四套评测输入均已放入本目录。总计 26 份上游文件，约 60.4 MiB；不包含模型权重、训练语料或我们的生成/评分缓存。

| 数据集 | 从哪里看 | 内容 |
|---|---|---|
| CoSER | [coser/data/test/test_set.json](coser/data/test/test_set.json) | 200 个测试场景；同时保留 `held_out_plots.json` |
| CharacterBench | [characterbench/eval_data/raw_data/](characterbench/eval_data/raw_data/) | 13 个维度、3,250 条原始记录，中英文共 6,500 个输入；原有 3 条无效源输入保留，由预检单列 |
| RMTBench | [rmtbench/](rmtbench/) | Claude 3.5 / 4.5 两版，各有中英文文件，每个文件 80 个角色；版本分别使用 |
| MiniMax Role-Play Bench | [minimax/data/](minimax/data/) | 45 条英文、50 条中文场景，合计 95 条 |

MiniMax 官方公开的其他模型对话、历史评分以及重复 Parquet 编码不参与新模型的输入生成，因此没有打包；它们的文件名保留在 `sources.lock.json` 的 `reference_files` 中。
没有这些参考输出时，预检会明确标记参考轨迹审查不可用，不会伪报审查通过，也不会影响 95 个场景的生成与评分。

## 获取和运行

```bash
git clone --recurse-submodules --branch main https://github.com/ChatoyantAI/Chat-Model-Train.git
cd Chat-Model-Train
python3 scripts/evaluation/setup_public_evaluation.py
```

输入数据随仓库下载；两个官方代码子模块仍用于提示词、处理函数和评分实现。
安装器按 SHA-256 校验本目录，优先恢复数据到原来的 `vendor/benchmarks/` 或 `data/external/roleplay_benchmarks/` 路径。
HF 数据可从此副本恢复，无需重复下载；Python 依赖、CoSER tokenizer 和 RMT 论文评分模板仍由安装步骤准备。
已有且一致的输入被复用，发现不同内容则停止并报告，不覆盖它们。

完整操作见[统一评测运行说明](../../../docs/evaluation/UNIFIED_PUBLIC_EVALUATION.md)。

## 来源和校验

[manifest.json](manifest.json) 记录每份文件的上游仓库、固定 revision、原始大小和 SHA-256。四套上游 README 原样保留在各自目录。
所有数据文件都与此前已核验的缓存逐字节相同，Git 禁止对此目录做换行转换；输入内容、提示词和评分参数没有因打包而变化。

- [CoSER](https://github.com/Neph0s/CoSER)：`624efc50db7be3a4e37cec6a1c58ed434dc9f2ce`。
- [CharacterBench](https://github.com/thu-coai/CharacterBench)：`66c113cf3c011697d4c232de6beabb7797a9a380`。
- [RMTBench](https://huggingface.co/datasets/xiangh/RMTBENCH)：`cc17f459c84c66051215376ab892deb593343455`。
- [MiniMax](https://huggingface.co/datasets/MiniMaxAI/role-play-bench)：`3c1be2a56afbcaab19ae6b40b8a24429eae792f5`。

本目录的公开上传已由仓库维护者确认。上游署名与使用条件保留原文：RMT 的 HF 上游下载仍有访问条件；MiniMax 元数据与 README 中存在不同许可声明，均保留，不重新指定第三方数据的许可证。
上游未发布的 RMT Poison/SEC 数据与 MiniMax 未公开部分不在本目录，不能把本地结果称为完整官方榜单复现。
