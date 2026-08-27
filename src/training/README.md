# Training

> 状态：目录与契约已建立，具体实现待模块负责人提交
>
> 主负责人：待认领
>
> 备份 / Reviewer：待认领

## 目标

以版本化数据和可审查配置启动、监控、恢复训练，并产出能追溯到代码、数据和配置的模型 checkpoint。

## 负责范围

- SFT、偏好优化等训练入口与编排；
- 配置加载、参数校验、随机种子和分布式环境；
- checkpoint 保存/恢复、日志、运行状态和失败恢复；
- 训练指标、资源信息和最终运行 manifest；
- 与模型存储之间的产物契约。

## 不负责

- 隐式清洗或改变上游数据语义；
- 使用正式测试集选择训练超参数；
- 把 checkpoint 或密钥提交到 Git。

## 每次正式训练必须记录

- 代码 commit、数据 manifest/hash、基础模型和 tokenizer；
- 完整解析配置、命令行覆盖、依赖环境和硬件；
- 随机种子、global batch、学习率、steps/tokens；
- 日志、checkpoint、最终模型的不可变 URI 和 hash；
- 成功/失败状态、恢复点和已知异常。

## 建议布局

```text
src/training/
├── README.md
├── cli.py
├── config.py
├── runners/
├── callbacks/
└── manifest.py
```

实现提交时，负责人必须补充实际启动、dry-run、恢复和最小验证命令。
