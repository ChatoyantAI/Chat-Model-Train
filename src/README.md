# 源代码目录

正式、可复用的实现放在 `src/<module>/`。当前核心模块为：

- [data](data/README.md)：数据契约、抽取、处理与质量校验；
- [training](training/README.md)：训练编排、恢复与运行记录；
- [evaluation](evaluation/README.md)：固定评测、指标与报告；
- [common](common/README.md)：稳定的跨模块公共能力。

新模块必须包含 README，并在 [职责登记](../docs/RESPONSIBILITIES.md) 中指定负责人。禁止在 `src/` 根目录堆放孤立脚本。
