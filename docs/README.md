# 文档目录

`docs/` 保存跨模块规范、架构决策、操作手册和可复现实验记录。模块自身的使用说明应优先放在对应的 `src/<module>/README.md`，避免模块文档与代码分离后失效。

## 导航

- [RESPONSIBILITIES.md](RESPONSIBILITIES.md)：模块负责人、备份负责人和维护边界；
- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md)：目录用途与新增模块规则；
- [templates/MODULE_README_TEMPLATE.md](templates/MODULE_README_TEMPLATE.md)：新模块说明模板；
- [templates/EXPERIMENT_RECORD_TEMPLATE.md](templates/EXPERIMENT_RECORD_TEMPLATE.md)：数据、训练和评测实验记录模板。

新增长期有效的团队约定时，应更新相应规范；单次运行记录应放入 `docs/experiments/`，文件名使用稳定实验 ID，例如 `2026-08-27-rp-sft-v1.md`。
