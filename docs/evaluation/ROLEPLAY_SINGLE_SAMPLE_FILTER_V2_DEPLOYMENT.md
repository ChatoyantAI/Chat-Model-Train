# 单样本过滤 V2：使用与部署

协议：`roleplay-single-sample-filter-v2.2`，质量量表来源 `roleplay-rubric-v0.18`。使用独立页面、Cookie、结果库和目录，只读共享样本索引，不替换人工标注或普通 Judge。

## 启动

从仓库根目录安装平台 requirements，将 [filter-v2.env.example](../../configs/evaluation/filter-v2.env.example) 复制为私有 `filter-v2.env` 后配置路径与服务端 Key。

```bash
python -m uvicorn src.evaluation.roleplay_annotation_platform.filter_v2_app:app --host 127.0.0.1 --port 8767 --workers 1 --env-file configs/evaluation/filter-v2.env
```

长期运行可调整包内 `roleplay-filter-v2.service` 模板。用户名只用于结果命名空间，不是密码认证；公网访问需另加 HTTPS 与访问控制。

## 评分协议

完整公式、档位、边界和案例见 [V2 评分标准](ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md)。V2 不是只输出通过 / 不通过：

- 硬 Gate 使用布尔值，true 表示发现错误。
- 质量评分按维度及标准长度拆成小批，使用 A/B/C/D/X 编码对应 0/1/2/3/N/A，保留每项完整标准与案例。
- 可用且完整的评分 token 分布用于条件期望分；否则回退为离散分，不伪造置信度。
- 每个适用项目按最大档归一化，组内等权，适用大类等权，汇总到 0–100。N/A 的适用性概率单独记录。
- 评分是否成功与概率是否可用分别报告；总分不覆盖单项缺陷和复核要求。

可人工评分，也可发起 LLM 评分及针对缺陷的确认。请求、原始响应、解析结果、渠道、token、费用、耗时保存在独立结果库。Key 不写入结果或页面。过滤阈值属于可校准协议，不声称已经保证训练收益。

## 构建与验证

```bash
python -m src.evaluation.roleplay_annotation_platform.build_filter_v2_rubric
python -m src.evaluation.roleplay_annotation_platform.build_filter_v2_docs
python -m unittest tests.evaluation.test_single_sample_filter_v2 tests.evaluation.test_filter_v2_app -v
```

测试使用虚构数据与 HTTP mock，不访问付费接口。运行产物放在忽略的 `outputs/filter-v2/`；不得将 SQLite、原始聊天或实际环境文件提交。
