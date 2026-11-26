# LightRAG 复杂问题知识助手 · 复习笔记

本项目用 LightRAG 建库和检索，在应用层增加问题拆解、逐题检索和证据归集。复习时先看两条原生链路，再看自研模块怎样调用检索接口。

## 1. 原项目怎样入库

```text
ainsert()                              lightrag/lightrag.py:2310
 ├─ 登记文档和切块配置
 └─ apipeline_process_enqueue_documents()  lightrag/pipeline.py:1724
      └─ _run_pipeline_batch()             lightrag/pipeline.py:2619
           └─ 启动异步 _process_worker()   lightrag/pipeline.py:4950
                └─ 从队列取文档，调用 process_single_document()
                                          lightrag/pipeline.py:4991
                     ├─ 切块，保存原文，建立文本块向量索引
                     ├─ 经抽取包装层调用 extract_entities()
                     │                    lightrag/operate.py:3942
                     └─ merge_nodes_and_edges()
                                          lightrag/operate.py:3514
                          合并实体和关系，更新图谱、向量索引及来源
```

批处理同时启动解析、分析和处理任务。文档经队列交接给处理任务，处理任务才调用单文档处理函数。源码见 [入库入口](../lightrag/lightrag.py) 和 [入库流水线](../lightrag/pipeline.py)。

同一段「星链 X2 支持断网运行」可同时提供三类数据。

| 数据 | 用途 |
|---|---|
| 文本块原文、文本块向量 | 找到语义相近的原文 |
| 实体和关系的向量索引 | 通过语义查找相关节点或边 |
| 图谱节点、边和 `source_id` | 查看实体连接，并沿来源块 ID 找回原文 |

默认图谱通过 NetworkX 保存节点和边。边的两个端点负责连接实体，关系含义保存在边属性中。实体向量由名称和描述生成，关系向量由关键词、端点名称和关系描述生成。源码见 [图谱存储](../lightrag/kg/networkx_impl.py) 和 [图谱构建与检索](../lightrag/operate.py)。

## 2. 原项目怎样查询

```text
aquery_llm()                           lightrag/lightrag.py:5114
 ├─ naive → naive_query()              lightrag/operate.py:6853
 │    文本块向量召回 → 组织上下文 → 生成答案
 └─ local / global / hybrid / mix → kg_query()
                                       lightrag/operate.py:4693
      ├─ 提取问题关键词
      ├─ _build_query_context()        lightrag/operate.py:6076
      │    ├─ 召回实体、关系及适用的向量文本块
      │    ├─ 按预算裁剪实体和关系
      │    ├─ 回查并合并原文块
      │    └─ 组织上下文与引用
      └─ 组装提示词，生成答案
```

| 模式 | 检索起点 |
|---|---|
| `naive` | 文本块向量 |
| `local` | 实体向量命中节点，再查相连的边 |
| `global` | 关系向量命中边，再查两端节点 |
| `hybrid` | 合并 local 与 global |
| `mix` | 图谱检索加文本块向量检索 |

`QueryParam` 控制模式、Top-K 和上下文预算。抽取、关键词和回答模板位于 `prompt.py`。入库切块决定文本处理单元，查询裁剪决定多少证据进入模型上下文。源码见 [查询参数](../lightrag/base.py)、[提示词](../lightrag/prompt.py) 和 [检索操作](../lightrag/operate.py)。

## 3. 我们新增了什么

```text
decompose()                           complex_qa/decompose.py:69
 ├─ 模型判断保留原问或拆成 2～4 个独立子问题
 └─ parse()                           complex_qa/decompose.py:93
      校验 JSON、数量、空值和重复项，产出 QuestionPlan

retrieve_plan()                       complex_qa/retrieval.py:17
 ├─ 遍历子问题
 │    └─ aquery_data()                lightrag/lightrag.py:4901
 │         获取本题候选证据
 ├─ 保留每题状态、文本块和引用
 └─ 按 chunk_id 去重，记录每块关联的子问题序号
```

拆解由聊天模型生成，程序校验的是结构。原问题中的条件是否完整保留，需要结合输出检查。检索阶段的 `aquery_data()` 返回证据。图谱模式的关键词抽取仍可能调用模型。

**已实现**问题拆解、逐题检索、来源记录、去重、证据检查和带引用的逐题回答，并支持按缺口最多补检索一次。更复杂的多轮依赖推理及更全面的对照评测仍需继续完善。源码见 [问题拆解](decompose.py)、[逐题检索](retrieval.py)和[证据检查与回答](answer.py)。

## 4. 用真实例子复习

「星链 X2 和蓝鲸 Y1 分别支持断网运行吗？」被拆成两个问题。每个问题都召回同样两段原文，去重后有两块，且每块都记录了两个召回来源。星链 X2 的原文支持断网，蓝鲸 Y1 的原文要求联网。**一个文本块被某题召回，只表示它是候选证据；能否支撑该题结论由下一步检查。**

历史验证中，子问题检索耗时约 10.56 秒，不含拆解；17 项离线测试通过。详情可查看 [已保存的拆解计划](../output/decomposition/baseline-product-comparison.json) 和 [检索记录](../output/retrieval/baseline-product-comparison-naive.json)。这些运行文件保存在本地，复跑命令见 [使用说明](README.md)。

基线的 4 道题中，`naive` 和 `mix` 的平均证据召回都约为 94%，平均耗时分别为 3.77 秒和 18.23 秒。两种模式在 Q3 都漏了一段关键证据。这是后续分析案例，答案关键词命中只用于粗筛。评测时比较直接检索、拆解检索和加补检索三种流程，记录检索次数、证据量、耗时及 Token，再检查答案和引用。见 [基线报告](../baseline/results/latest/report.md)。

## 5. 开发困难：部分证据被整题拒绝，缺口无法继续检索

**现象**：Q3 问处罚原因、金额和项目技术负责人。第一次召回了处罚相关证据，却漏掉负责人。由于这是有前后依赖的问题，拆解阶段保留了整题；原先证据检查只按整题判断“充分/不足”，导致有依据的处罚信息也没有进入答案。

**原因**：逐个子问题检查并不等于逐个事实检查；当多个事实仍在同一个子问题里，二元判断无法表达“部分有依据”。第一次向量召回缺少负责人时，原流程也不会主动再查。

**改进思路**：

```text
answer_retrieval()                     complex_qa/answer.py:87
 ├─ check_evidence()                   complex_qa/answer.py:43
 │    └─ 判断充分 / 部分支持 / 不足，保留有依据的块 ID，提出缺口检索词
 ├─ 针对缺口最多补检索一次             complex_qa/answer.py:104
 ├─ 合并去重并再次检查证据
 └─ 用已选证据生成带引用的答案；仍缺什么就明确说明
```

补检索失败或复核失败时，保留原有部分证据，不把缺失信息编成答案。程序检查引用 ID 是否属于当前问题，但模型对证据和答案的语义判断仍可能出错，需要人工核查。`--supplement` 才会启用补检索。

**Q3 实测**：首次召回命中 4 个标注证据中的 3 个；检查识别缺口，并生成“极光研究院 穹顶计划 技术负责人”作为补检索词。补检索找到“周敏”所在原文，合并后再次检查支持完整回答，输出违规处理用户信标数据、罚款 320 万元及负责人周敏，引用了 4 个证据块。[查看复测结果](../output/answer/q3-after-supplement.json)。

首次复测中，补检索也找到了负责人，但复核报错，系统按设计保留了处罚信息的部分答案。后来提高模型响应长度上限并再次复测成功；无法据此确定首次报错的根因或保证每次调用都成功。这是单个案例，不足以证明整体效果优于基线，还需要更多题目对比准确率、引用、耗时和调用成本。

## 6. 前后端从哪里接

```text
useQuerySession                       lightrag_webui/src/features/retrieval/useQuerySession.ts:443
 └─ queryTextStream()                 lightrag_webui/src/api/lightrag.ts:879
      └─ POST /query/stream
           └─ query_text_stream()     lightrag/api/routers/query_routes.py:837
                └─ aquery_llm()        lightrag/lightrag.py:5114
```

原 WebUI 用 React 和 TypeScript，后端用 FastAPI。现有流式接口通过 NDJSON 逐行传数据。自己的功能可以先用普通 JSON 展示子问题、证据和答案，完成后再接流式更新。源码见 [前端会话](../lightrag_webui/src/features/retrieval/useQuerySession.ts) 和 [后端路由](../lightrag/api/routers/query_routes.py)。

## 7. 合上文档自测

1. 为什么入库已经有文本块向量，还要保存实体、关系和原文来源？
2. `local` 与 `global` 各从哪里开始检索，怎样回到原文？
3. 为什么复杂问答的子问题阶段调用 `aquery_data()`？
4. 两个子问题召回同一个块，怎样去重又保留各自来源？
5. 某题召回了两段文字，怎么判断其中一段是否真的支撑答案？
6. 如果拆解组比直接检索多找回一个块，怎样检查是不是多查几次带来的收益？
