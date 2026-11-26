# 复杂问题知识助手

本模块实现问题拆解、子问题检索、证据检查、可选补检索和带引用的逐题回答。环境配置、索引构建和完整运行流程见[仓库首页](../README.md)。

## 调用框架

```text
__main__.py: main()                    解析命令行参数
 └─ run()                             读取模型配置，创建调用适配器
      └─ QuestionDecomposer.decompose() 发送一次问题规划请求
           └─ parse()                 校验 JSON、数量、重复和空值
                └─ QuestionPlan       原问题、策略、子问题列表、原因
```

在仓库根目录运行，使用现有 `.venv`。`complex_qa` 是本仓库的应用模块，
不包含在上游 `lightrag-hku` 的包发现范围内；目前不支持在任意目录安装后运行。

## 先看离线示例

```powershell
.\.venv\Scripts\python.exe -m complex_qa --demo
```

这是固定的人工示例，`source=offline_fixture`，不访问网络、不调用模型，
只能演示数据格式，不能证明真实拆解质量。

## 真实问题拆解

复用 `.env.smoke` 中的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`；
进程中同名环境变量优先。无需向量模型或知识库，运行会产生一次聊天模型调用费用。
使用 OpenAI 兼容的聊天接口，服务端需支持 `response_format=json_object`。

```powershell
.\.venv\Scripts\python.exe -m complex_qa "产品A和产品B，哪个支持离线部署，各有什么限制？"
.\.venv\Scripts\python.exe -m complex_qa "产品A支持离线部署吗？"
```

保存结果可加 `--output output/decomposition/compare.json`，文件必须不存在，防止覆盖记录。
`output/` 已被仓库忽略；基线结果与本模块输出分开保存。输出包括模型名、实际 Token
用量（接口未返回时为 null）、耗时、原问题和拆解计划，不保存密钥。

## 第一版边界与验收

- 简单问题采用 `direct`，程序始终保留原问题全文。
- 复杂问题采用 `decompose`，拆成 2～4 个可独立检索的问题；可用
  `--max-subquestions 2` 缩小上限。比较题应保留一致的比较条件。
- 不执行需要前置检索才能确定对象的依赖问题，提示模型对此采用 `direct`。
  后续加入依赖任务执行时再扩展协议。
- JSON、字段类型、空值、数量、重复问题在本地校验；无效响应、超时或调用失败
  明确报错并返回非零退出码，不伪造“拆解成功”。默认一次请求，无自动重试。
- 对象、日期、否定和条件是否忠实仍需人工核对。提示词约束和结构校验
  不等于语义保证；合理示例也不等于相对基线已经有质量增益。

首次验收：简单问题保留原问；产品比较拆成两个对象的问题；每个子问题都保留
“离线部署、限制”等条件；不要出现资料中的答案或模型猜出的产品属性。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/complex_qa -v
```

测试使用注入的假模型，覆盖输入、格式、边界、条件保留、超时与取消，
以及证据去重、空召回和部分失败。

## 按已保存计划检索证据

```powershell
.\.venv\Scripts\python.exe -m complex_qa "星链X2和蓝鲸Y1分别支持断网运行吗？" --output output/decomposition/offline-comparison.json
.\.venv\Scripts\python.exe -m complex_qa.retrieval output/decomposition/offline-comparison.json --mode naive --output output/retrieval/offline-comparison-naive.json
```

检索读取现有基线索引，沿用基线的默认 Top-K 配置和模型接线。
结果记录各子问题的状态、原文和来源，以及按块 ID 去重的证据列表。
同一个文本块可关联多个子问题。召回非空仅表示找到了候选证据，
具体是否足够回答还需要下一阶段检查。

## 检查证据并生成回答

使用已经保存的子问题检索结果；默认不会重新调用检索接口：

```powershell
.\.venv\Scripts\python.exe -m complex_qa.answer output/retrieval/offline-comparison-naive.json --output output/answer/comparison.json
```

每个有候选证据的子问题调用一次聊天模型检查，存在充分证据时再调用一次模型生成逐题答案。
结果保留每题判断、原因、答案、引用块 ID、来源和各次调用的 token。未知引用或跨题引用会报错。
部分证据可以回答的题目保留已有答案，并标明缺少的信息；证据不足或检索失败的题目明确标注无法回答。
加上 `--supplement` 可针对证据检查识别出的缺口最多补检索一次，再检查合并后的证据：

```powershell
.\.venv\Scripts\python.exe -m complex_qa.answer output/retrieval/offline-comparison-naive.json --supplement --output output/answer/comparison-supplement.json
```

补检索使用已有索引及输入文件记录的检索参数（未记录时默认 naive、Top-K 3）。模型对证据和查询词的判断仍须人工检查。真实服务请求可能产生费用，输出文件必须不存在。
