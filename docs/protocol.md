# 编辑器与分析进程协议

分析核心和工作进程仅依赖 Python 标准库；MCP 适配器使用可选的官方 Python MCP SDK。工作进程通过 UTF-8 JSON Lines 通信，标准输出不包含日志。

## 请求与响应

```json
{"id":"42","method":"analyze","params":{"source":"...","document_id":"file:///kernel.py","version":3,"parameters":{"BLOCK":8}}}
```

```json
{"id":"42","result":{"document_id":"file:///kernel.py","version":3,"kernel":"kernel","kernels":["kernel"],"nodes":[],"diagnostics":[],"missing_parameters":[],"parameters":{"BLOCK":8}}}
```

失败时响应同一请求 ID，字段为 `error: {message, type}`。语法错误和无法分析的操作通常在正常分析结果的 diagnostics 中返回，界面仍可显示已知部分。

方法：

| method | params | 返回 |
| --- | --- | --- |
| `analyze` | source、kernel、parameters、input_shapes、program_ids、document_id、version | 操作图与诊断 |
| `inspect` | analysis 或 session_id、node_id、可选 index、limit | 当前操作与直接输入坐标映射 |
| `sync_context` | session_id、context | accepted、session_id |
| `get_context` | 可选 session_id | 明确会话上下文或会话列表 |
| `clear_context` | session_id | cleared |

`source` 最多一百万字符。`index` 包含完整秩的零基整数坐标，标量的坐标是 `[]`。坐标映射默认每个输入最多返回 128 个位置，硬上限 4096；`total` 保留真实数量，`truncated` 表示列举被截断。

## 节点

节点包含 id、op、name、inputs、shape、status、source、attrs、dtype。shape 维度为正整数或符号字符串，status 为 resolved、symbolic 或 unsupported。resolved 表示分析规则能够确定结果，不表示 kernel 已执行或已经取得实际计算值。

源码范围采用 **1 基行号、0 基 UTF-16 列偏移**，与 VS Code 转换时行号减一。节点标识只在对应分析结果内使用；源码或参数变化后应按当前结果重新选择。

`inspect` 的 origins 指向直接输入，不展开完整的祖先依赖集合。用户可沿操作图继续查看前一步。`status=unavailable` 时不显示看似精确的来源连线。

## 会话

上下文以独立 session_id 存储在工作区 `.tiletrace/`。VS Code 扩展与 MCP 进程必须配置同一个目录。会话标识只允许字母、数字、下划线和连字符，长度 1–80。

只在界面接受结果后发布新的非过期上下文。上下文至少包含 document_id、version 与 stale；有分析结果时添加 analysis、selected_node_id、parameters、input_shapes 和 program_ids。

界面收到新修改、切换文件、参数更改或分析失败时，不得沿用旧的精确映射。源码范围、文档版本和请求代次共同用于拒绝过期响应。stale=true 的上下文可读取供 agent 获知状态，但 `inspect` 拒绝使用它作精确映射。

MCP 服务的 `analyze_kernel` 是独立查询，不会自动覆盖编辑器上下文。`get_visualization_context` 未指定会话时只列出会话，不任选一个窗口。
