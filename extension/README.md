# TileTrace

通过静态 Python AST 分析 Triton kernel 的逻辑形状和变换，界面不执行用户源码或 kernel，不显示实际张量数值。采用 MIT 许可证（见仓库根目录 LICENSE）。

在工作区中打开带 `@triton.jit` 的已保存 `.py` 文件，扩展默认在右侧打开中文面板。也可运行命令 **TileTrace: 打开变换可视化**。支持常见 `import triton as ...`、`from triton import jit as ...` 装饰器别名；复杂导入形式可能需要改用常见形式。未保存文件、工作区外文件不会建立会话。切到普通 Python 文件会清除当前会话，面板获得焦点时保留对应源码编辑器。

选择 kernel，在“分析参数”中提供 JSON：例如 constexpr `{"BLOCK":8}`、输入形状 `{"x":[2,4]}` 和 program ID `[0]`。输入形状是可选静态信息。查看操作列表、上一步/下一步、代码表达式、输入和输出形状，点击输出格子查询直接输入的来源坐标。选中源码表达式也会同步选中的操作；查看源码按钮跳转至原文件。网格代表逻辑坐标，不代表 GPU 线程、全局内存地址或实际数据值。

默认列表显示关键张量步骤，折叠常量、参数和纯标量索引准备。勾选“显示全部节点”恢复完整列表；代码选择与前后导航跟随该视图。`arange` 的起点、终点、长度显示在“参数与属性”中；张量归约产生的标量以及使用这些结果的计算继续作为步骤和数据输入显示。MCP 接收完整图，同时提供当前 `view_mode` 与 `visible_node_ids`。

每张卡片最多显示 128 个坐标（8 行 × 16 列）；超出范围可用轴起点浏览。rank > 2 使用前缀轴切片，最后两个轴作为网格轴。符号形状显示文字，未支持操作不给出精确映射。映射区标注返回与总计数量、枚举截断、当前网格可见高亮数量，并提供“定位首个来源坐标”。编辑源码或改变参数期间清除网格及旧映射，忽略过期的分析和检查响应。

按钮、下拉框、JSON 输入与轴切片支持键盘操作，并使用 VS Code 主题与焦点颜色。查看诊断可以区分缺少参数、形状不兼容和分析能力限制。无 kernel 节点时显示诊断与补充参数提示。

同一文件的重绘保留页面与内部列表/网格滚动、参数区展开状态、焦点及 JSON 输入光标位置。等待新分析的短页面不会覆盖完整视图的阅读位置；新文件使用自己的阅读位置，首次打开从顶部开始。页面内容减少时，滚动位置受浏览器新的范围限制。关闭并重新创建面板后阅读位置重新初始化。

## Python 和会话

运行时只需要 Python 3.10+，不需要 Triton、NumPy 或 Python MCP SDK。扩展没有 JavaScript 运行时依赖；VSIX 内置本仓库的标准库 Python worker。仅在 VS Code 已信任的工作区启动分析；配置的 Python 可执行文件及工作区 `.venv` 也必须可信。

- `tiletrace.pythonPath`：默认 `python`，优先使用拥有当前文件的工作区 `.venv/Scripts/python.exe`（Linux/macOS 为 `.venv/bin/python`）；也可设置 Python 可执行文件的绝对路径，不添加命令行参数或额外引号。
- `tiletrace.autoOpen`：是否自动打开相关源文件的面板，默认 true。
- `tiletrace.sessionDirectory`：默认 `.tiletrace`，相对路径基于拥有当前源文件的工作区。MCP 服务必须使用相同的绝对目录。多根工作区切换时清理旧会话并为新的文件夹启动进程。

独立的分析进程和上下文进程避免耗时分析阻塞 `stale:true` 发布。进程使用 JSON Lines，stderr 写入 **TileTrace** 输出通道。请求约 15 秒超时会终止进程，下一次请求重新启动；可主动运行 **TileTrace: 重启分析进程**。Python 找不到时检查该输出通道并配置可执行路径。

进程以 `-I -S` 隔离模式从扩展自带的 Python 目录启动，通过固定的扩展代码加载明确的后端目录；开发时使用此工具仓库中的后端。源码工作区不进入当前目录或导入搜索路径，`PYTHONPATH`、`PYTHONHOME`、用户 site 和 site 启动脚本不影响 worker。用户源码仅作为 JSON 字符串交给静态分析器。每次新建或重启 worker 组使用新的 session_id，旧进程的延迟清理只作用于旧会话；重启后请重新复制 Agent 提示词。

**TileTrace: 复制 Agent 提示词** 将精确 session_id 放入剪贴板，要求 Agent 调用 `get_visualization_context(session_id)`，再用 `inspect_transform(...)` 检查当前选中的节点和坐标。扩展不修改全局 Codex 配置，也不自动向 Agent 发送消息。MCP 连接需按仓库根目录 README 配置；这不是已完成实时 Codex 对话的证明。

## 开发与安装

在仓库根目录的 PowerShell 中：

```powershell
Set-Location extension
npm.cmd install
npm.cmd test
npm.cmd run package
```

`package` 会编译 TypeScript，将根目录 `tiletrace/*.py` 复制到生成的 `extension/python/tiletrace/`，输出 `dist/tiletrace-0.1.7.vsix`。不打包虚拟环境、node_modules、tests 或 pycache。必须从完整仓库构建；从 VSIX 安装后不需要可编辑 Python 安装。

在 VS Code 打开仓库后选择调试配置 **TileTrace Extension** 并按 F5；先在 `extension/` 安装开发依赖。扩展开发宿主中打开 `examples/` 的 Triton 文件。通过 VS Code 扩展面板的 **Install from VSIX…** 手动安装生成的包。

自动验证包含节点选择、版本和请求代次保护、真实子进程通信/退出/超时、耗时分析与独立上下文通信，以及 jsdom 中的输出点击、来源高亮、高维切片、符号状态和巨型网格限制。自动测试不替代真实 VS Code 扩展宿主中的布局/焦点检查。
