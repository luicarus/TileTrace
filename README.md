# TileTrace

独立实现的 Triton 张量变换可视化工具。打开 Triton 脚本，选中表达式，在 VS Code 右侧查看形状变化与元素来源；Codex 或其他 MCP agent 可查询同一份分析结果。

分析器不执行脚本、不导入 Triton、不启动 kernel；使用 Python 3.10+，无需 GPU。采用 [MIT 许可证](LICENSE)。

> 状态：0.1.0 早期版本。VS Code 扩展尚未发布到扩展市场，请按下方说明从源码安装；CLI 与 MCP 服务可直接使用。

## 当前功能

- `arange`、新增轴、广播、reshape、transpose、sum/max，以及常见逐元素操作的静态形状分析。
- 点击输出坐标，查看参与该位置计算的直接输入坐标；大归约限定展示数量。
- 支持符号尺寸和缺失参数提示，未知语法提供诊断。
- VS Code 读取未保存缓冲区、跟随选择和参数变化、反向定位源码。
- 每个编辑器会话独立同步状态；Codex 通过 MCP 查询，不自动猜测当前文件。

展示的是逻辑坐标与依赖。它不表示实际数值执行、GPU 线程布局、内存搬运或性能测量。复杂循环、数据相关分支与跨函数调用不在第一版完整分析范围。

## Windows 快速开始

在本项目目录执行：

```powershell
.\scripts\setup.ps1
code --install-extension .\dist\tiletrace-0.1.0.vsix
```

安装后打开本项目的 `examples/transforms.py`。面板默认随 Triton 文件打开，也可以通过命令面板运行 **TileTrace: 打开变换可视化**。若修改了 Python 路径，使用 **TileTrace: 重启分析进程** 重启分析进程。

选择 `broadcast_demo`，点击 `matrix`，再点击输出网格中的任意坐标，即可观察两个广播输入的对应位置。切换到 `reshape_demo` 查看 reshape 与转置；`softmax_demo` 可补入 `{"N":6}`。

输入参数与 `input_shapes` 是 JSON 对象；program IDs 是坐标数组。形状未知时先补充缺失参数。源文件仅被静态读取，不能通过修改示例参数取得执行数值。

## Codex 接入

`setup.ps1` 仅在 `.codex/config.toml` 不存在时生成本项目配置；已有文件会完整保留，请按下方示例手动添加或更新服务。生成的配置包含本机绝对路径，已加入 Git 忽略规则；移动项目后请手动更新路径。脚本不会修改用户级配置。该配置启动本地 STDIO MCP 服务；项目需要受信任，并在新 Codex 会话中加载。官方说明：[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp)。

如果把源码移到其他工作区，请让扩展的 `tiletrace.sessionDirectory` 与 MCP 的 `--session-dir` 指向同一个目录。可在需要分析的项目中手动添加：

```toml
[mcp_servers.tiletrace]
command = "本工具虚拟环境的 Python 绝对路径"
args = ["-m", "tiletrace", "mcp", "--session-dir", "目标工作区/.tiletrace 的绝对路径"]
cwd = "本工具项目的绝对路径"
```

Windows 路径可使用正斜杠 `/`；若使用反斜杠，请在 TOML 双引号字符串中写成 `\\`。

面板提供复制 agent 提示词的操作。提示词携带明确 session ID；让 Codex 查询 `get_visualization_context` 并调用 `inspect_transform`。三个工具：

| 工具 | 功能 |
| --- | --- |
| `analyze_kernel` | 从提供的源码及参数生成操作图 |
| `inspect_transform` | 查询一个操作的形状与输出坐标来源 |
| `get_visualization_context` | 列出会话或读取明确会话的当前选择 |

独立分析调用不会改变编辑器已发布的上下文。源码变化后上下文会标记为过期；不要基于过期结果作精确解释。

## 命令行与开发

```powershell
.\.venv\Scripts\python.exe -m tiletrace analyze examples/transforms.py --kernel reshape_demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

VS Code 扩展开发与打包说明见 [extension/README.md](extension/README.md)。设计与边界见 [设计文档](docs/design/tiletrace.md)，自动化证据与实际验收限制见 [验证记录](docs/verification.md)。

## 来源

分析规则依据 [Triton 官方 API](https://triton-lang.org/main/python-api/triton.language.html)，编辑器接入依据 [VS Code Webview API](https://code.visualstudio.com/api/extension-guides/webview)。[TileLens](https://github.com/Deep-Learning-Profiling-Tools/tilelens) 用于功能覆盖比较；本项目没有复制其代码或界面。

## 许可证

[MIT](LICENSE)
