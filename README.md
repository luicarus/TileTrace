# TileTrace

独立实现的 Triton 张量变换可视化工具。打开 Triton 脚本，选中表达式，在 VS Code 右侧查看形状变化与元素来源；Codex 或其他 MCP agent 可查询同一份分析结果。

分析器不执行脚本、不导入 Triton、不启动 kernel；使用 Python 3.10+，无需 GPU。采用 [MIT 许可证](LICENSE)。

> 状态：0.1.5 早期版本。VS Code 扩展尚未发布到扩展市场，请按下方说明从源码安装；CLI 与 MCP 服务可直接使用。

## 当前功能

- `arange`、新增轴、广播、reshape、transpose、sum/max、二维 `tl.dot`、显式类型转换，以及常见逐元素操作的静态形状分析。
- 点击输出坐标，查看参与该位置计算的直接输入坐标；大归约限定展示数量。
- 支持符号尺寸和缺失参数提示，未知语法提供诊断。
- VS Code 读取未保存缓冲区、跟随选择和参数变化、反向定位源码。
- 每个编辑器会话独立同步状态；Codex 通过 MCP 查询，不自动猜测当前文件。
- 默认只显示关键张量步骤，参数与常量折叠到操作详情；“显示全部节点”可以查看完整分析图。标量归约及依赖归约结果的计算仍然保留。

展示的是逻辑坐标与依赖。它不表示实际数值执行、GPU 线程布局、内存搬运或性能测量。只展开已知整数边界的 `tl.static_range`，一次分析共享最多 16 次循环迭代；未知/超限循环、循环控制跳转、数据相关分支与跨函数调用仍保留诊断。FlashAttention 示例是单 batch/head 的教学前向实现，尚未在本机验证 GPU 数值或性能。

## Windows 快速开始

在本项目目录执行：

```powershell
.\scripts\setup.ps1
code --install-extension .\dist\tiletrace-0.1.5.vsix --force
```

安装后打开本项目的 `examples/flash_attention.py`。面板默认随 Triton 文件打开，也可以通过命令面板运行 **TileTrace: 打开变换可视化**。若修改了 Python 路径，使用 **TileTrace: 重启分析进程** 重启分析进程。

示例只保留一个 `flash_attention_forward`，默认参数无需填写 JSON。每个 program 处理 16 个 Q 行，每次循环读取 32 个 K/V 行，head 维度为 32，序列长度为 64。`program_ids` 填 `[1]` 可切换到第二块 Q 行；`{"CAUSAL":false}` 可关闭因果掩码。

依次查看行列 `arange`、新增轴、`q`/`k`/`v`、K 转置、`scores`、softmax 的归约与广播、`accumulator`、`output`。点击矩阵乘法输出坐标可查看左右输入的整行/整列；循环内操作标注 `start_n=0` 或 `start_n=32`。示例使用在线 softmax，逐块更新结果，无需构建整个注意力矩阵。

默认只展示张量步骤，常量与参数折叠到操作详情。选中 `arange` 时展示起点、终点和长度；源码中的边界常量或归约轴会选中所属张量操作。前后导航遵循当前视图，底层完整图仍供 MCP 查询。

一维张量的行列方向根据实际新增轴用法展示：`query_rows[:, None]` 对应行轴，因此 `query_rows` 的 16 个索引纵向排列；`head_cols[None, :]` 对应列轴，横向排列。真实 shape 仍是 `[16]` 或 `[32]`。同一向量用于两个方向时，查看具体新增轴操作可获得相应方向；单独查看时会提示方向未指定。矩阵卡片同时说明完整行列数与网格显示窗口，避免将只显示的 8 行误读成整个 program 的行数。

超过 8 个坐标的轴默认显示前 3 项、`…`、后 3 项，例如 16 行显示为 `0、1、2、…、13、14、15`；行列方向与完整 shape 保持不变。点击省略号或“展开显示窗口”查看中间坐标，仍最多显示 128 个格子。“定位首个来源坐标”会自动展开对应输入；来源文本也只展示返回坐标的首尾各 3 项，并保留枚举截断说明。

同一文件中点击格子、切换步骤或收到分析结果时，会保留页面、列表和网格的滚动位置，以及参数区展开状态和输入焦点。重新分析时暂时缩短的等待页面不会覆盖原阅读位置；首次切换到另一文件从顶部开始。内容变短时，浏览器会将位置限制在新的可滚动范围内。

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
.\.venv\Scripts\python.exe -m tiletrace analyze examples/flash_attention.py --kernel flash_attention_forward
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

VS Code 扩展开发与打包说明见 [extension/README.md](extension/README.md)。设计与边界见 [设计文档](docs/design/tiletrace.md)，自动化证据与实际验收限制见 [验证记录](docs/verification.md)。

## 来源

分析规则依据 [Triton 官方 API](https://triton-lang.org/main/python-api/triton.language.html)，编辑器接入依据 [VS Code Webview API](https://code.visualstudio.com/api/extension-guides/webview)。[TileLens](https://github.com/Deep-Learning-Profiling-Tools/tilelens) 用于功能覆盖比较；本项目没有复制其代码或界面。

## 许可证

[MIT](LICENSE)
