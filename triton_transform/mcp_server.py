"""Optional official-SDK MCP adapter; static analysis never runs user kernels."""

from __future__ import annotations

from typing import Any

from .bridge import Bridge


def create_server(session_dir: str = ".triton-transform"):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError('Install the MCP extra first: python -m pip install -e ".[mcp]"') from exc

    bridge = Bridge(session_dir)
    server = FastMCP(
        "Triton Transform Visualizer",
        instructions=("Explain logical Triton tensor transformations using static analysis. Never execute kernels. "
                      "Use get_visualization_context to list editor sessions, then request an explicit session_id. "
                      "Treat stale, symbolic and unsupported results honestly. Source ranges use one-based lines "
                      "and zero-based UTF-16 columns. Origins describe immediate input coordinates."),
        log_level="ERROR",
    )
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @server.tool(annotations=annotations)
    def analyze_kernel(source: str, kernel: str | None = None,
                       parameters: dict[str, Any] | None = None,
                       input_shapes: dict[str, list[int]] | None = None,
                       program_ids: list[int] | None = None,
                       document_id: str = "", version: int = 0) -> dict[str, Any]:
        """Analyze supplied Triton source without importing it; return shapes, dependencies and diagnostics.

        Supply constexpr/scalar values in parameters. Unspecified dimensions remain symbolic.
        This query does not change the editor's published visualization context.
        """
        return bridge.dispatch("analyze", {"source": source, "kernel": kernel, "parameters": parameters,
                                            "input_shapes": input_shapes, "program_ids": program_ids,
                                            "document_id": document_id, "version": version})

    @server.tool(annotations=annotations)
    def inspect_transform(node_id: str, index: list[int] | None = None,
                          session_id: str | None = None, analysis: dict[str, Any] | None = None,
                          limit: int = 128) -> dict[str, Any]:
        """Explain a transform and map one output coordinate to its immediate input coordinates.

        Provide either analysis from analyze_kernel or an explicit synchronized editor session_id.
        Symbolic/unsupported mappings are reported as unavailable; large reductions are bounded.
        """
        return bridge.dispatch("inspect", {"node_id": node_id, "index": index, "session_id": session_id,
                                            "analysis": analysis, "limit": limit})

    @server.tool(annotations=annotations)
    def get_visualization_context(session_id: str | None = None) -> dict[str, Any]:
        """List editor sessions, or read one explicit session's current selection and versioned analysis.

        Context is synchronized by the VS Code extension, including unsaved source analysis.
        If stale is true, wait for the editor's new analysis before making exact claims.
        """
        return bridge.dispatch("get_context", {"session_id": session_id})

    return server
