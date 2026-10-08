import asyncio
import json
import sys
import tempfile
import unittest

from triton_transform.sessions import SessionStore

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@unittest.skipUnless(MCP_AVAILABLE, "Install the optional [mcp] dependency for protocol integration tests")
class McpIntegrationTests(unittest.TestCase):
    def test_real_stdio_client_discovers_tools_and_queries_analysis(self):
        async def exercise(directory):
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "triton_transform", "mcp", "--session-dir", directory])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    self.assertEqual({t.name for t in tools.tools},
                                     {"analyze_kernel", "inspect_transform", "get_visualization_context"})
                    source = "import triton\nimport triton.language as tl\n@triton.jit\ndef demo():\n    x = tl.arange(0, 8)\n    y = tl.reshape(x, (2, 4))\n"
                    response = await client.call_tool("analyze_kernel", {"source": source})
                    self.assertFalse(response.isError)
                    analysis = response.structuredContent
                    if analysis is None:
                        analysis = json.loads(response.content[0].text)
                    node = next(n for n in analysis["nodes"] if n["name"] == "y")
                    inspected = await client.call_tool("inspect_transform", {
                        "analysis": analysis, "node_id": node["id"], "index": [1, 2]})
                    self.assertFalse(inspected.isError)
                    data = inspected.structuredContent or json.loads(inspected.content[0].text)
                    self.assertEqual(data["origins"][0]["indices"], [[6]])
                    contexts = await client.call_tool("get_visualization_context", {})
                    self.assertFalse(contexts.isError)
                    context_data = contexts.structuredContent or json.loads(contexts.content[0].text)
                    self.assertEqual(context_data["sessions"], [])
                    # Simulate the editor's atomic publication; the MCP server is
                    # a separate process reading the same workspace session.
                    store = SessionStore(directory)
                    store.put("editor-smoke", {"document_id": "", "version": 0, "analysis": analysis,
                                               "selected_node_id": node["id"], "stale": False})
                    selected = await client.call_tool("get_visualization_context", {"session_id": "editor-smoke"})
                    self.assertFalse(selected.isError)
                    selected_data = selected.structuredContent or json.loads(selected.content[0].text)
                    self.assertEqual(selected_data["selected_node_id"], node["id"])
                    from_session = await client.call_tool("inspect_transform", {
                        "session_id": "editor-smoke", "node_id": node["id"], "index": [1, 2]})
                    self.assertFalse(from_session.isError)
                    session_mapping = from_session.structuredContent or json.loads(from_session.content[0].text)
                    self.assertEqual(session_mapping["origins"][0]["indices"], [[6]])
                    store.put("editor-smoke", {"document_id": "", "version": 1,
                                               "analysis": analysis, "stale": True})
                    stale = await client.call_tool("inspect_transform", {
                        "session_id": "editor-smoke", "node_id": node["id"], "index": [1, 2]})
                    self.assertTrue(stale.isError)
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(asyncio.wait_for(exercise(directory), timeout=25))


if __name__ == "__main__":
    unittest.main()
