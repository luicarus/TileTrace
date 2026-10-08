"""Shared service surface used by the editor worker and MCP tools."""

from __future__ import annotations

from pathlib import Path

from .analyzer import analyze, inspect_transform
from .sessions import SessionStore


MAX_SOURCE_CHARACTERS = 1_000_000


class Bridge:
    def __init__(self, session_dir: str | Path = ".triton-transform"):
        self.sessions = SessionStore(session_dir)

    def dispatch(self, method: str, params: dict) -> dict:
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        if method == "analyze":
            allowed = {"source", "kernel", "parameters", "input_shapes", "program_ids", "document_id", "version"}
            if set(params) - allowed:
                raise ValueError("Unknown analyze parameters: " + ", ".join(sorted(set(params) - allowed)))
            if not isinstance(params.get("source"), str):
                raise ValueError("source must be a string")
            if len(params["source"]) > MAX_SOURCE_CHARACTERS:
                raise ValueError("Source exceeds the one-million-character analysis limit")
            return analyze(**params)
        if method == "inspect":
            analysis = params.get("analysis")
            if analysis is None:
                context = self._context(params.get("session_id"))
                if context.get("stale"):
                    raise ValueError("Editor context is stale; wait for the latest analysis")
                analysis = context.get("analysis")
            if not isinstance(analysis, dict):
                raise ValueError("Provide an analysis object or a session with a completed analysis")
            return inspect_transform(analysis, params["node_id"], params.get("index"), params.get("limit", 128))
        if method == "sync_context":
            context = params.get("context")
            if not isinstance(context, dict):
                raise ValueError("context must be an object")
            analysis = context.get("analysis")
            if analysis is not None:
                if not isinstance(analysis, dict) or analysis.get("document_id") != context.get("document_id"):
                    raise ValueError("Analysis document does not match context")
                if not context.get("stale") and analysis.get("version") != context.get("version"):
                    raise ValueError("Analysis version does not match context")
                selected = context.get("selected_node_id")
                if selected is not None and not any(n.get("id") == selected for n in analysis.get("nodes", [])):
                    raise ValueError("Selected node is not part of the analysis")
            accepted = self.sessions.put(params["session_id"], context)
            return {"accepted": accepted, "session_id": params["session_id"]}
        if method == "get_context":
            if params.get("session_id") is None:
                return {"sessions": self.sessions.list_sessions(),
                        "message": "Choose an explicit session_id; each editor window has its own context."}
            return {"session_id": params["session_id"], **self._context(params["session_id"])}
        if method == "clear_context":
            self.sessions.clear(params["session_id"])
            return {"cleared": True}
        raise ValueError(f"Unknown method: {method}")

    def _context(self, session_id: str | None) -> dict:
        if session_id is None:
            raise ValueError("session_id is required")
        context = self.sessions.get(session_id)
        if context is None:
            raise ValueError(f"No synchronized editor context for session {session_id}")
        return context
