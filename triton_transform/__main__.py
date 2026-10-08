"""Command line analysis, editor JSON-lines worker and optional MCP server."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def worker(session_dir: str) -> None:
    from .bridge import Bridge

    bridge = Bridge(session_dir)
    for line in sys.stdin:
        request_id = None
        try:
            if len(line) > 2_000_000:
                raise ValueError("Request exceeds the two-million-character limit")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            raw_id = request.get("id")
            if not (raw_id is None or type(raw_id) in (str, int) or
                    (type(raw_id) is float and math.isfinite(raw_id))):
                raise ValueError("Request id must be null, a string or a finite number")
            request_id = raw_id
            # json.loads accepts NaN/Infinity and overflowing exponents;
            # reject them anywhere in the request before invoking services.
            json.dumps(request, allow_nan=False)
            result = bridge.dispatch(request["method"], request.get("params", {}))
            emit({"id": request_id, "result": result})
        except Exception as exc:
            emit({"id": request_id,
                  "error": {"message": str(exc), "type": type(exc).__name__}})


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Explain Triton tensor transformations without executing kernels")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze", help="Analyze a Python source file")
    analyze_parser.add_argument("file", type=Path)
    analyze_parser.add_argument("--kernel")
    analyze_parser.add_argument("--parameters", default="{}", help="JSON scalar/constexpr values")
    analyze_parser.add_argument("--input-shapes", default="{}", help="JSON mapping from argument name to shape")
    analyze_parser.add_argument("--program-ids", default="[0,0,0]", help="JSON list of selected program coordinates")
    for name in ("worker", "mcp"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--session-dir", default=".triton-transform")
    args = parser.parse_args()
    try:
        if args.command == "worker":
            worker(args.session_dir)
        elif args.command == "mcp":
            from .mcp_server import create_server
            create_server(args.session_dir).run(transport="stdio")
        else:
            from .bridge import Bridge, MAX_SOURCE_CHARACTERS
            with args.file.open(encoding="utf-8-sig") as source_file:
                source = source_file.read(MAX_SOURCE_CHARACTERS + 1)
            result = Bridge().dispatch("analyze", {
                "source": source, "kernel": args.kernel, "parameters": json.loads(args.parameters),
                "input_shapes": json.loads(args.input_shapes), "program_ids": json.loads(args.program_ids),
                "document_id": str(args.file.resolve()),
            })
            emit(result)
            if any(d.get("severity") == "error" for d in result["diagnostics"]):
                return 1
        return 0
    except Exception as exc:
        if args.command == "mcp":
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        else:
            emit({"error": {"message": str(exc), "type": type(exc).__name__}})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
