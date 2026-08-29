from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        prog="deepgit",
        description="DeepGit: intelligent GitHub repository search",
    )
    sub = parser.add_subparsers(dest="command")

    # --- search command ---
    search_p = sub.add_parser("search", help="Search GitHub from the command line")
    search_p.add_argument("query", nargs="+", help="Natural language search query")

    # --- mcp command ---
    mcp_p = sub.add_parser("mcp", help="Run the DeepGit MCP server (stdio by default)")
    mcp_p.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport. stdio for local clients (Claude Desktop, Cursor, "
        "VS Code); streamable-http to expose it over a port.",
    )
    mcp_p.add_argument("--host", default="0.0.0.0")
    mcp_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.command == "search":
        from deepgit import search

        query = " ".join(args.query)
        print(f"\nSearching: {query}\n")
        print(search(query))

    elif args.command == "mcp":
        import os

        os.environ["MCP_TRANSPORT"] = args.transport
        os.environ.setdefault("HOST", args.host)
        os.environ.setdefault("PORT", str(args.port))
        from deepgit.mcp_server import main as mcp_main
        mcp_main()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
