"""``python -m grl_snam.ariadne <port> [args]``: run one of GRL-SNAM's Ariadne ports."""

from __future__ import annotations

import importlib
import sys

from grl_snam.ariadne import PORTS


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print(
            "usage: python -m grl_snam.ariadne <port> [--offscreen | --headless] "
            "[--frames N] [--png PATH] [port args]\n\nports:"
        )
        for name, (_, what, original) in PORTS.items():
            print(f"  {name:9} {what}  [{original}]")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in PORTS:
        print(f"grl_snam.ariadne: no port {name!r} (--list names them)", file=sys.stderr)
        return 2
    return int(importlib.import_module(PORTS[name][0]).main(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
