"""Load STM-Bench's maintained controller command registry.

The registry keeps the legacy-compatible binary formats needed by the simulator's
implemented controller surface. It is deliberately a local maintenance record,
not a claim that STM-Bench invented the protocol or that the formats' provenance
has been independently established.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SPEC_PATH = Path(__file__).resolve().parent.parent / "spec" / "command_registry.json"

# ``c``-typed request arguments that are string ARRAYS on the wire (see codec.decode_args).
ARRAY_STRING_ARGS: dict[str, tuple[int, ...]] = {
    "Scan.PropsSet": (5,),
}


@dataclass(frozen=True)
class CommandSpec:
    """One locally maintained, legacy-compatible wire command definition."""

    method: str
    command: str
    module: str
    arg_fmts: tuple[str, ...]
    arg_names: tuple[str, ...]
    ret_fmts: tuple[str, ...]
    source: str = "local-registry"


@lru_cache(maxsize=1)
def load_spec(path: Path | str = SPEC_PATH) -> dict:
    """Return the checked-in local registry without consulting external sources."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def commands() -> dict[str, CommandSpec]:
    """Wire command name → command definition."""
    out: dict[str, CommandSpec] = {}
    for command, entry in load_spec()["commands"].items():
        method_names = entry["methods"]
        out[command] = CommandSpec(
            method=method_names[0], command=command, module=command.split(".", 1)[0],
            arg_fmts=tuple(entry["args"]),
            arg_names=tuple(f"arg{n}" for n in range(len(entry["args"]))),
            ret_fmts=tuple(entry["returns"]),
        )
    return out


@lru_cache(maxsize=1)
def methods() -> dict[str, CommandSpec]:
    """Client-style method name (including retained aliases) → command definition."""
    by_command = commands()
    out: dict[str, CommandSpec] = {}
    for command, entry in load_spec()["commands"].items():
        for method in entry["methods"]:
            out[method] = by_command[command]
    return out
