"""controller-style errors, as the wire encodes them.

A controller reply body always ends in an *error section*::

    [int32 status][int32 desc_len][desc bytes]

``status == 0`` means "no error" (and ``desc_len`` is normally 0). When the instrument
rejects a command (module not running, unknown verb, bad argument) the body contains
**only** this section — no declared return fields.

The compatibility client relies on two properties of the description text:

* the length identity ``len(body) == 8 + desc_len`` — that is how it tells a rejection
  apart from a mis-declared reply;
* the substring ``NeedModule`` — that is how acquisition guards decide "module not loaded"
  rather than "wire fault".
"""
from __future__ import annotations


class WireError(Exception):
    """Raised by a module handler; the server turns it into an error-only reply."""

    status: int = 1

    def __init__(self, description: str, status: int = 1):
        super().__init__(description)
        self.description = description
        self.status = status


class ModuleNotRunning(WireError):
    """The verb belongs to a module that is not loaded on this rig profile."""

    def __init__(self, module: str):
        super().__init__(
            f"NeedModule: Cannot access the '{module}' module. "
            f"Please make sure it is running.", status=1)
        self.module = module


class UnknownCommand(WireError):
    def __init__(self, command: str):
        super().__init__(f"Command not recognized: '{command}'", status=1)
        self.command = command


class BadArguments(WireError):
    def __init__(self, command: str, detail: str):
        super().__init__(f"Invalid arguments for '{command}': {detail}", status=2)


class NotImplementedVerb(WireError):
    """Module is loaded on this rig profile but the simulator does not model the verb yet."""

    def __init__(self, command: str):
        super().__init__(f"Simulator does not implement '{command}' (module loaded, verb unmodelled)",
                         status=3)
        self.command = command
