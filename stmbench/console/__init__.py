"""The benchmark instrument in a web page (``stmbench.cli console``): for a person at the instrument and
for an agent that works through a screen. See :mod:`stmbench.console.server`."""
from .server import Console, make_server, serve

__all__ = ["Console", "make_server", "serve"]
