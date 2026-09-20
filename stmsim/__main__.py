"""``python -m stmsim serve`` — run the software STM as a controller wire-protocol server."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .modules import build_dispatcher
from .paths import sessions_dir
from .physics.rig import RigProfile
from .physics.world import World
from .wire.server import WireServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stmsim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the simulator on loopback ports")
    s.add_argument("--profile", default="reference-stm")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--material", default="Au(111)")
    s.add_argument("--ports", default="6501,6502,6503,6504")
    s.add_argument("--session-dir", default=str(sessions_dir("serve")),
                   help="where Scan_Save writes .sxm (default $STM_BENCH_DATA/sessions/serve)")
    s.add_argument("--time-scale", type=float, default=1.0)
    s.add_argument("--contamination", type=float, default=0.0)
    s.add_argument("--approached", action="store_true", help="start in tunnelling instead of withdrawn")
    s.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    world = World(rig=RigProfile.load(a.profile), seed=a.seed, material=a.material,
                  time_scale=a.time_scale, session_dir=a.session_dir, contamination=a.contamination)
    if a.approached:
        world.coarse.coarse_gap_m = 0.6e-9 + world.surface_height_here()
        world.withdrawn = False
        world.zctrl_set(True)
    disp = build_dispatcher(world)
    ports = [int(p) for p in a.ports.split(",")]
    srv = WireServer(disp, ports=ports).start()
    print(f"stmsim serving profile={a.profile} seed={a.seed} ports={srv.bound_ports} "
          f"session_dir={world.session_dir} implemented_verbs={len(disp.implemented())}", flush=True)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    try:
        while not stop["flag"]:
            time.sleep(0.5)
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
