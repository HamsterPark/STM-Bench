# Sources and external components

The project license is in [LICENSE](LICENSE). It does not grant rights to external
software or private datasets. This file records the sources identified during
release preparation on 2026-09-20; it is not a replacement for their licenses.

## Controller protocol metadata

`stmsim/spec/command_registry.json` is the simulator's locally maintained interface
registry. It retains the established command names and binary field formats needed
by implemented handlers, compatibility aliases and selected error probes. It does
not contain client function implementations or the external runtime's call-site
inventory. `stmsim/spec/extract.py` exports this local registry and does not read an
external client checkout.

The wire contracts have a history: earlier versions of this project used a larger
catalog extracted from a controller client identified as `nanonis_spm` version 1.0.9 and MAST's
protocol patches. This version removes that full catalog and its extraction code;
it preserves the applicable interface formats for compatibility. This is a change
in scope and maintenance, not a claim that those protocol conventions were invented
here or that the implementation was developed in a clean room.

The original client implementation is not bundled. Its package carries its own
copyright and distribution terms; the project's MIT license does not relicense
that external software. A compatible name or data layout is an interface contract,
not an implementation of the corresponding instrument operation.

## MAST and calibration data

MAST is a separate runtime and is not bundled here. Its availability and terms must
be checked independently. The repository includes references to MAST interfaces and
calibration provenance; it does not include the private raw instrument corpus or
the historical episode ledgers cited in the design record.

## Python dependencies

Runtime and optional Python dependencies are declared in `pyproject.toml` and
installed separately. Their respective licenses apply to those packages.
