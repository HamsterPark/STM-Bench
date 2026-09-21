"""Export selected, public replay evidence without copying private run ledgers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re


def export(replay_dir: Path, output_dir: Path) -> None:
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    output_dir.joinpath("scans").mkdir(exist_ok=True)
    evidence = {
        "schema_version": 1,
        "description": "Recorded scan renderings and event summaries for a post-episode narrative. Explanatory labels are not agent observations.",
        "difficulty": results["difficulty"],
        "seed": results["seed"],
        "trials": [],
    }
    for name in ["Astra", "Terra"]:
        result = next(m for m in results["models"] if m["model"] == name)
        filename = f"p4-easy-seed0-mode-h-{result['model_id']}.html"
        path = replay_dir / filename
        raw = path.read_bytes()
        match = re.search(r"window\.REPLAY\s*=\s*(.*?);\s*</script>", raw.decode("utf-8"), re.S)
        if match is None:
            raise ValueError(f"Missing replay data: {filename}")
        replay = json.loads(match.group(1))
        timeline = replay["timeline"]
        assert replay["meta"]["difficulty"] == results["difficulty"] == "easy"
        assert replay["meta"]["run_id"] == result["run_id"]
        assert sum(c["ok"] for c in timeline["claims"]) == result["claims_verified"]
        assert sum(m["n"] for m in timeline["atoms"]["moves"]) == result["recorded_atom_hops"]
        frames = []
        for frame in timeline["frames"]:
            image = base64.b64decode(frame["png"].split(",", 1)[1])
            image_path = f"scans/{name.lower()}-{frame['idx']:02d}.png"
            (output_dir / image_path).write_bytes(image)
            frames.append({
                "index": frame["idx"], "image": image_path,
                "sha256": hashlib.sha256(image).hexdigest(),
                "geom": frame["geom"], "sim_start": frame["sim0"],
                "sim_end": frame["sim1"], "complete": frame["complete"],
                "drift_nm": frame["drift_nm"], "tip_events": frame["tip_events"],
            })
        attempts = []
        for step in timeline["steps"]:
            if step["name"] == "MoveAtomTo":
                moves = [m for m in timeline["atoms"]["moves"]
                         if step["sim0"] <= m["sim"] <= step["sim1"]]
                attempts.append({"tool_step_index": step["i"], "arguments": step["args"],
                                 "summary_original": step["result"],
                                 "recorded_hops": sum(m["n"] for m in moves),
                                 "last_recorded_position_nm": [moves[-1]["x"], moves[-1]["y"]] if moves else None})
        trial = {
            "model": name, "model_id": result["model_id"], "run_id": result["run_id"],
            "episode_sha256": result["episode_sha256"],
            "source_replay": filename, "source_replay_sha256": hashlib.sha256(raw).hexdigest(),
            "verified_checks": result["claims_verified"], "checks_total": 2,
            "sim_instrument_seconds": result["sim_instrument_seconds"],
            "frames": frames, "attempts": attempts,
            "tip_events": [{"sim": t["sim"], "cause": t["cause"], "detail": t["detail"]}
                           for t in timeline["tip"] if t["cause"] is not None],
        }
        if name == "Astra":
            # These are image estimates stated in the recorded operator messages,
            # kept separate from the post-episode simulator positions above.
            trial["image_estimates"] = {
                "selected_start_nm": [-6.0045, -11.8242],
                "requested_target_nm": [-2.0045, -11.8242],
                "positions_by_frame": {
                    "2": [-6.0045, -11.8242], "3": [-5.3383, -11.8102],
                    "4": [-2.163, -11.9174], "5": [-2.1571, -11.9129],
                    "6": [-1.9649, -11.7423],
                },
                "displacement_by_attempt_nm": [0.67, 3.84, 3.84, 4.04],
                "largest_neighbour_shift_nm": 0.018,
                "localization_uncertainty_nm": 0.05,
                "source": "Recorded operator messages and original Astra transcript; rounded image estimates, not hidden coordinates.",
            }
            assert [a["recorded_hops"] for a in attempts] == [3, 16, 0, 3]
            assert not trial["tip_events"]
        else:
            trial["diagnostic_context"] = {
                "warning_frame": 3, "audit_reproduced_change_row": 119,
                "warning_row_source": "The retained analysis reply is truncated before the warning details. Row 119 comes from offline reproduction; the model's final report independently confirms receipt of a tip-change warning.",
                "tool_reported_quality": [None, 0, 0, 0], "quality_threshold": 0.3,
                "quality_source": "The retained ConditionTip driver preview records quality_history [0, 0, 0] after the three pulses. The baseline score was not retained; null is unknown, not zero.",
                "recovery_scan_indices": [4, 5, 6, 7],
                "post_episode_audit": "No corresponding tip event during frame 3. The stability warning was a false positive. The quality metric can return zero for these scans without establishing a bad tip; full original tool replies are not retained.",
                "audit_source": "Recorded events and the 2026-09-20/21 three-model development audit; diagnostic interpretation is retrospective.",
            }
            assert len(trial["tip_events"]) == 3 and not attempts
        assert all(f["complete"] and f["drift_nm"] == [0.0, 0.0] for f in frames)
        evidence["trials"].append(trial)
    (output_dir / "replay-evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Exported 13 original scan renderings; scores, events and zero drift checked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    export(args.replay_dir, args.output_dir)
