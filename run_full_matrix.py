"""Run the full SALBP-CT experimental matrix.

The script is intentionally a thin orchestrator around the existing solver
entrypoints. It writes one normalized CSV row per attempted run, even when a
solver executable, Python package, license, or run itself fails.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from search_support import average_power_cap, upper_lower_power_cap


ROOT = Path(__file__).resolve().parent
# The orchestrator owns the experiment-wide cutoff. Commercial solvers receive
# a short serialization grace period because they also stop internally.
DEFAULT_TIMEOUT = 3600


@dataclass(frozen=True)
class Instance:
    name: str
    m: int
    salbp_cycle: int


SAT_SOLVERS = {
    "origin": "Minimize_makespan_origin.py",
    "sm": "Minimize_makespan_SM.py",
    "sm_tij": "Minimize_makespan_SM_Ti,j.py",
}

# E** (= E+ in paper) uses dedicated solver scripts that hard-code the arc set.
SAT_SOLVERS_ESTARSTAR = {
    "origin": "Minimize_makespan_origin_E2star.py",
    "sm": "Minimize_makespan_SM_E2star.py",
    "sm_tij": "Minimize_makespan_SM_Ti,j_E2star.py",
}

MAIN_FAMILIES = {
    "BOWMAN",
    "BUXEY",
    "GUNTHER",
    "HESKIA",
    "JACKSON",
    "JAESCHKE",
    "LUTZ2",
    "MANSOOR",
    "MERTENS",
    "MITCHELL",
    "ROSZIEG",
    "SAWYER",
    "WARNECKE",
}

ILP_SOLVERS = {
    "cplex_cp": {
        "Peak_UB_LB": "Minimize_makespan_cplex.py",
        "AVG_Peak": "Minimize_makespan_cplex.py",
    },
    "cplex_mip": {
        "Peak_UB_LB": "Minimize_makespan_cplex_mp.py",
        "AVG_Peak": "Minimize_makespan_cplex_mp.py",
    },
    "gurobi": {
        "Peak_UB_LB": "Minimize_makespan_gurobi.py",
        "AVG_Peak": "Minimize_makespan_gurobi.py",
    },
}

SOLVER_OUTPUTS = {
    "origin": "incremental_binary_merger.csv",
    "sm": "incremental_SM.csv",
    "sm_tij": "incremental_SM_Ti,j.csv",
    "cplex_cp": "result_cplex.csv",
    "cplex_mip": "result_cplex_mip.csv",
    "gurobi": "result_gurobi.csv",
}

THRESHOLDS = {
    "peak_ub_lb": "Peak_UB_LB",
    "avg_peak": "AVG_Peak",
}

CSV_FIELDS = [
    "run_id",
    "instance",
    "n",
    "m",
    "threshold",
    "solver",
    "edge_set",
    "search_policy",
    "cse17_enabled",
    "c_initial",
    "qmax",
    "best_cycle_time",
    "status",
    "timeout_seconds",
    "thread_policy",
    "seed_policy",
    "initial_lb",
    "initial_ub",
    "final_lb",
    "final_ub",
    "absolute_gap",
    "relative_gap_pct",
    "reference_value",
    "reference_type",
    "rpd_to_bks_pct",
    "variables",
    "constraints",
    "sat_calls",
    "elapsed_seconds",
    "time_to_first_incumbent",
    "time_to_best_incumbent",
    "time_to_proof",
    "termination_reason",
    "input_sha256",
    "power_sha256",
    "witness_path",
    "witness_valid",
    "stdout_tail",
]

EVENT_PREFIX = "SALBP_EVENT "


def load_instances() -> list[Instance]:
    source = ROOT / "run_origin.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "file_name" in names:
                values = ast.literal_eval(node.value)
                # The legacy list may repeat (family, m) under several SALBP
                # cycle times.  That cycle time is not an input to the revised
                # makespan problem, so those rows are the same configuration.
                unique: dict[tuple[str, int], Instance] = {}
                for row in values:
                    instance = Instance(str(row[0]), int(row[1]), int(row[2]))
                    unique.setdefault((instance.name.upper(), instance.m), instance)
                return list(unique.values())
    raise RuntimeError("Could not find file_name benchmark list in run_origin.py")


def read_instance_data(name: str) -> tuple[list[int], list[int]]:
    data_path = ROOT / "data" / f"{name}.IN2"
    power_path = ROOT / "task_power" / f"{name}.txt"
    lines = [line.strip() for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = int(lines[0])
    times = [int(value) for value in lines[1 : n + 1]]
    powers = [int(line.strip()) for line in power_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return times, powers


def c_initial(times: list[int], m: int) -> int:
    return max(max(times), (2 * sum(times) + m - 1) // m)


def power_aware_lower_bound(times: list[int], powers: list[int], m: int, qmax: int) -> int:
    workload = (sum(times) + m - 1) // m
    weighted_energy = sum(time * power for time, power in zip(times, powers))
    energy = (weighted_energy + qmax - 1) // qmax
    return max(max(times), workload, energy)


def qmax_for(threshold_dir: str, powers: list[int], m: int) -> int:
    if threshold_dir == "Peak_UB_LB":
        return upper_lower_power_cap(powers, m)
    return average_power_cap(powers, m)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_last(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text)
    return str(matches[-1]) if matches else ""


def parse_solver_output(stdout: str, timed_out: bool, failed: bool) -> dict[str, str]:
    reported_status = parse_last(r"Status:\s*([A-Za-z0-9_]+)", stdout)
    if timed_out:
        status = "TIMEOUT"
    elif failed:
        status = "FAILED"
    elif reported_status:
        status = reported_status
    elif "Optimal makespan:" in stdout or "Optimal solution found" in stdout or "No better solution found" in stdout:
        status = "Optimal"
    elif "Timeout" in stdout:
        status = "TIMEOUT"
    elif "No solution found" in stdout:
        status = "No solution"
    else:
        status = "Finished"

    best = parse_last(r"Optimal makespan:\s*([0-9.]+)", stdout)
    if not best:
        best = parse_last(r"New makespan:\s*([0-9.]+)", stdout)
    if not best:
        best = parse_last(r"Makespan\s*=\s*([0-9.]+)", stdout)

    return {
        "status": status,
        "best_cycle_time": best,
        "variables": "",
        "constraints": "",
        "sat_calls": "",
    }


def parse_events(stdout: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in stdout.splitlines():
        if not line.startswith(EVENT_PREFIX):
            continue
        try:
            event = json.loads(line[len(EVENT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def event_summary(events: list[dict[str, object]], parsed: dict[str, str]) -> dict[str, str]:
    lower_bounds = [
        int(event["lower_bound"])
        for event in events
        if event.get("lower_bound") is not None
    ]
    for event in events:
        if event.get("event") == "FEASIBILITY_RESULT" and event.get("result") == "UNSAT":
            lower_bounds.append(int(event["horizon"]) + 1)
        if event.get("event") == "IMPROVEMENT_RESULT" and event.get("result") == "UNSAT":
            lower_bounds.append(int(event["target"]) + 1)

    incumbent_events = [
        event for event in events
        if event.get("event") in {"SAFE_INCUMBENT", "INCUMBENT"}
        and event.get("incumbent") is not None
    ]
    incumbent_values = [int(event["incumbent"]) for event in incumbent_events]
    if parsed.get("best_cycle_time"):
        try:
            incumbent_values.append(int(float(parsed["best_cycle_time"])))
        except ValueError:
            pass

    initial_lb = lower_bounds[0] if lower_bounds else None
    final_lb = max(lower_bounds) if lower_bounds else None
    initial_ub = int(incumbent_events[0]["incumbent"]) if incumbent_events else None
    final_ub = min(incumbent_values) if incumbent_values else None
    if parsed.get("status", "").lower() == "optimal" and final_ub is not None:
        final_lb = final_ub

    absolute_gap = final_ub - final_lb if final_ub is not None and final_lb is not None else None
    relative_gap = (
        100.0 * absolute_gap / final_ub
        if absolute_gap is not None and final_ub not in {None, 0}
        else None
    )
    first_time = float(incumbent_events[0]["elapsed"]) if incumbent_events else None
    best_time = None
    if final_ub is not None:
        matching = [event for event in incumbent_events if int(event["incumbent"]) == final_ub]
        if matching:
            best_time = float(matching[0]["elapsed"])
    proof_events = [
        event for event in events
        if event.get("event") == "IMPROVEMENT_RESULT" and event.get("result") == "UNSAT"
    ]
    proof_time = float(proof_events[-1]["elapsed"]) if proof_events else None

    return {
        "initial_lb": "" if initial_lb is None else str(initial_lb),
        "initial_ub": "" if initial_ub is None else str(initial_ub),
        "final_lb": "" if final_lb is None else str(final_lb),
        "final_ub": "" if final_ub is None else str(final_ub),
        "absolute_gap": "" if absolute_gap is None else str(absolute_gap),
        "relative_gap_pct": "" if relative_gap is None else f"{relative_gap:.9f}",
        "time_to_first_incumbent": "" if first_time is None else f"{first_time:.9f}",
        "time_to_best_incumbent": "" if best_time is None else f"{best_time:.9f}",
        "time_to_proof": "" if proof_time is None else f"{proof_time:.9f}",
    }


def append_events(path: Path, run_id: str, metadata: dict[str, str], events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps({"run_id": run_id, **metadata, **event}, sort_keys=True) + "\n")


def write_schedule_witness(
    directory: Path,
    instance: Instance,
    threshold: str,
    solver: str,
    edge_set: str,
    row: dict[str, str],
    stdout: str,
    times: list[int],
) -> Path | None:
    """Archive the last complete schedule printed by a solver entrypoint."""
    assignments: dict[int, dict[str, int]] = {}
    pattern = r"Task\s+(\d+)\s+assigned to machine\s+(\d+)\s+at time\s+(\d+)"
    for task, station, start in re.findall(pattern, stdout):
        task_id = int(task)
        assignments[task_id] = {
            "task": task_id,
            "station": int(station),
            "start": int(start),
        }

    if len(assignments) != len(times) or not row.get("best_cycle_time", "").strip():
        return None
    try:
        cycle = int(float(row["best_cycle_time"]))
        qmax = int(float(row["qmax"]))
    except ValueError:
        return None

    witness: dict[str, object] = {
        "instance": instance.name,
        "m": instance.m,
        "threshold": threshold,
        "solver": solver,
        "edge_set": edge_set,
        "reported_cycle_time": cycle,
        "qmax": qmax,
        "assignments": [assignments[task] for task in sorted(assignments)],
    }
    if row.get("status", "").strip().lower() == "optimal":
        if "No better solution found" in stdout:
            witness["proof"] = {"status": "UNSAT", "bound": cycle - 1}
        elif cycle == max(times) and "Optimal solution found" in stdout:
            witness["proof"] = {"status": "LOWER_BOUND", "bound": cycle}
        else:
            best_bound = parse_last(r"Best bound:\s*([0-9.]+)", stdout)
            if best_bound and abs(float(best_bound) - cycle) <= 1e-6:
                witness["proof"] = {"status": "BOUND_MATCH", "bound": cycle}

    directory.mkdir(parents=True, exist_ok=True)
    if edge_set == "E**":
        edge_label = "estarstar"
    elif edge_set == "E*":
        edge_label = "estar"
    else:
        edge_label = "e"
    path = directory / f"{instance.name}__m{instance.m}__{threshold}__{solver}__{edge_label}.json"
    path.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def native_csv_path(threshold_dir: str, solver: str) -> Path:
    return ROOT / threshold_dir / "Output" / SOLVER_OUTPUTS[solver]


def latest_native_csv_row(
    threshold_dir: str,
    solver: str,
    instance: Instance,
    minimum_offset: int = 0,
) -> dict[str, str]:
    path = native_csv_path(threshold_dir, solver)
    if not path.exists():
        return {}

    latest: list[str] | None = None
    with path.open("rb") as handle:
        if minimum_offset <= path.stat().st_size:
            handle.seek(minimum_offset)
        payload = handle.read().decode("utf-8", errors="replace")
    for row in csv.reader(io.StringIO(payload)):
        if len(row) >= 4 and row[0] == instance.name and row[2] == str(instance.m):
            latest = row

    if not latest:
        return {}

    if solver in SAT_SOLVERS:
        # ins,n,m,c,makespan,peak,val,cons,sol,status,time_elapsed
        return {
            "best_cycle_time": latest[4] if len(latest) > 4 else "",
            "qmax": latest[5] if len(latest) > 5 else "",
            "variables": latest[6] if len(latest) > 6 else "",
            "constraints": latest[7] if len(latest) > 7 else "",
            "sat_calls": latest[8] if len(latest) > 8 else "",
            "status": latest[9] if len(latest) > 9 else "",
            "elapsed_seconds": latest[10] if len(latest) > 10 else "",
        }

    # New schema: instance,n,m,c,makespan,var,cons,elapsed,status,qmax,gap.
    # The first eight columns preserve compatibility with archived files.
    makespan = latest[4] if len(latest) > 4 else ""
    parsed = {
        "best_cycle_time": "" if "Timeout" in makespan else makespan,
        "status": "TIMEOUT" if "Timeout" in makespan else "Finished",
        "variables": latest[5] if len(latest) > 5 else "",
        "constraints": latest[6] if len(latest) > 6 else "",
        "elapsed_seconds": latest[7] if len(latest) > 7 else "",
    }
    if len(latest) > 8:
        parsed["status"] = latest[8]
    if len(latest) > 9:
        parsed["qmax"] = latest[9]
    if len(latest) > 10:
        parsed["gap"] = latest[10]
    return parsed


def run_command(cmd: list[str], cwd: Path, timeout: int) -> tuple[str, float, bool, bool]:
    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return completed.stdout or "", elapsed, False, completed.returncode != 0
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="ignore")
        return stdout, elapsed, True, False


def solver_command(
    threshold_dir: str,
    solver: str,
    instance: Instance,
    c0: int,
    edge_set: str,
) -> list[str]:
    if solver in SAT_SOLVERS:
        if edge_set == "E**":
            script = SAT_SOLVERS_ESTARSTAR[solver]
            # E** solvers hard-code the arc set; no edge_set argument needed
            return [
                sys.executable,
                "-u",
                str(ROOT / threshold_dir / script),
                instance.name,
                str(instance.m),
            ]
        return [
            sys.executable,
            "-u",
            str(ROOT / threshold_dir / SAT_SOLVERS[solver]),
            instance.name,
            str(instance.m),
            edge_set,
        ]
    script = ILP_SOLVERS[solver][threshold_dir]
    return [sys.executable, "-u", str(ROOT / threshold_dir / script), instance.name, str(instance.m), str(c0)]


def write_header_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        return
    with path.open(newline="", encoding="utf-8") as handle:
        existing = next(csv.reader(handle), [])
    if existing != CSV_FIELDS:
        raise RuntimeError(
            f"existing result header does not match the current schema: {path}; "
            "use a new output file"
        )


def append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def get_completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        try:
            reader = csv.DictReader(handle)
            for row in reader:
                if row and row.get("run_id"):
                    completed.add(row["run_id"].strip())
        except Exception:
            pass
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT.parent / "results" / "full_matrix_runs.csv"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--event-log",
        default=str(ROOT.parent / "results" / "full_matrix_events.jsonl"),
        help="append-only JSONL trajectory log",
    )
    parser.add_argument(
        "--instances",
        default="main",
        help="'main' (72 paper configurations), 'all', 'smoke', or comma-separated NAME and NAME:m selectors",
    )
    parser.add_argument("--solvers", default="origin,sm,sm_tij")
    parser.add_argument("--thresholds", default="peak_ub_lb,avg_peak")
    parser.add_argument(
        "--edge-sets",
        default="E",
        help="comma-separated SAT edge sets: E and/or E*; baselines run once with E",
    )
    parser.add_argument(
        "--witness-dir",
        default=str(ROOT.parent / "results" / "schedule_witnesses"),
        help="directory for JSON schedule witnesses; use --no-witnesses to disable",
    )
    parser.add_argument("--no-witnesses", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip run_ids already recorded in result CSV")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    instances = load_instances()
    if args.instances == "main":
        instances = [item for item in instances if item.name in MAIN_FAMILIES]
    elif args.instances == "smoke":
        instances = [item for item in instances if item.name in MAIN_FAMILIES][:2]
    elif args.instances != "all":
        selectors = {item.strip().upper() for item in args.instances.split(",") if item.strip()}
        names = {item for item in selectors if ":" not in item}
        exact: set[tuple[str, int]] = set()
        for item in selectors.difference(names):
            try:
                name, station_count = item.rsplit(":", maxsplit=1)
                exact.add((name, int(station_count)))
            except ValueError:
                parser.error(f"invalid instance selector: {item}; expected NAME or NAME:m")
        selected = [
            item for item in instances
            if item.name.upper() in names or (item.name.upper(), item.m) in exact
        ]
        matched = {item.name.upper() for item in selected}.intersection(names)
        matched.update(f"{item.name.upper()}:{item.m}" for item in selected if (item.name.upper(), item.m) in exact)
        missing = selectors.difference(matched)
        if missing:
            parser.error(f"unknown instances: {', '.join(sorted(missing))}")
        instances = selected

    solvers = [item.strip() for item in args.solvers.split(",") if item.strip()]
    thresholds = [item.strip() for item in args.thresholds.split(",") if item.strip()]
    unknown_solvers = set(solvers).difference(set(SAT_SOLVERS) | set(ILP_SOLVERS))
    unknown_thresholds = set(thresholds).difference(THRESHOLDS)
    if not solvers or unknown_solvers:
        parser.error(f"unknown solvers: {', '.join(sorted(unknown_solvers)) or '(none selected)'}")
    if not thresholds or unknown_thresholds:
        parser.error(f"unknown thresholds: {', '.join(sorted(unknown_thresholds)) or '(none selected)'}")
    edge_sets = [item.strip().upper() for item in args.edge_sets.split(",") if item.strip()]
    if not edge_sets or any(item not in {"E", "E*", "E**"} for item in edge_sets):
        parser.error("--edge-sets accepts only E, E*, and E**")
    result_path = Path(args.results).resolve()
    event_path = Path(args.event_log).resolve()
    if not args.dry_run:
        write_header_if_needed(result_path)

    completed_run_ids = get_completed_run_ids(result_path) if args.resume else set()
    if args.resume and completed_run_ids:
        print(f"Resuming run: found {len(completed_run_ids)} already completed runs in {result_path}")

    for instance in instances:
        times, powers = read_instance_data(instance.name)
        n = len(times)
        for threshold in thresholds:
            threshold_dir = THRESHOLDS[threshold]
            qmax = qmax_for(threshold_dir, powers, instance.m)
            initial_lb = power_aware_lower_bound(times, powers, instance.m, qmax)
            c0 = max(c_initial(times, instance.m), initial_lb)
            for solver in solvers:
                solver_edge_sets = edge_sets if solver in SAT_SOLVERS else ["E"]
                for edge_set in solver_edge_sets:
                    run_id = (
                        f"{instance.name}__m{instance.m}__{threshold}__{solver}__"
                        f"{'estarstar' if edge_set == 'E**' else ('estar' if edge_set == 'E*' else 'e')}"
                    )
                    if args.resume and run_id in completed_run_ids:
                        print(f"Skipping completed run_id: {run_id}")
                        continue
                    cmd = solver_command(threshold_dir, solver, instance, c0, edge_set)
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue
                    native_path = native_csv_path(threshold_dir, solver)
                    native_offset = native_path.stat().st_size if native_path.exists() else 0
                    watchdog = args.timeout if solver in SAT_SOLVERS else args.timeout + 60
                    stdout, elapsed, timed_out, failed = run_command(cmd, ROOT, watchdog)
                    parsed = parse_solver_output(stdout, timed_out, failed)
                    events = parse_events(stdout)
                    bounds = event_summary(events, parsed)
                    if not bounds["initial_lb"]:
                        bounds["initial_lb"] = str(initial_lb)
                    if not timed_out and not failed:
                        parsed.update({
                            key: value
                            for key, value in latest_native_csv_row(
                                threshold_dir, solver, instance, native_offset
                            ).items()
                            if value
                        })
                    witness_path = None
                    row = {
                        "run_id": run_id,
                        "instance": instance.name,
                        "n": str(n),
                        "m": str(instance.m),
                        "threshold": threshold,
                        "solver": solver,
                        "edge_set": edge_set,
                        "search_policy": "two_phase_feasible_first" if solver in SAT_SOLVERS else "native_optimizer",
                        "cse17_enabled": os.environ.get("SALBP_ENABLE_CSE17", "0") if solver in SAT_SOLVERS else "",
                        "c_initial": str(c0),
                        "qmax": parsed.get("qmax") or str(qmax),
                        "best_cycle_time": parsed["best_cycle_time"],
                        "status": parsed["status"],
                        "timeout_seconds": str(args.timeout),
                        "thread_policy": "1" if solver in SAT_SOLVERS else "solver_default_auto",
                        "seed_policy": "solver_default",
                        **bounds,
                        "reference_value": "",
                        "reference_type": "",
                        "rpd_to_bks_pct": "",
                        "variables": parsed["variables"],
                        "constraints": parsed["constraints"],
                        "sat_calls": parsed["sat_calls"],
                        "elapsed_seconds": parsed.get("elapsed_seconds") or f"{elapsed:.6f}",
                        "termination_reason": parsed["status"],
                        "input_sha256": sha256_file(ROOT / "data" / f"{instance.name}.IN2"),
                        "power_sha256": sha256_file(ROOT / "task_power" / f"{instance.name}.txt"),
                        "witness_path": "",
                        "witness_valid": "",
                        "stdout_tail": stdout[-500:].replace("\n", " "),
                    }
                    if not args.no_witnesses:
                        witness_path = write_schedule_witness(
                            Path(args.witness_dir).resolve(),
                            instance,
                            threshold,
                            solver,
                            edge_set,
                            row,
                            stdout,
                            times,
                        )
                        if witness_path is not None:
                            row["witness_path"] = str(witness_path)
                    append_events(
                        event_path,
                        run_id,
                        {
                            "instance": instance.name,
                            "m": str(instance.m),
                            "threshold": threshold,
                            "solver": solver,
                            "edge_set": edge_set,
                        },
                        events,
                    )
                    append_row(result_path, row)
                    print(
                        f"{instance.name} {threshold} {solver}/{edge_set}: "
                        f"{row['status']} {row['best_cycle_time']}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
