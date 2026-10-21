"""Portable JSON/CSV experiment bundles; no device drivers or instrument calls."""

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

from .compiler import PulseBank, Sequence
from .experiment import Experiment
from .readout import ReadoutPlan, validate_plan
from .queries import QueryBank, QueryPlan, QuerySpace, validate_query_plan


def _validate(plan):
    (validate_query_plan if isinstance(plan, QueryPlan) else validate_plan)(plan)


def _payload(plan):
    if isinstance(plan, QueryPlan):
        return dict(schema_version=1, software_version="0.1.0", kind="queries",
                    experiment=plan.bank.experiment.to_dict(), query_space=asdict(plan.bank.space),
                    coordinates=[_payload(p) for p in plan.coordinate_plans],
                    readout=dict(precisions=plan.precisions, confidence=plan.confidence))
    return dict(schema_version=1, software_version="0.1.0", experiment=plan.bank.experiment.to_dict(),
                phase_tolerance=plan.bank.phase_tolerance,
                compilation_search=plan.bank.search,
                sequences=[asdict(s) for s in plan.bank.sequences],
                readout={k:v for k,v in asdict(plan).items() if k != "bank"})


def _plan_id(plan):
    return hashlib.sha256(json.dumps(_payload(plan),sort_keys=True).encode()).hexdigest()


def _csv(path, columns, rows):
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def export_bundle(plan, directory, *, overwrite=False):
    """Export pulse times, modulation intervals, acquisition rows and an empty counts template."""
    _validate(plan)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is nonempty; choose a new name or overwrite=True.")
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"bundle.json").write_text(json.dumps(_payload(plan),indent=2)+"\n")
    (directory/"compilation.json").write_text(json.dumps(plan.bank.attempts,indent=2)+"\n")
    pulses, intervals = [], []
    for i,s in enumerate(plan.bank.sequences):
        for j,t in enumerate(s.switch_times_s):
            pulses.append(dict(sequence_index=i,pulse_index=j,time_s=t,
                               reference_time_s=s.start_time_s+t,axis="X",angle_rad=3.141592653589793))
        edges = [0.0,*s.switch_times_s,s.duration_s]
        intervals.extend(dict(sequence_index=i,begin_s=a,end_s=b,modulation=1 if j%2 == 0 else -1)
                         for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])))
    _csv(directory/"pulses.csv",["sequence_index","pulse_index","time_s","reference_time_s","axis","angle_rad"],pulses)
    _csv(directory/"intervals.csv",["sequence_index","begin_s","end_s","modulation"],intervals)
    rows = [dict(plan_id=_plan_id(plan),**row) for row in plan.rows]
    columns = ["plan_id","row_id","sequence_index","basis","analysis_phase_rad","shots",
               "duration_s","contrast","start_time_s"]
    if isinstance(plan, QueryPlan):
        columns.append("coordinate_index")
    _csv(directory/"acquisition.csv",columns,rows)
    _csv(directory/"counts_template.csv",["plan_id","row_id","shots","plus"],
         [dict(plan_id=_plan_id(plan),row_id=r["row_id"],shots=r["shots"],plus="") for r in rows])
    protocol = """Prepare |+X>. Each sequence begins and ends in the +1 modulation frame.
Apply ideal instantaneous X(pi) pulses at the relative times in pulses.csv.
For X readout, P(+1)=(1+bias+contrast*cos(phase))/2; for Y replace cos by sin.
contrast=visibility*exp(-duration/T2); analysis_phase_rad uses sin(phase+angle).
Repeat with the specified tone phases at start_time_s relative to one phase origin.
Replaying a waveform at arbitrary start phases is NOT valid. Phase-lock/reset the
tones, or recompile at the actual reference start time. Trigger waits are not budgeted.
Map ideal X(pi), preparation and X/Y analysis to your calibrated hardware operations.
Finite pulse evolution, pulse errors, timing jitter and colored noise are not modeled.
All possible input tones and an upper bound on their joint amplitude norm are required.
Fill a COPY of counts_template.csv with +1 outcome counts; keep plan_id and shot counts.
"""
    files = ["bundle.json","compilation.json","pulses.csv","intervals.csv","acquisition.csv","counts_template.csv","RUN.txt"]
    if isinstance(plan, QueryPlan):
        space = plan.bank.space
        u_names, v_names = [f"u{i}" for i in range(space.rank)], [f"v{i}" for i in range(space.rank)]
        e = plan.bank.experiment
        labels = [(component, f) for component in ("cosine", "sine") for f in e.frequencies_hz]
        _csv(directory/"query_space.csv", ["fourier_index","component","frequency_hz",*u_names],
             [dict(fourier_index=i, component=c, frequency_hz=f, **dict(zip(u_names, u)))
              for i, ((c, f), u) in enumerate(zip(labels, space.basis))])
        _csv(directory/"query_coefficients.csv", ["query_index","kappa","residual_bound",*v_names],
             [dict(query_index=j, kappa=space.kappa, residual_bound=b, **dict(zip(v_names, v)))
              for j, (v, b) in enumerate(zip(space.coefficients, space.residual_bounds))])
        files += ["query_space.csv", "query_coefficients.csv"]
        protocol += """Acquisition row q{i}_stage{j}_{basis} uses the matched waveform for coordinate i.
sequence_index is global across all coordinate banks; indices are zero-based.
query_space.csv gives U; query_coefficients.csv gives kappa and each v_j.
decode_queries reconstructs all F_j = kappa * v_j @ y from these shared counts.
"""
        protocol += "Confidence is simultaneous over all queries; numerical rank residuals are included.\n"
    (directory/"RUN.txt").write_text(protocol)
    manifest = dict(schema_version=1,plan_id=_plan_id(plan),
                    sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files})
    if isinstance(plan, QueryPlan):
        manifest["kind"] = "queries"
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return directory


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    expected = {"bundle.json","compilation.json","pulses.csv","intervals.csv","acquisition.csv","counts_template.csv","RUN.txt"}
    if manifest.get("kind") == "queries":
        expected |= {"query_space.csv", "query_coefficients.csv"}
    elif "kind" in manifest:
        raise ValueError("Unknown bundle kind.")
    if manifest.get("schema_version") != 1 or set(manifest.get("sha256",{})) != expected:
        raise ValueError("Unknown or incomplete bundle manifest.")
    for name,digest in manifest["sha256"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundle file changed: {name}")
    data = json.loads((directory/"bundle.json").read_text())
    if data.get("kind") != manifest.get("kind"):
        raise ValueError("Bundle kind mismatch.")
    plan = _from_payload(data)
    _validate(plan)
    if _plan_id(plan) != manifest["plan_id"]:
        raise ValueError("Bundle identity mismatch.")
    return plan


def _from_payload(data):
    if data.get("schema_version") != 1:
        raise ValueError("Unknown bundle schema.")
    if data.get("kind") == "queries":
        s = data["query_space"]
        space = QuerySpace(tuple(map(tuple, s["basis"])), tuple(map(tuple, s["coefficients"])),
                           s["kappa"], tuple(s["residual_bounds"]), tuple(s["singular_values"]), s["rank_rtol"])
        children = tuple(_from_payload(child) for child in data["coordinates"])
        if any(not isinstance(child, ReadoutPlan) for child in children):
            raise ValueError("Query coordinates must have scalar readout plans.")
        bank = QueryBank(Experiment(**data["experiment"]), space, tuple(p.bank for p in children))
        return QueryPlan(bank, children, tuple(data["readout"]["precisions"]), data["readout"]["confidence"])
    if "kind" in data:
        raise ValueError("Unknown bundle kind.")
    sequences = tuple(Sequence(**{**s,"switch_times_s":tuple(s["switch_times_s"]),
                                  "phase_vector":tuple(s["phase_vector"])}) for s in data["sequences"])
    bank = PulseBank(Experiment(**data["experiment"]),sequences,data["phase_tolerance"],
                     search=data["compilation_search"])
    readout = data["readout"]
    plan = ReadoutPlan(bank,tuple(readout["sequence_indices"]),tuple(readout["shots_per_basis"]),
                       tuple(readout["phase_radii"]),readout["precision"],readout["confidence"],readout["optimize"])
    return plan


def write_counts(path, plan, positive_counts):
    """Write raw +1 counts; this never changes the blank template or pulse bundle."""
    if Path(path).name in {"counts_template.csv","pulses.csv","intervals.csv","acquisition.csv",
                          "query_space.csv","query_coefficients.csv","bundle.json","manifest.json","compilation.json","RUN.txt"}:
        raise ValueError("Save counts in a separate file, preserving the exported plan files.")
    rows = plan.rows
    if set(positive_counts) != {r["row_id"] for r in rows}:
        raise ValueError("Counts must match the acquisition rows.")
    from numbers import Integral
    for row in rows:
        k = positive_counts[row["row_id"]]
        if isinstance(k,bool) or not isinstance(k,Integral) or not 0 <= k <= row["shots"]:
            raise ValueError(f"Invalid positive count for {row['row_id']}.")
    _csv(path,["plan_id","row_id","shots","plus"],
         [dict(plan_id=_plan_id(plan),row_id=r["row_id"],shots=r["shots"],plus=positive_counts[r["row_id"]]) for r in rows])


def read_counts(path, plan):
    """Require the matching plan ID, exact shot counts, and complete integer data."""
    expected = {r["row_id"]:r for r in plan.rows}
    result = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["plan_id","row_id","shots","plus"]:
            raise ValueError("Use the counts_template.csv columns unchanged.")
        for row in reader:
            key = row["row_id"]
            if key not in expected or key in result or row["plan_id"] != _plan_id(plan):
                raise ValueError("Duplicate, unknown, or mismatched acquisition row.")
            try:
                shots, plus = int(row["shots"]), int(row["plus"])
            except (ValueError,TypeError) as exc:
                raise ValueError("Fill every plus cell with an integer outcome count.") from exc
            if shots != expected[key]["shots"] or not 0 <= plus <= shots:
                raise ValueError("Shot totals must match the plan and 0 <= plus <= shots.")
            result[key] = plus
    if set(result) != set(expected):
        raise ValueError("Missing acquisition rows.")
    return result
