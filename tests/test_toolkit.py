from dataclasses import replace
import csv

import numpy as np
import pytest
from scipy.integrate import quad

from matched_sensing import (
    CompilationError, Experiment, ReadoutError, audit, compile_bank, decode,
    export_bundle, load_bundle, plan_readout, read_counts, response, simulate, write_counts,
)
from matched_sensing.compiler import _objective


@pytest.fixture(scope="module")
def bank():
    e = Experiment((.8e6,1.05e6,1.31e6,1.72e6),(.5,-.3,0,.4),(.2,.4,.1,-.2),
                   100e3,t2_s=200e-6)
    return compile_bank(e,start_time_s=123e-9,seed=4)


def test_integrals_against_independent_quadrature():
    e = Experiment((0,.83e6,1.37e6),(1,0,0),(0,0,0),70e3)
    edges = np.array([.21,.57,.93,1.46])*1e-6
    duration, start = 2e-6, 137e-9
    endpoints = np.r_[0,edges,duration]
    expected = []
    for trig in (np.cos,np.sin):
        for f in e.frequencies_hz:
            value = sum((-1)**i*quad(lambda t:trig(2*np.pi*f*(t+start)),a,b,
                                     epsabs=1e-15)[0]
                        for i,(a,b) in enumerate(zip(endpoints[:-1],endpoints[1:])))
            expected.append(2*np.pi*e.coupling_hz*value)
    np.testing.assert_allclose(response(e,edges,duration,start),expected,rtol=1e-11,atol=1e-11)


def test_analytic_gap_gradient():
    e = Experiment((.6e6,1.31e6),(1,-.3),(.2,.4),80e3)
    n = 12; minimum = np.full(n+1,20e-9)
    evaluate = _objective(e,2.,8e-6,n,151e-9,minimum)
    x = np.random.default_rng(12).normal(0,.1,n+1)
    gradient = evaluate(x)[1]
    finite = []
    for i in range(n+1):
        delta = np.zeros(n+1);delta[i]=1e-6
        finite.append((evaluate(x+delta)[0]-evaluate(x-delta)[0])/2e-6)
    np.testing.assert_allclose(gradient,finite,rtol=2e-6,atol=2e-7)


def test_complete_response_preserves_observable_units(bank):
    e = bank.experiment
    rng = np.random.default_rng(8)
    for s in bank.sequences:
        check = audit(e,s)
        assert check["timing_valid"] and check["phase_error_bound"] <= bank.phase_tolerance
        for _ in range(8):
            a = rng.normal(size=len(e.weights));a *= e.amplitude_bound/np.linalg.norm(a)
            phase = np.array(s.phase_vector) @ a
            target = s.gain*(e.weights @ a)/e.observable_scale
            assert abs(phase-target) <= s.phase_error_bound+1e-10
    assert {s.gain for s in bank.sequences} == {.5,1.,2.,4.}


def test_clock_and_minimum_free_intervals():
    e = Experiment((.8e6,1.05e6),(1,.3),(.2,-.4),50e3,
                   min_gap_s=40e-9,edge_guard_s=80e-9,clock_s=2e-9)
    b = compile_bank(e,gains=(1.,),durations_s=(10e-6,),pulse_counts=(32,),
                     phase_tolerance=.01,seed=1)
    s = b.sequences[0];check=audit(e,s)
    assert check["timing_valid"] and check["phase_error_bound"] <= .01
    np.testing.assert_allclose(np.array(s.switch_times_s)/e.clock_s,
                               np.rint(np.array(s.switch_times_s)/e.clock_s),atol=1e-9)
    assert check["minimum_gap_s"] >= e.min_gap_s
    with pytest.raises(ValueError,match="multiples"):
        compile_bank(e,durations_s=(10e-6+0.5e-9,))


def test_infeasible_timing_is_rejected():
    e = Experiment((1e6,),(1,),(0,),100e3,min_gap_s=1e-6,edge_guard_s=1e-6)
    with pytest.raises(CompilationError):
        compile_bank(e,gains=(1,),durations_s=(1e-6,),pulse_counts=(4,),restarts=1)


def test_search_limits_bound_physical_sequences():
    e = Experiment((.8e6, 1.05e6), (1., .3), (.2, -.4), 100e3, clock_s=2e-9)
    # The caps need not be a clock-aligned duration or an even pulse count.
    b = compile_bank(e, max_gain=1.5, max_duration_s=10.001e-6, max_pulses=33,
                     phase_tolerance=.01, seed=1)
    assert max(s.gain for s in b.sequences) == 1.5
    for s in b.sequences:
        assert s.gain <= 1.5 and s.duration_s <= 10.001e-6
        assert len(s.switch_times_s) <= 33 and len(s.switch_times_s) % 2 == 0
        check = audit(e, s)
        assert check["timing_valid"] and check["phase_error_bound"] <= .01
    plan = plan_readout(b, precision=.1)
    result = decode(plan, simulate(plan, [.2, .1, -.1, .3], seed=12))
    assert result.interval[0] <= e.weights @ [.2, .1, -.1, .3] <= result.interval[1]


def test_zero_pulse_budget():
    e = Experiment((0.,), (1.,), (0.,), 100e3)
    duration = 1/(2*np.pi*e.coupling_hz)
    b = compile_bank(e, max_gain=1, max_duration_s=duration, max_pulses=0)
    assert {s.gain for s in b.sequences} == {.5, 1.}
    assert all(s.switch_times_s == () and s.duration_s <= duration for s in b.sequences)


@pytest.mark.parametrize("options", [
    {"max_gain": 0}, {"max_duration_s": 0}, {"max_pulses": -1},
    {"max_pulses": 2.5}, {"max_gain": 2, "gains": (1, 2)},
])
def test_invalid_search_limits(options):
    e = Experiment((1e6,), (1.,), (0.,), 100e3)
    with pytest.raises(ValueError):
        compile_bank(e, **options)


def test_dc_and_zero_pulses():
    e=Experiment((0,),(2,),(0,),100e3,amplitude_bound=2)
    duration=1/(2*np.pi*e.coupling_hz*e.amplitude_bound)
    b=compile_bank(e,gains=(1,),durations_s=(duration,),pulse_counts=(0,))
    assert b.sequences[0].switch_times_s==()
    p=plan_readout(b,precision=.2)
    assert abs(decode(p,simulate(p,[1.,0.],seed=2)).value-2.) <= .2


def test_unselected_input_modes_are_suppressed():
    e=Experiment((.8e6,1.05e6,1.31e6),(1,0,0),(0,0,0),100e3)
    b=compile_bank(e,gains=(1,),durations_s=(10e-6,),pulse_counts=(32,),seed=5)
    v=np.array(b.sequences[0].phase_vector)
    assert abs(v[0]-1) < b.phase_tolerance
    assert np.linalg.norm(v[1:]) < b.phase_tolerance


def test_start_phase_changes_the_physical_response(bank):
    s=bank.sequences[0]
    shifted=replace(s,start_time_s=s.start_time_s+173e-9)
    check=audit(bank.experiment,shifted)
    assert check["phase_error_bound"] > 100*bank.phase_tolerance


def test_readout_over_the_full_scalar_range(bank):
    plan=plan_readout(bank,precision=.08,confidence=.95)
    e=bank.experiment;u=e.weights/np.linalg.norm(e.weights)
    rng=np.random.default_rng(3);covered=0
    for i,x in enumerate(np.linspace(-1,1,101)):
        w=rng.normal(size=len(u));w-=u*(u@w);w/=np.linalg.norm(w)
        a=e.amplitude_bound*(x*u+np.sqrt(max(0,1-x*x))*w)
        truth=e.weights@a
        try:
            estimate=decode(plan,simulate(plan,a,seed=1000+i))
        except ReadoutError:
            continue
        covered+=estimate.interval[0] <= truth <= estimate.interval[1]
        assert estimate.interval[1]-estimate.interval[0] <= 2*plan.precision+1e-12
    assert covered >= 96


def test_dephasing_changes_gain_selection_and_cost(bank):
    quiet=plan_readout(bank,precision=.08)
    noisy_bank=replace(bank,experiment=replace(bank.experiment,t2_s=10e-6))
    noisy=plan_readout(noisy_bank,precision=.08)
    assert noisy.active_time_s > quiet.active_time_s
    assert noisy.bank.sequences[noisy.sequence_indices[-1]].gain <= bank.sequences[quiet.sequence_indices[-1]].gain


def test_bundle_and_raw_counts_roundtrip(bank,tmp_path):
    plan=plan_readout(bank,precision=.08)
    folder=export_bundle(plan,tmp_path/'experiment')
    restored=load_bundle(folder)
    a=np.random.default_rng(8).normal(size=len(bank.experiment.weights));a*=.7/np.linalg.norm(a)
    counts=simulate(plan,a,seed=9)
    path=folder/'measured_counts.csv';write_counts(path,plan,counts)
    loaded=read_counts(path,restored)
    assert loaded==counts and decode(plan,counts)==decode(restored,loaded)
    with pytest.raises(ValueError,match="integer"):
        read_counts(folder/'counts_template.csv',plan)
    with pytest.raises(FileExistsError):
        export_bundle(plan,folder)
    with pytest.raises(ValueError,match="separate"):
        write_counts(folder/'counts_template.csv',plan,counts)
    rows=list(csv.DictReader(path.open()))
    rows[0]['plan_id']='wrong experiment'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="mismatched"):
        read_counts(path,plan)
    with (folder/'pulses.csv').open('a') as stream:stream.write('edited\n')
    with pytest.raises(ValueError,match="changed"):
        load_bundle(folder)


def test_invalid_experiments_and_counts(bank):
    with pytest.raises(ValueError,match="distinct"):
        Experiment((1e6,1e6),(1,0),(0,0),100e3)
    with pytest.raises(ValueError,match="DC"):
        Experiment((0,),(0,),(1,),100e3)
    with pytest.raises(ValueError,match="visibility"):
        replace(bank.experiment,visibility=0)
    plan=plan_readout(bank,precision=.08)
    with pytest.raises(ReadoutError,match="every"):
        decode(plan,{})
    with pytest.raises(ValueError,match="amplitude_bound"):
        simulate(plan,10*np.ones(len(bank.experiment.weights)))
