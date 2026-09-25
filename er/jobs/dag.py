#!/usr/bin/env python3
"""Tiny DAG runner for SLURM (login node, stdlib only: sleeps + sbatch/sacct).

Spec file, one job per line:   name | dep1 dep2 ... | sbatch command (must print the job id: use --parsable)
A job is submitted when all its dependencies have COMPLETED and the user has < MAXJ jobs in the queue.
If a job fails, everything that depends on it (directly or indirectly) is skipped; independent jobs continue.
State is logged to logs/dag.log.   usage: python3 jobs/dag.py jobs/final.dag
"""
import subprocess
import sys
import time

MAXJ = 3
spec, log = sys.argv[1], 'logs/dag.log'


def say(msg):
    with open(log, 'a') as f:
        f.write(time.strftime('%H:%M ') + msg + '\n')


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


jobs = {}
for ln in open(spec):
    if ln.strip() and not ln.startswith('#'):
        name, deps, cmd = [x.strip() for x in ln.split('|', 2)]
        jobs[name] = dict(deps=deps.split(), cmd=cmd, id=None, state='WAIT')
say(f'start {spec}: {len(jobs)} jobs')
while any(j['state'] in ('WAIT', 'RUN') for j in jobs.values()):
    for n, j in jobs.items():                                    # poll running jobs
        if j['state'] == 'RUN':
            st = sh(f"sacct -j {j['id']} -n -X --format=State").split()
            st = st[0] if st else 'PENDING'
            if st == 'COMPLETED':
                j['state'] = 'DONE'
                say(f'{n} ({j["id"]}) COMPLETED')
            elif st.startswith(('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_ME', 'NODE_FAIL')):
                j['state'] = 'FAIL'
                say(f'{n} ({j["id"]}) {st}')
    changed = True
    while changed:                                               # propagate failures
        changed = False
        for n, j in jobs.items():
            if j['state'] == 'WAIT' and any(jobs[d]['state'] in ('FAIL', 'SKIP') for d in j['deps']):
                j['state'], changed = 'SKIP', True
                say(f'{n} SKIPPED (dependency failed)')
    for n, j in jobs.items():                                    # submit ready jobs
        if j['state'] == 'WAIT' and all(jobs[d]['state'] == 'DONE' for d in j['deps']):
            if int(sh('squeue -u $USER -h | wc -l') or 0) >= MAXJ:
                break
            out = sh(j['cmd'])
            if out.isdigit():
                j['id'], j['state'] = out, 'RUN'
                say(f'{n} submitted {out}')
            else:
                j['state'] = 'FAIL'
                say(f'{n} submit failed: {out[:200]}')
    time.sleep(30)
say('dag finished: ' + ' '.join(f'{n}={j["state"]}' for n, j in jobs.items()))
