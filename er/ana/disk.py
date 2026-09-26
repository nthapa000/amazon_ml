"""Print how the compute node sees /scratch (mount, free space, quota) and test a 2 GB write there."""
import os
import subprocess

W = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
for c in ('hostname', 'mount | grep scratch', 'df -h /scratch', 'quota -s', f'dd if=/dev/zero of={W}/.qtest bs=64M count=32'):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print('$', c, '\n', r.stdout.strip(), r.stderr.strip()[-300:], flush=True)
if os.path.exists(f'{W}/.qtest'):
    os.remove(f'{W}/.qtest')
