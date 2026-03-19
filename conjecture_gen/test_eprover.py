"""Quick test to debug E prover invocation."""
import subprocess
import tempfile
import os

EPROVER = os.path.expanduser('~/bin/eprover')
PROBLEM = 'problems/l100_fomodel0'

# Read and filter problem
with open(PROBLEM) as f:
    lines = f.readlines()
content = ''.join(line for line in lines if line.strip().startswith('cnf('))

print(f"Problem has {len(content.splitlines())} cnf lines")
print(f"First line: {content.splitlines()[0][:80]}...")

# Write temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as tmp:
    tmp.write(content)
    tmp_path = tmp.name
print(f"Temp file: {tmp_path}")

# Run E
cmd = [EPROVER, '--auto', '--cpu-limit=10', '-s', '--print-statistics', tmp_path]
print(f"Command: {' '.join(cmd)}")

proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
print(f"Return code: {proc.returncode}")
print(f"STDOUT ({len(proc.stdout)} chars):")
for line in proc.stdout.split('\n'):
    if 'status' in line.lower() or 'processed' in line.lower():
        print(f"  {line}")
if proc.stderr:
    print(f"STDERR: {proc.stderr[:200]}")

os.unlink(tmp_path)

# Now test with a conjecture added
conj = 'cnf(gen_001, axiom, (v1_xboole_0(esk1_0))).'
content2 = content + '\n' + conj + '\n'

with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as tmp:
    tmp.write(content2)
    tmp_path2 = tmp.name

proc2 = subprocess.run(
    [EPROVER, '--auto', '--cpu-limit=10', '-s', '--print-statistics', tmp_path2],
    capture_output=True, text=True, timeout=15,
)
print(f"\nWith conjecture added:")
print(f"Return code: {proc2.returncode}")
for line in proc2.stdout.split('\n'):
    if 'status' in line.lower() or 'processed' in line.lower():
        print(f"  {line}")

os.unlink(tmp_path2)
