#!/usr/bin/env python3
"""Convert FOF claims to CNF format by stripping universal quantifiers.

Input:  00allconjm1.gz  — fof(prob__lid, claim, (! [X] : (literals))).
Output: 00allconjm11.gz — cnf(prob__lid, plain, (literals)).
"""

import gzip
import re
import sys

infile = sys.argv[1] if len(sys.argv) > 1 else 'vmin1/00allconjm1.gz'
outfile = sys.argv[2] if len(sys.argv) > 2 else 'vmin1/00allconjm11.gz'

# Pattern to match one or more chained universal quantifiers: ! [X0] : ! [X1] : ...
quant_re = re.compile(r'(?:\!\s*\[[^\]]*\]\s*:\s*)+')

n_in = 0
n_out = 0
n_quant = 0

with gzip.open(infile, 'rt') as fin, gzip.open(outfile, 'wt') as fout:
    for line in fin:
        n_in += 1
        # fof -> cnf, claim -> plain
        line = line.replace('fof(', 'cnf(', 1).replace(',claim,', ',plain,', 1)
        # Strip universal quantifiers
        new_line, nsubs = quant_re.subn('', line)
        if nsubs > 0:
            n_quant += 1
            line = new_line
        fout.write(line)
        n_out += 1

print(f"Converted {n_in} lines ({n_quant} had quantifiers)")
print(f"Written to {outfile}")
