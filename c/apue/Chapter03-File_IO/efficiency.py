#!/usr/bin/python

import sys
import random

if len(sys.argv) != 2:
	print('Usage: %s <output-filename' % sys.argv[0])
	sys.exit(1)

nbytes = 516581760
written_bytes = ''.join((chr(random.randint(97,97 + 25))) for _ in range(nbytes))
with open(sys.argv[1],'w+') as fp:
	fp.write(written_bytes)
