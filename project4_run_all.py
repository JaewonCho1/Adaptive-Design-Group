"""
Project 4 -- one command regenerates every number on the site.

    python project4_run_all.py

1. project4_verify.py    verification ladder V1-V6       -> project4_verification.json, figures
2. project4_optimize.py  uniform design + optimizations  -> project4_optimization.json
3. project4_finalize.py  results file, figures, report   -> assets/project4_results.json,
                                                            project4_report.md
Runtime: about 4 minutes on a laptop.
"""

import project4_verify as V
import project4_optimize as O
import project4_finalize as F
import json

if __name__ == '__main__':
    V.v1()
    V.v2()
    conv = V.v3()
    V.v4()
    V.v5()
    checks, taylor, cost = V.v6()
    json.dump({'ladder': V.rows, 'convergence': conv, 'gradient_checks': checks,
               'taylor': taylor, 'cost': cost}, open(V.OUT, 'w'), indent=1)
    if not all(r['pass'] for r in V.rows):
        raise SystemExit('verification ladder failed -- not optimizing')
    O.main()
    F.main()
