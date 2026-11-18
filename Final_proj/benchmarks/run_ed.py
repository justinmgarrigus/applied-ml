"""Generate a fresh small reference; training never calls this program."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('ising', 'fqhe'), required=True)
    parser.add_argument('--ne', type=int, required=True, help='Spins or electrons')
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    parser.add_argument('--v3', type=float, default=0., help='FQHE V3/V1; default pure V1')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh reference directory')
    root = Path(__file__).resolve().parent
    if args.system == 'fqhe':
        sys.path.insert(0, str(root.parent / 'shared'))
        from fqh_spring.references import build_reference, load_reference
        build_reference(args.ne, output, v3=args.v3, eigenpairs=8, max_dimension=20000)
        reference = load_reference(output)
        result = dict(system='fqhe', ne=args.ne,
                      ground_energy=float(reference['eigenvalues'][0]),
                      energy_unit='V1', validation=reference['validation'])
    else:
        if args.v3 != 0:
            parser.error('--v3 applies only to fqhe')
        path = root / 'ising' / 'ed_solver.py'
        spec = importlib.util.spec_from_file_location('ising_ed', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(sys.stderr):
            result = module.run(output, [args.ne], False, 8)
    print(json.dumps(dict(output=str(output), result=result), indent=2))


if __name__ == '__main__':
    main()
