#!/usr/bin/env python3
"""
scale_stats.py

Compute min/max or mean/std for each feature from CSV and write a small header
that can be copied into device to set Scaler params.
Usage:
  python tools/scale_stats.py data.csv out_dir --mode minmax
"""
import numpy as np, sys, os, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv'); p.add_argument('out_dir'); p.add_argument('--mode', choices=['minmax','standard'], default='minmax')
    args = p.parse_args()
    data = np.loadtxt(args.csv, delimiter=',')
    X = data[:, :-2]  # adjust target slicing if needed
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == 'minmax':
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        fname = os.path.join(args.out_dir, 'scaler_minmax.h')
        with open(fname, 'w') as f:
            f.write('#pragma once\n#include <vector>\nstatic std::vector<float> SCALER_MIN = { ' + ', '.join(f'{v:.8f}f' for v in mins.tolist()) + ' };\n')
            f.write('static std::vector<float> SCALER_MAX = { ' + ', '.join(f'{v:.8f}f' for v in maxs.tolist()) + ' };\n')
    else:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        fname = os.path.join(args.out_dir, 'scaler_std.h')
        with open(fname, 'w') as f:
            f.write('#pragma once\n#include <vector>\nstatic std::vector<float> SCALER_MEAN = { ' + ', '.join(f'{v:.8f}f' for v in means.tolist()) + ' };\n')
            f.write('static std::vector<float> SCALER_STD = { ' + ', '.join(f'{v:.8f}f' for v in stds.tolist()) + ' };\n')
    print("Wrote scaler to", fname)

if __name__ == '__main__':
    main()
