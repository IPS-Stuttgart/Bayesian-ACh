"""Independent SciPy cross-check of the JavaScript scientific primitives."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.special import digamma
from scipy.stats import chi2

root = Path(__file__).resolve().parent
program = r"""
const s=require('./supported_set.js');
const q=Array.from({length:60},(_,i)=>s.chiSquareQuantile(0.95,i+1));
const normal=s.randomSource(104395303), cases=[];
for(let i=0;i<500;i++){
 const n=2+i%5,z=Array.from({length:n},()=>normal());
 const a=Array.from({length:n},()=>normal()),b=Array.from({length:n},()=>normal());
 if(i%10===0)for(let j=0;j<n;j++)b[j]=2*a[j];
 cases.push({z,a,b,residual:s.coneResidual(z,a,b)});
}
console.log(JSON.stringify({q,cases,grid:s.transitionGrid().rows}));
"""
result = json.loads(subprocess.check_output(["node", "-e", program], cwd=root, text=True))
quantile_error = max(abs(np.array(result["q"]) - chi2.ppf(0.95, np.arange(1, 61))))
assert quantile_error < 1e-10, quantile_error
residual_error = 0.0
for case in result["cases"]:
    matrix = np.column_stack([case["a"], case["b"]])
    _, expected = nnls(matrix, case["z"])
    residual_error = max(residual_error, abs(expected**2 - case["residual"]))
assert residual_error < 1e-9, residual_error
grid_error = 0.0
index = 0
for q in [0.05, 0.15, 0.35, 0.65, 0.90]:
    for _shape in [0.5, 0.9]:
        for concentration in [2.0, 8.0, 32.0, 128.0]:
            for _reset in [0.05, 0.5, 0.95]:
                for _hazard in [0.01, 0.15]:
                    expected = (
                        -np.log(q) + digamma(concentration * q + 1) - digamma(concentration + 1)
                    )
                    grid_error = max(grid_error, abs(expected - result["grid"][index][4]))
                    index += 1
assert grid_error < 1e-11, grid_error
print(
    json.dumps(
        {
            "status": "passed",
            "chi_square_max_abs_error": quantile_error,
            "cone_residual_max_abs_error": residual_error,
            "information_gain_max_abs_error": grid_error,
            "quantile_checks": 60,
            "cone_checks": 500,
            "grid_checks": index,
        }
    )
)
