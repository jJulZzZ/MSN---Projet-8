"""Deterministic checks for the double-well PIMC notebook (reduced units).

Energies are divided by V0, positions by a, and theta = k_B T / V0.
The transfer matrix integrates the same primitive finite-P action as the MC.
"""

import base64
import json
import re
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal


def classical_reference(theta=0.4, half_width=2.5, points=2001):
    y = np.linspace(-half_width, half_width, points)
    potential = (y**2 - 1)**2
    density = np.exp(-potential / theta)
    density /= np.trapezoid(density, y)
    return {
        "y": y, "density": density,
        "potential_energy": float(np.trapezoid(potential * density, y)),
    }


def quantum_reference(C, theta, half_width=4.0, points=4001, states=32):
    """Second-order finite differences with Dirichlet endpoints at +/-L.

    scipy returns vectors with sum |u_i|^2=1, so psi_i=u_i/sqrt(h).
    Potential energy is <v>, not the total energy eigenvalue epsilon.
    """
    full_grid = np.linspace(-half_width, half_width, points)
    h = full_grid[1] - full_grid[0]
    y = full_grid[1:-1]
    potential = (y**2 - 1)**2
    energies, vectors = eigh_tridiagonal(
        2*C/h**2 + potential, np.full(len(y) - 1, -C/h**2),
        select="i", select_range=(0, states - 1),
    )
    weights = np.exp(-(energies - energies[0]) / theta)
    weights /= weights.sum()
    density = vectors**2 @ weights / h
    mean_potential = np.sum(vectors**2 * potential[:, None], axis=0) @ weights
    return {
        "y": y, "density": density, "levels": energies,
        "ground_energy": float(energies[0]),
        "potential_energy": float(mean_potential),
        "last_state_weight": float(weights[-1]),
    }


def transfer_reference(C, theta, P, edges=None, grid_points=141,
                       half_width=3.5, period=6.4, modes=80, output_points=256,
                       include_centroid=True):
    """Finite-P bead energy and centroid density by quadrature and Fourier inversion.

    The omitted free-particle prefactor cancels in normalized expectations.
    D(k)_ii=exp(i*k*y_i/P); chi(k)=Tr[(D(k)K)^P]/Tr[K^P].
    Bin averages use the analytic sinc integral of each Fourier mode.
    """
    if P < 1 or int(P) != P or modes >= output_points // 2:
        raise ValueError("P must be a positive integer; modes < output_points/2")
    if edges is None:
        edges = np.linspace(-2.5, 2.5, 101)
    edges = np.asarray(edges)
    if np.max(np.abs(edges)) >= period/2:
        raise ValueError("Fourier period must contain the histogram interval")
    y = np.linspace(-half_width, half_width, grid_points)
    h = y[1] - y[0]
    potential = (y**2 - 1)**2
    kernel = h * np.exp(
        -P*theta/(4*C) * (y[:, None] - y[None, :])**2
        -(potential[:, None] + potential[None, :])/(2*P*theta)
    )
    eigenvalues, vectors = eigh(kernel, check_finite=False)
    scale = eigenvalues[-1]
    kernel /= scale  # Avoid underflow when taking high matrix powers.
    weights = np.maximum(eigenvalues/scale, 0)**P
    trace = weights.sum()
    bead_density = vectors**2 @ (weights/trace) / h
    bead_energy = np.sum(bead_density * potential) * h
    if not include_centroid:
        return {"y": y, "bead_density": bead_density,
                "potential_energy": float(bead_energy)}

    frequencies = 2*np.pi*np.arange(1, modes + 1)/period
    characteristic = np.empty(modes, dtype=complex)
    for i, k in enumerate(frequencies):
        tilted = np.exp(1j*k*y[:, None]/P) * kernel
        characteristic[i] = np.trace(np.linalg.matrix_power(tilted, P))/trace
    q = (np.arange(output_points) - output_points//2)*period/output_points
    centroid_density = (
        1 + 2*np.real(np.exp(-1j*q[:, None]*frequencies) @ characteristic)
    )/period
    centers = (edges[:-1] + edges[1:])/2
    widths = np.diff(edges)
    bin_modes = np.exp(-1j*centers[:, None]*frequencies)
    bin_modes *= np.sinc(widths[:, None]*frequencies/(2*np.pi))
    centroid_bins = (1 + 2*np.real(bin_modes @ characteristic))/period
    middle = np.searchsorted(edges, 0.0, side="right") - 1
    barrier = max(0.0, float(-theta*np.log(centroid_bins[middle]/centroid_bins.max())))
    return {
        "y": y, "bead_density": bead_density,
        "potential_energy": float(bead_energy),
        "centroid_y": q, "centroid_density": centroid_density,
        "centroid_bins": centroid_bins, "edges": edges,
        "barrier": float(barrier),
        "fourier_tail": float(abs(characteristic[-1])),
    }


def saved_mc_results(notebook_path):
    """Read the saved printed MC estimates; do not rerun or infer new MC data.

    Runtime mc_comparisons in the notebook takes precedence after a fresh run.
    This fallback lets the deterministic section run in a fresh kernel alone.
    """
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    headings = {"## Question 6": (0.3, 0.4), "## Question 7": (0.3, 0.6),
                "## Question 6 bis": (0.9, 0.4), "## Question 7 bis": (0.9, 0.6)}
    results = {}
    case = None
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            case = headings.get(source.strip())
            continue
        if case is None:
            continue
        printed = "".join("".join(o.get("text", []))
                          for o in cell.get("outputs", [])
                          if o.get("output_type") == "stream" and o.get("name") == "stdout")
        pooled = re.search(r"Barrier from pooled density: ([0-9.eE+-]+)", printed)
        single = re.search(r"Effective barrier height: ([0-9.eE+-]+)", printed)
        match = pooled or single
        if match:
            sd = re.search(r"Run-to-run standard deviation: ([0-9.eE+-]+)", printed)
            results[case] = {
                "barrier": float(match.group(1)),
                "sd": float(sd.group(1)) if sd else None,
                "runs": 3 if pooled else 1,
                "source": "saved printed output (rounded)",
            }
        case = None
    return results


def write_report_tables(summary, destination):
    """Generate LaTeX tables from computed values and the measured MC summaries."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    quantum_rows = []
    barrier_rows = []
    for case in summary['cases']:
        C, theta = case['C'], case['theta']
        bias = 100*(case['P15_potential']/case['quantum_potential']-1)
        quantum_rows.append(
            f"{C:.1f} & {theta:.1f} & {case['ground_energy']:.6f} & "
            f"{case['quantum_potential']:.6f} & {case['P15_potential']:.6f} & {bias:+.2f} " + r"\\")
        mc = case['mc']
        if mc is not None:
            spread = '--' if mc['sd'] is None else f"{mc['sd']:.3f}"
            barrier_rows.append(
                f"{C:.1f} & {theta:.1f} & {mc['runs']} & {mc['barrier']:.4g} & {spread} & "
                f"{case['P15_barrier']:.6f} & {case['P60_barrier']:.6f} " + r"\\")
    quantum = r"""\begin{table}[H]
\centering\small
\begin{tabular}{rrrrrr}
\toprule
$C$ & $\theta$ & $E_0/V_0$ & $\langle V\rangle_q/V_0$ & $\langle V\rangle_{15}/V_0$ & Bias (\%)\\
\midrule
""" + '\n'.join(quantum_rows) + r"""
\bottomrule
\end{tabular}
\caption{Computed deterministic energies. The final column is the signed finite-$P=15$ deviation of potential energy from the quantum reference.}
\label{tab:quantum}
\end{table}
"""
    barriers = r"""\begin{table}[H]
\centering\small
\begin{tabular}{rrrrrrr}
\toprule
$C$ & $\theta$ & Runs & MC & Run SD & Reference $P=15$ & Reference $P=60$\\
\midrule
""" + '\n'.join(barrier_rows) + r"""
\bottomrule
\end{tabular}
\caption{Central-bin barriers divided by $V_0$. Lower-temperature MC values use pooled densities; higher-temperature values use one seed. Run SD is the standard deviation of individual run estimates; a dash means it was not estimated. The deterministic references match the MC bins.}
\label{tab:barriers}
\end{table}
"""
    (destination/'quantum_reference_table.tex').write_text(quantum, encoding='utf-8')
    (destination/'barrier_comparison_table.tex').write_text(barriers, encoding='utf-8')


def export_saved_mc_figures(notebook_path, destination):
    """Refresh the report's measured-data figures from saved notebook outputs."""
    targets = {
        '## Question 3': ['classical_tuning.png', 'classical_validation.png'],
        '## Question 4': ['mc_energy_C03.png'],
        '## Question 4 bis': ['mc_energy_C09.png'],
        '## Question 5': ['mc_distribution_C03.png'],
        '## Question 5 bis': ['mc_distribution_C09.png'],
    }
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    heading = None
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            # A short note between the two Q5 experiments is not a new section.
            if source.lstrip().startswith('#'):
                heading = source.strip()
            continue
        if heading not in targets:
            continue
        if heading in ['## Question 5', '## Question 5 bis'] and '500_000' not in source:
            continue
        images = [o['data']['image/png'] for o in cell.get('outputs', [])
                  if 'image/png' in o.get('data', {})]
        for name, data in zip(targets[heading], images):
            encoded = ''.join(data) if isinstance(data, list) else data
            (destination/name).write_bytes(base64.b64decode(encoded))
