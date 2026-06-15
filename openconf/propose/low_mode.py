"""Low mode following for conformer seeding.

Numerically evaluates Hessians via gradient finite differences and scans
minimized geometries along low-eigenvalue eigenvectors to generate structurally
diverse starting points. Most effective for macrocycles and correlated flexible
systems where independent torsion moves fail to sample collective soft motions.
"""

from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from ..relax import Minimizer

_N_RIGID_MODES: int = 6
_RIGID_BODY_GAP_WINDOW: int = _N_RIGID_MODES + 2
_DEFAULT_FD_STEP: float = 0.005
_DEFAULT_EIGENVALUE_THRESHOLD: float = 100.0
_DEFAULT_MAX_MODES: int = 5
_DEFAULT_SCAN_STEP_SIZE: float = 0.25
_DEFAULT_SCAN_ENERGY_THRESHOLD: float = 2390.0  # kcal/mol ≈ 10 000 kJ/mol (paper value)
_DEFAULT_SCAN_MAX_STEPS: int = 10


def _compute_hessian(
    mol: Chem.Mol,
    ff_props: object,
    conf_id: int,
    step: float = _DEFAULT_FD_STEP,
) -> np.ndarray:
    """Compute numerical Hessian via central differences of MMFF gradients.

    Perturbs each Cartesian coordinate of each atom by ±step, evaluates
    the MMFF gradient at each displaced geometry, and assembles the
    3N×3N Hessian matrix. Conformer positions are restored on exit.

    Args:
        mol: molecule containing the conformer
        ff_props: pre-prepared MMFFMoleculeProperties used to build force fields
            at displaced geometries
        conf_id: ID of the conformer at which to evaluate the Hessian;
            should be at or near a local minimum for meaningful low modes
        step: finite difference displacement in Å

    Returns:
        Symmetric 3N×3N Hessian matrix in kcal/mol/Å²
    """
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()
    n_dof = 3 * n_atoms
    pos0 = conf.GetPositions().copy()
    hessian = np.zeros((n_dof, n_dof))

    for i in range(n_dof):
        atom_i = i // 3
        coord_i = i % 3
        original = pos0[atom_i].copy()

        fwd = original.copy()
        fwd[coord_i] += step
        conf.SetAtomPosition(atom_i, fwd.tolist())
        ff_fwd = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=conf_id)
        if ff_fwd is None:
            conf.SetAtomPosition(atom_i, original.tolist())
            continue
        grad_plus = np.array(ff_fwd.CalcGrad())

        bwd = original.copy()
        bwd[coord_i] -= step
        conf.SetAtomPosition(atom_i, bwd.tolist())
        ff_bwd = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=conf_id)
        if ff_bwd is None:
            conf.SetAtomPosition(atom_i, original.tolist())
            continue
        grad_minus = np.array(ff_bwd.CalcGrad())

        conf.SetAtomPosition(atom_i, original.tolist())
        hessian[i] = (grad_plus - grad_minus) / (2.0 * step)

    return (hessian + hessian.T) * 0.5


def _select_low_modes(
    hessian: np.ndarray,
    eigenvalue_threshold: float,
    max_modes: int,
) -> np.ndarray:
    """Select low-frequency eigenvectors of a Hessian.

    Diagonalizes the Hessian, locates the rigid-body / conformational boundary
    via the largest eigenvalue gap within the first _RIGID_BODY_GAP_WINDOW
    entries, skips all modes up to that boundary, and returns eigenvectors
    whose eigenvalues are below eigenvalue_threshold, capped at max_modes.

    Using a gap rather than a fixed count of 6 handles linear molecules
    (only 5 rigid-body modes) and numerical drift when minimization is not
    fully converged.

    Args:
        hessian: symmetric 3N×3N Hessian matrix
        eigenvalue_threshold: upper eigenvalue bound (kcal/mol/Å²) for mode
            selection; modes at or above this value are excluded
        max_modes: maximum number of modes to return

    Returns:
        Array of shape (3N, k) where k ≤ max_modes; columns are unit
        eigenvectors in ascending eigenvalue order; shape (3N, 0) when
        no conformational modes satisfy the threshold
    """
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    window = min(_RIGID_BODY_GAP_WINDOW, len(eigenvalues) - 1)
    n_skip = int(np.argmax(np.diff(eigenvalues[: window + 1]))) + 1
    conf_vals = eigenvalues[n_skip:]
    conf_vecs = eigenvectors[:, n_skip:]

    mask = conf_vals < eigenvalue_threshold
    if not np.any(mask):
        return np.empty((hessian.shape[0], 0))
    return conf_vecs[:, mask][:, :max_modes]


def _scan_along_mode(
    mol: Chem.Mol,
    ff_props: object,
    start_conf_id: int,
    direction: np.ndarray,
    step_size: float,
    energy_threshold: float,
    max_steps: int,
) -> np.ndarray:
    """Scan from a minimized conformer along a unit direction vector.

    Takes discrete steps of step_size Å in direction, evaluating the MMFF
    energy after each step and stopping as soon as the per-step energy increase
    exceeds energy_threshold. Returns the positions from the last accepted step,
    or the starting positions when the first step already exceeds the threshold.

    A temporary conformer is created and removed; the start conformer is
    never modified.

    Args:
        mol: molecule to scan; receives a temporary conformer during the call
        ff_props: pre-prepared MMFFMoleculeProperties for energy evaluation
        start_conf_id: ID of the minimized starting conformer
        direction: unit displacement direction of shape (n_atoms, 3) in Å;
            each step moves the geometry by step_size × direction in 3N space
        step_size: distance (Å) to move in 3N Cartesian space per step
        energy_threshold: maximum allowed per-step energy increase (kcal/mol);
            scanning stops when ΔE in a single step exceeds this value
        max_steps: maximum number of steps regardless of energy criterion

    Returns:
        Positions of shape (n_atoms, 3) at the last accepted scan point
    """
    n_atoms = mol.GetNumAtoms()
    src_conf = mol.GetConformer(start_conf_id)
    start_pos = src_conf.GetPositions().copy()

    ff0 = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=start_conf_id)
    if ff0 is None:
        return start_pos
    prev_energy = float(ff0.CalcEnergy())

    working_conf = Chem.Conformer(src_conf)
    working_id = mol.AddConformer(working_conf, assignId=True)

    current_pos = start_pos.copy()
    accepted_pos = start_pos.copy()

    try:
        for _ in range(max_steps):
            current_pos = current_pos + step_size * direction
            working = mol.GetConformer(working_id)
            for i in range(n_atoms):
                working.SetAtomPosition(i, current_pos[i].tolist())

            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=working_id)
            if ff is None:
                break

            curr_energy = float(ff.CalcEnergy())
            if curr_energy - prev_energy > energy_threshold:
                break

            accepted_pos = current_pos.copy()
            prev_energy = curr_energy
    finally:
        mol.RemoveConformer(working_id)

    return accepted_pos


def generate_low_mode_seeds(
    mol: Chem.Mol,
    ff_props: object,
    conf_id: int,
    minimizer: "Minimizer",
    *,
    eigenvalue_threshold: float = _DEFAULT_EIGENVALUE_THRESHOLD,
    max_modes: int = _DEFAULT_MAX_MODES,
    scan_step_size: float = _DEFAULT_SCAN_STEP_SIZE,
    scan_energy_threshold: float = _DEFAULT_SCAN_ENERGY_THRESHOLD,
    scan_max_steps: int = _DEFAULT_SCAN_MAX_STEPS,
    fd_step: float = _DEFAULT_FD_STEP,
) -> list[tuple[int, float]]:
    """Generate conformers by scanning along low-frequency Hessian eigenvectors.

    Numerically evaluates the MMFF Hessian at the given minimized conformer,
    identifies eigenvectors with eigenvalues below eigenvalue_threshold (soft
    conformational modes), and for each such mode scans in both the positive and
    negative directions using discrete steps of scan_step_size Å. Scanning along
    a direction stops when the per-step energy increase exceeds scan_energy_threshold
    or scan_max_steps is reached. Each scan endpoint is then minimized to a local
    minimum.

    This mirrors the LMOD procedure of Kolossváry & Guida (JACS 1996): the scan
    naturally traverses soft conformational barriers and terminates at the onset
    of severe steric clashes, placing the starting geometry for minimization on
    the far side of a barrier.

    New conformers are added to mol in place. Callers are responsible for
    removing conformers that are not kept (e.g., pool rejects).

    Note:
        Hessian evaluation requires 6N MMFF force-field constructions where N
        is the atom count. This is the dominant cost; each scan step adds two
        further force-field constructions (one per direction).

    Args:
        mol: molecule containing the conformer; receives new conformers in place
        ff_props: pre-prepared MMFFMoleculeProperties used to build force fields
        conf_id: ID of a minimized conformer to compute modes from
        minimizer: minimizer applied to each scan endpoint
        eigenvalue_threshold: Hessian eigenvalue cutoff (kcal/mol/Å²); modes
            below this value are treated as conformationally soft
        max_modes: maximum number of low modes to scan per conformer
        scan_step_size: distance to advance per scan step in Å (3N Euclidean
            norm of the displacement vector); smaller values give finer
            resolution of the stopping point
        scan_energy_threshold: maximum per-step energy increase (kcal/mol)
            before scanning stops; the default (~2390 kcal/mol ≈ 10 000 kJ/mol)
            follows the paper and effectively allows the scan to pass through
            conformational barriers, stopping only at severe steric clashes
        scan_max_steps: upper bound on scan steps per direction regardless of
            the energy criterion; acts as a safety cap on total displacement
        fd_step: finite difference step size for numerical Hessian in Å

    Returns:
        Pairs of (conformer_id, energy_kcal_mol) for each successfully
        minimized scan endpoint; at most 2 results per mode (the two scan
        senses); empty when no low modes satisfy the threshold or all
        minimizations fail
    """
    hessian = _compute_hessian(mol, ff_props, conf_id, fd_step)
    low_vecs = _select_low_modes(hessian, eigenvalue_threshold, max_modes)
    if low_vecs.shape[1] == 0:
        return []

    n_atoms = mol.GetNumAtoms()
    start_pos = mol.GetConformer(conf_id).GetPositions().copy()

    results: list[tuple[int, float]] = []
    for col in range(low_vecs.shape[1]):
        direction = low_vecs[:, col].reshape(n_atoms, 3)

        for sign in (1.0, -1.0):
            final_pos = _scan_along_mode(
                mol,
                ff_props,
                conf_id,
                sign * direction,
                scan_step_size,
                scan_energy_threshold,
                scan_max_steps,
            )

            if np.allclose(final_pos, start_pos, atol=1e-8):
                continue

            new_conf = Chem.Conformer(mol.GetConformer(conf_id))
            new_conf_id = mol.AddConformer(new_conf, assignId=True)
            displaced = mol.GetConformer(new_conf_id)
            for i in range(n_atoms):
                displaced.SetAtomPosition(i, final_pos[i].tolist())

            energy = minimizer.minimize(mol, new_conf_id)
            if not np.isfinite(energy):
                mol.RemoveConformer(new_conf_id)
                continue

            results.append((new_conf_id, energy))

    return results
