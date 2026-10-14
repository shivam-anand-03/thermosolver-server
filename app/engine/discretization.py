"""Phase II-I — discretisation of the energy equation on a structured grid.

All three techniques produce the same semi-discrete form

    C · dT/dt + K · T = F          (free nodes)
    T_p = g_p                       (Dirichlet nodes: temperature / inflow faces)

with C the (lumped) heat capacity, K the diffusion + advection − β-source
operator plus convective (Robin) boundary terms, and F the q0 source, flux and
T∞ terms. The steady problem is K T = F; transient problems use the θ-scheme.

Governing equation (q''' = q0 + βT):

    ρc_p (∂T/∂t + u·∇T) = ∇·(k∇T) + q0 + βT

* conduction            u = 0, solid properties
* convection–diffusion  fluid properties, prescribed uniform u
* convection            fluid properties, k = 0 (pure advection)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .grid import FACE_AXIS, Grid, SparseBuilder, SparseMatrix
from .schema import FaceBC, SimulationSetup

DIRICHLET_TYPES = ("temperature", "inflow")


@dataclass
class PhysicalModel:
    """Everything the discretisation needs, in SI units, resolved from the setup."""

    dim: int
    physics: str
    k: float
    rho_cp: float
    velocity: tuple[float, ...]
    q0: float
    beta: float
    bcs: dict[str, FaceBC]
    lengths: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def from_setup(cls, setup: SimulationSetup) -> "PhysicalModel":
        pre = setup.preprocessor
        dim = pre.problem.dimension
        physics = pre.problem.physics
        mat = pre.solid if physics == "conduction" else pre.fluid
        k = 0.0 if physics == "convection" else mat.k
        vel = (0.0, 0.0, 0.0) if physics == "conduction" else (pre.velocity.u, pre.velocity.v, pre.velocity.w)
        g = pre.geometry
        bcs = {f: pre.boundaries.get(f, FaceBC()) for f in FACE_AXIS if FACE_AXIS[f][0] < dim}
        return cls(dim, physics, k, mat.rho * mat.cp, vel[:dim], pre.problem.source.q0, pre.problem.source.beta,
                   bcs, (g.length, g.width, g.height)[:dim], (g.nx, g.ny, g.nz)[:dim])

    @property
    def alpha(self) -> float:
        return self.k / self.rho_cp if self.rho_cp else 0.0


def robin(bc: FaceBC) -> tuple[float, float]:
    """Heat entering through a non-Dirichlet face written as  q''_in = b − a·T_s."""
    if bc.type == "heat_flux":
        return 0.0, bc.value
    if bc.type == "convection":
        return bc.h, bc.h * bc.t_inf
    return 0.0, 0.0  # insulated wall / outflow


@dataclass
class Discretized:
    method: str
    grid: Grid
    K: SparseMatrix  # rows of free nodes only
    F: np.ndarray
    C: np.ndarray
    dirichlet: np.ndarray  # bool mask
    g: np.ndarray  # Dirichlet values (0 elsewhere)
    notes: list[str] = field(default_factory=list)
    advection_used: str | None = None


def _dirichlet_vertex(grid: Grid, bcs: dict[str, FaceBC]) -> tuple[np.ndarray, np.ndarray]:
    """Nodes on temperature/inflow faces; edge and corner nodes take the mean of their faces."""
    total = np.zeros(grid.n)
    count = np.zeros(grid.n)
    ijk = grid.multi_index
    for face, bc in bcs.items():
        if bc.type not in DIRICHLET_TYPES:
            continue
        a, side = FACE_AXIS[face]
        on = ijk[:, a] == (0 if side == "min" else grid.counts[a] - 1)
        total[on] += bc.value
        count[on] += 1
    mask = count > 0
    g = np.zeros(grid.n)
    g[mask] = total[mask] / count[mask]
    return mask, g


def _advection_choice(scheme: str, rho_cp_u: float, k: float, h: float) -> str:
    if k == 0.0:
        return "upwind"
    if scheme == "hybrid":
        return "central" if abs(rho_cp_u) * h / k <= 2.0 else "upwind"
    return scheme


# ------------------------------------------------------------------ FDM
def discretize_fdm(m: PhysicalModel, scheme: str) -> Discretized:
    """Vertex-centred finite differences.

    Interior:  −k(T_{i−1} − 2T_i + T_{i+1})/h²  (central)  and  ρc_p u ∂T/∂x
    (central or first-order upwind). Flux / convective / insulated faces use a
    ghost node eliminated with the boundary condition (second-order).
    """
    grid = Grid(m.dim, m.lengths, m.counts, "vertex")
    dmask, g = _dirichlet_vertex(grid, m.bcs)
    B = SparseBuilder(grid.n)
    F = np.zeros(grid.n)
    C = np.where(dmask, 0.0, m.rho_cp)
    ijk_all = grid.multi_index
    used: set[str] = set()

    for p in np.nonzero(~dmask)[0].tolist():
        ijk = ijk_all[p]
        for a in range(m.dim):
            h, N, s = grid.spacing[a], grid.counts[a], grid.stride(a)
            i = int(ijk[a])
            kd = m.k / h**2
            c = m.rho_cp * m.velocity[a]
            at_min, at_max = i == 0, i == N - 1
            face = ("xyz"[a] + "_min") if at_min else ("xyz"[a] + "_max")
            ra, rb = robin(m.bcs[face]) if (at_min or at_max) else (0.0, 0.0)

            # --- diffusion
            if not (at_min or at_max):
                B.add(p, p, 2 * kd)
                B.add(p, p - s, -kd)
                B.add(p, p + s, -kd)
            elif m.k > 0:
                nb = p + s if at_min else p - s
                B.add(p, p, 2 * kd)
                B.add(p, nb, -2 * kd)
                B.add(p, p, 2 * ra / h)
                F[p] += 2 * rb / h

            # --- advection
            if c == 0.0:
                continue
            choice = _advection_choice(scheme, c, m.k, h)
            used.add(choice)
            if not (at_min or at_max):
                if choice == "central":
                    B.add(p, p + s, c / (2 * h))
                    B.add(p, p - s, -c / (2 * h))
                elif c > 0:
                    B.add(p, p, c / h)
                    B.add(p, p - s, -c / h)
                else:
                    B.add(p, p + s, c / h)
                    B.add(p, p, -c / h)
                continue
            upstream_inside = (at_max and c > 0) or (at_min and c < 0)
            if choice == "upwind" and upstream_inside:
                if at_max:
                    B.add(p, p, c / h)
                    B.add(p, p - s, -c / h)
                else:
                    B.add(p, p + s, c / h)
                    B.add(p, p, -c / h)
            elif m.k > 0:
                # ∂T/∂x at the face from the boundary condition: left −k T' = b − aT, right k T' = b − aT
                if at_min:
                    B.add(p, p, c * ra / m.k)
                    F[p] += c * rb / m.k
                else:
                    B.add(p, p, -c * ra / m.k)
                    F[p] -= c * rb / m.k

        B.add(p, p, -m.beta)
        F[p] += m.q0

    notes = ["Second-order central differences for diffusion; ghost nodes eliminated at flux/convective faces."]
    return Discretized("fdm", grid, B.build(), F, C, dmask, g, notes, _label(used))


# ------------------------------------------------------------------ FVM
def discretize_fvm(m: PhysicalModel, scheme: str) -> Discretized:
    """Cell-centred finite volumes: face fluxes balance the generation in each cell.

    Diffusive face conductance D = kA/h (kA/(h/2) at a boundary face); the
    advective face value uses central, upwind or hybrid interpolation. Boundary
    faces always use the upwind value for advection (the inflow temperature when
    fluid enters through a Dirichlet face).
    """
    grid = Grid(m.dim, m.lengths, m.counts, "cell")
    V = grid.cell_volume
    B = SparseBuilder(grid.n)
    F = np.zeros(grid.n)
    C = np.full(grid.n, m.rho_cp * V)
    ijk_all = grid.multi_index
    used: set[str] = set()
    warned_wall = False

    for p in range(grid.n):
        ijk = ijk_all[p]
        for a in range(m.dim):
            h, N, s = grid.spacing[a], grid.counts[a], grid.stride(a)
            A = V / h
            D = m.k * A / h
            Fl = m.rho_cp * m.velocity[a] * A
            i = int(ijk[a])
            for side, sign in (("min", -1.0), ("max", 1.0)):
                F_out = sign * Fl
                interior = (i > 0) if side == "min" else (i < N - 1)
                if interior:
                    nb = p - s if side == "min" else p + s
                    B.add(p, p, D)
                    B.add(p, nb, -D)
                    if Fl:
                        choice = _advection_choice(scheme, m.rho_cp * m.velocity[a], m.k, h) if D else "upwind"
                        used.add(choice)
                        if choice == "central":
                            B.add(p, p, F_out / 2)
                            B.add(p, nb, F_out / 2)
                        elif F_out > 0:
                            B.add(p, p, F_out)
                        else:
                            B.add(p, nb, F_out)
                    continue
                bc = m.bcs["xyz"[a] + "_" + side]
                if bc.type in DIRICHLET_TYPES:
                    Db = 2 * D
                    B.add(p, p, Db)
                    F[p] += Db * bc.value
                    if F_out > 0:
                        B.add(p, p, F_out)
                    elif F_out < 0:
                        F[p] -= F_out * bc.value
                    continue
                if bc.type == "convection":
                    U = bc.h if m.k == 0 else 1.0 / (1.0 / bc.h + (h / 2) / m.k) if bc.h > 0 else 0.0
                    B.add(p, p, U * A)
                    F[p] += U * A * bc.t_inf
                elif bc.type == "heat_flux":
                    F[p] += bc.value * A
                if F_out != 0:
                    if F_out < 0 and not warned_wall:
                        warned_wall = True
                    B.add(p, p, F_out)
        B.add(p, p, -m.beta * V)
        F[p] += m.q0 * V

    notes = ["Cell-centred control volumes; boundary faces at half a cell from the nearest node."]
    if warned_wall:
        notes.append("Fluid enters through a face that is not an inflow/temperature face; the face value was "
                     "taken from the adjacent cell.")
    return Discretized("fvm", grid, B.build(), F, C, np.zeros(grid.n, bool), np.zeros(grid.n), notes, _label(used))


# ------------------------------------------------------------------ FEM
def _reference_element(dim: int, h: tuple[float, ...], k: float, rho_cp_u: tuple[float, ...]):
    """Element matrices of a d-linear box element by 2-point Gauss quadrature."""
    corners = list(itertools.product((-1.0, 1.0), repeat=dim))  # local node a ↔ signs (x first)
    nloc = len(corners)
    gp = 1.0 / np.sqrt(3.0)
    detJ = float(np.prod([hi / 2 for hi in h]))
    Kd = np.zeros((nloc, nloc))
    Ka = np.zeros((nloc, nloc))
    Me = np.zeros((nloc, nloc))
    for xi in itertools.product((-gp, gp), repeat=dim):
        Nv = np.array([np.prod([(1 + s[d] * xi[d]) / 2 for d in range(dim)]) for s in corners])
        dN = np.zeros((nloc, dim))
        for a_, s in enumerate(corners):
            for d in range(dim):
                others = np.prod([(1 + s[e] * xi[e]) / 2 for e in range(dim) if e != d]) if dim > 1 else 1.0
                dN[a_, d] = s[d] / 2 * others * 2 / h[d]
        Kd += k * dN @ dN.T * detJ
        Ka += np.outer(Nv, dN @ np.array(rho_cp_u)) * detJ
        Me += np.outer(Nv, Nv) * detJ
    return corners, Kd, Ka, Me


def discretize_fem(m: PhysicalModel) -> Discretized:
    """Galerkin finite elements with linear (1-D), bilinear (2-D) or trilinear (3-D) elements.

    Weak form: ∫ k∇N·∇T + ∫ N ρc_p u·∇T − ∫ N βT = ∫ N q0 + ∮ N q''_in.
    Heat capacity and boundary flux/convection terms are lumped to the nodes.
    """
    grid = Grid(m.dim, m.lengths, m.counts, "vertex")
    dmask, g = _dirichlet_vertex(grid, m.bcs)
    corners, Kd, Ka, Me = _reference_element(m.dim, grid.spacing, m.k, tuple(m.rho_cp * u for u in m.velocity))
    Ke = Kd + Ka - m.beta * Me

    # global node numbers of every element (n_el, 2^d)
    el_counts = [N - 1 for N in grid.counts]
    el_ijk = np.indices(tuple(reversed(el_counts))).reshape(m.dim, -1)[::-1].T
    offsets = np.array([sum(int((s[d] + 1) / 2) * grid.stride(d) for d in range(m.dim)) for s in corners])
    base = sum(el_ijk[:, d] * grid.stride(d) for d in range(m.dim))
    conn = base[:, None] + offsets[None, :]
    rows = np.repeat(conn, conn.shape[1], axis=1).ravel()
    cols = np.tile(conn, (1, conn.shape[1])).ravel()
    vals = np.tile(Ke.ravel(), conn.shape[0])
    keep = ~dmask[rows]
    K = SparseMatrix.from_coo(grid.n, rows[keep], cols[keep], vals[keep])

    # lumped nodal volume and face areas
    ijk = grid.multi_index
    vol = np.ones(grid.n)
    for d in range(m.dim):
        end = (ijk[:, d] == 0) | (ijk[:, d] == grid.counts[d] - 1)
        vol *= np.where(end, grid.spacing[d] / 2, grid.spacing[d])
    F = np.where(dmask, 0.0, m.q0 * vol)
    C = np.where(dmask, 0.0, m.rho_cp * vol)

    extra = SparseBuilder(grid.n)
    for face, bc in m.bcs.items():
        if bc.type in DIRICHLET_TYPES:
            continue
        ra, rb = robin(bc)
        if ra == 0.0 and rb == 0.0:
            continue
        a, side = FACE_AXIS[face]
        on = (ijk[:, a] == (0 if side == "min" else grid.counts[a] - 1)) & ~dmask
        area = vol / (grid.spacing[a] / 2)  # nodal volume divided by the half-width normal to the face
        for p in np.nonzero(on)[0].tolist():
            extra.add(p, p, ra * area[p])
            F[p] += rb * area[p]
    K = K + extra.build()
    notes = ["Galerkin weak form, 2-point Gauss quadrature; lumped capacity and boundary terms."]
    adv = "galerkin" if any(m.velocity) else None
    return Discretized("fem", grid, K, F, C, dmask, g, notes, adv)


def _label(used: set[str]) -> str | None:
    if not used:
        return None
    return "upwind" if used == {"upwind"} else "central" if used == {"central"} else "hybrid"


def discretize(setup: SimulationSetup) -> tuple[PhysicalModel, Discretized]:
    m = PhysicalModel.from_setup(setup)
    method = setup.solver.discretization
    scheme = setup.solver.advection_scheme
    if method == "fdm":
        return m, discretize_fdm(m, scheme)
    if method == "fvm":
        return m, discretize_fvm(m, scheme)
    return m, discretize_fem(m)
