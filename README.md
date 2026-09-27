# ThermoSolve API — `thermosolver-server`

The backend for **ThermoSolve**, an educational numerical solver for one-dimensional heat conduction in a plane slab. It is a FastAPI service that assembles the finite-difference system, solves it with a hand-written Gaussian elimination, and returns every intermediate quantity (Δx, each matrix row, each row operation, each back-substitution line) so the UI can show how the answer was produced.

It follows the finite-difference workflow used in ME46220:

```
INPUT → GOVERNING EQUATION → DISCRETISATION → BOUNDARY CONDITIONS → FD EQUATIONS
      → MATRIX ASSEMBLY → GAUSS ELIMINATION → BACKWARD SUBSTITUTION → VALIDATION → REPORT
```

The web front end lives in a separate repository, `thermosolver-ui`.

---

## 1. Features

| Area | What you get |
|---|---|
| Problems | Steady state (`d²T/dx² + q'''/k = 0`) and transient (`∂T/∂t = α ∂²T/∂x²`, optional `q'''`) |
| Mesh | L and N given (Δx = L/(N−1) computed), or a custom Δx; 2 – 2000 nodes (configurable) |
| Boundary conditions | Temperature (Dirichlet), heat flux, gradient (Neumann), adiabatic, independently at each end |
| Boundary scheme | First-order one-sided difference (default, matches the course) or second-order ghost node |
| Linear solver | Hand-written Gaussian elimination with partial pivoting and zero, near-zero and singular pivot detection. It never calls `numpy.linalg.solve` |
| Transient | Fully implicit (backward Euler); Δt given directly or computed from a Fourier number; uniform or custom initial field; output sampling |
| Validation | Residual `max|A·T − b|`, separate boundary residuals, finiteness, solution size, informational energy balance |
| Report | PDF with derivation, matrices, elimination steps, Matplotlib charts and validation (ReportLab) |
| Units | m / cm / mm; °C / K; SI internally |

## 2. Quick start

Requirements: Python 3.11+ (3.12 recommended).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

- Health check: http://127.0.0.1:8000/api/health
- Interactive docs: http://127.0.0.1:8000/docs

To use it with the UI, run `thermosolver-ui` with `npm run dev`; its dev server proxies `/api` to `http://127.0.0.1:8000`.

## 3. Configuration

Settings are read from environment variables prefixed `THERMOSOLVE_`, or from a `.env` file in this folder. Copy [.env.example](.env.example) to `.env` to override them.

| Variable | Default | Meaning |
|---|---|---|
| `THERMOSOLVE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated origins allowed to call the API. In production, set this to the UI's URL. |
| `THERMOSOLVE_MAX_NODES` | `2000` | Largest mesh accepted |
| `THERMOSOLVE_MAX_TIME_STEPS` | `20000` | Largest number of transient steps |
| `THERMOSOLVE_MAX_NODE_STEPS` | `20000000` | Cap on nodes × steps for one transient run |
| `THERMOSOLVE_RECORD_LIMIT` | `12` | Store every intermediate augmented matrix when N ≤ this |
| `THERMOSOLVE_MATRIX_PAYLOAD_LIMIT` | `100` | Send the dense matrix in the response when N ≤ this |

## 4. Governing equations

**Steady state**, constant k:

```
d/dx(k dT/dx) + q''' = 0      ⟹      d²T/dx² + q'''/k = 0
```

**Transient**:

```
∂T/∂t = α ∂²T/∂x² + q'''/(ρCp),      α = k/(ρCp)
```

Example: k = 0.25 W/m·K, ρ = 1300 kg/m³, Cp = 2000 J/kg·K → α = 9.6154 × 10⁻⁸ m²/s.

## 5. Numerical method

Uniform mesh: `Δx = L/(N−1)`, with xᵢ = (i−1)Δx for i = 1…N.

The second derivative at interior nodes uses the central difference scheme:

```
d²T/dx² |ᵢ ≈ (Tᵢ₊₁ − 2Tᵢ + Tᵢ₋₁) / Δx²
```

### Steady-state solver

Substituting the central difference and multiplying by −Δx²:

```
−Tᵢ₋₁ + 2Tᵢ − Tᵢ₊₁ = (q'''/k) Δx²          i = 2 … N−1
```

The two boundary rows complete an N×N tridiagonal system `[A]{T} = {b}`, which is solved once.

### Transient solver (fully implicit)

```
(Tᵢⁿ⁺¹ − Tᵢⁿ)/Δt = α (Tᵢ₊₁ⁿ⁺¹ − 2Tᵢⁿ⁺¹ + Tᵢ₋₁ⁿ⁺¹)/Δx²

−Fo·Tᵢ₋₁ⁿ⁺¹ + (1 + 2Fo)·Tᵢⁿ⁺¹ − Fo·Tᵢ₊₁ⁿ⁺¹ = Tᵢⁿ + Fo·(q'''/k)·Δx²,      Fo = αΔt/Δx²
```

For **Fo = 1** this becomes the −1, 3, −1 stencil `−Tᵢ₋₁ⁿ⁺¹ + 3Tᵢⁿ⁺¹ − Tᵢ₊₁ⁿ⁺¹ = Tᵢⁿ`, as in the MST solution.

> The general form keeps Fo on the off-diagonals. The shorthand `−Tᵢ₋₁ + (1+2Fo)Tᵢ − Tᵢ₊₁ = Tᵢⁿ` is only correct when Fo = 1, so ThermoSolve always uses the general form.

Each time step:
1. Build `b = c ⊙ Tⁿ + d` from the previous level.
2. Apply the boundary conditions (they are built into the rows of A and d).
3. Run forward elimination.
4. Run backward substitution.
5. Store Tⁿ⁺¹.
6. Repeat until the final step.

[A] does not change between steps, so the elimination multipliers are computed once and applied again to each new right-hand side. This gives the same result as eliminating from scratch each step. The first step is also solved with full recording, so the Algorithm tab can show it.

The fully implicit scheme is **unconditionally stable** for the diffusion equation. ThermoSolve therefore never labels Fo > 0.5 as unstable. It only warns when a very large Fo (> 10) may hurt accuracy.

Δt either comes from a target Fourier number, `Δt = Fo·Δx²/α` (Fo = 1 gives Δt = 10.40 s for the MST example), or is entered manually.

## 6. Gaussian elimination

The solver is in [app/solver/gaussian_elimination.py](app/solver/gaussian_elimination.py) (`GaussianEliminationSolver`).

1. Form the augmented matrix `[A | b]`.
2. For each column k:
   - **Partial pivoting**: choose the row p ≥ k with the largest `|a_pk|` and swap it into row k.
   - Reject zero pivots (singular matrix) and warn about near-zero pivots.
   - For each row i > k with `a_ik ≠ 0`, compute `factor = a_ik / a_kk` and apply `Rᵢ → Rᵢ − factor·Rₖ`.
   - Record the operation, e.g. `R2 → R2 + 1·R1`.
3. **Backward substitution**: `xₙ = bₙ/aₙₙ`, then `xᵢ = (bᵢ − Σⱼ>ᵢ aᵢⱼxⱼ)/aᵢᵢ`.

The solver returns the solution plus the initial augmented matrix, every pivot, row swap and row operation, the matrix after each step (stored when N ≤ 12), the upper-triangular system and every back-substitution line.

Rows that already have a zero below the pivot are skipped. For tridiagonal finite-difference systems this cuts the cost from O(N³) to O(N²), so N = 1000 solves in well under a second.

## 7. Boundary conditions

**Sign convention:** a positive heat flux q'' always means heat entering the slab. With q = −k dT/dx:

| Side | Heat flux in | Implied gradient |
|---|---|---|
| Left (x = 0) | `q''_in = −k dT/dx` | `dT/dx = −q''/k` |
| Right (x = L) | `q''_in = +k dT/dx` | `dT/dx = +q''/k` |

Adiabatic means q'' = 0, so dT/dx = 0. Gradient conditions prescribe dT/dx directly, in K/m (identical to °C/m).

Matrix rows:

| Type | First-order (default) | Ghost node (second order) |
|---|---|---|
| Temperature | `T₁ = T_left` → row `[1, 0, …]` | same |
| Derivative, left | `T₁ − T₂ = −g·Δx` | steady: `T₁ − T₂ = q'''Δx²/2k − g·Δx`; transient: `(1+2Fo)T₁ − 2Fo·T₂ = T₁ⁿ − 2Fo·g·Δx` |
| Derivative, right | `−T_{N−1} + T_N = g·Δx` | steady: `−T_{N−1} + T_N = q'''Δx²/2k + g·Δx`; transient: mirror of the left row |

The ghost-node rows are what you get from an energy balance on the half control volume at the boundary.

**Invalid combinations:** a steady problem with flux or gradient conditions at both ends is rejected before assembly, because the matrix would be singular.
- If the heat balance `q''in,L + q''in,R + q'''L ≠ 0`, the message says no steady solution exists.
- If the balance is zero, the message says the solution is only defined up to a constant, and asks you to fix a temperature.

Transient problems allow flux conditions at both ends.

## 8. Temperature units

Inputs and results use °C or K. Only temperature differences enter the derivative terms, and the equations are linear, so the system is assembled directly in the chosen temperature scale. The matrices then look exactly like hand calculations. The results view can switch the display unit, which converts absolute temperatures only. Lengths are converted to metres internally.

## 9. Example problems

The examples are served by `GET /api/examples` (and appear in the UI's **Load Example** menu):

| Example | Notes |
|---|---|
| **MST-1 steady**: L = 0.5 m, N = 5, k = 1, T(0) = 373 K, q''(L) = 1000 W/m² in | Reference example (see the discrepancy below) |
| **MST-1 transient**: L = 5 mm, N = 6, k = 0.25, ρ = 1300, Cp = 2000, T⁰ = 30 °C, adiabatic left, T(L) = 250 °C, Fo = 1 | Reproduces the MST first step |
| Wall with heat generation: q''' = 5×10⁵ W/m³, both faces 100 °C | Matches the analytic parabola exactly; T_max = 131.25 °C |
| Steel plate quench: 500 °C → faces at 25 °C | Transient cooling demo |

### Reference results and discrepancies

**Transient MST-1 matches.** With α = 9.6154×10⁻⁸ m²/s, Fo = 1 and Δt = 10.40 s, the first step gives:

| Node | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ThermoSolve (°C) | 36.47 | 36.47 | 42.94 | 62.35 | 114.12 | 250 |
| MST-1 printed | 36.47 | 36.47 | 42.94 | 62.36 | 114.12 | 250 |

The exact value of T₄ is 62.353, so the printed 62.36 is a rounding difference. The match requires the first-order adiabatic row T₁ = T₂. The ghost-node alternative gives 33.58, 35.37, 42.52, … instead. That identifies the course's boundary treatment, and it is why first order is the default.

**Steady MST-1 cannot be reproduced from its own equations.**

| Node | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ThermoSolve (q''' = 0) | 373 | 498 | 623 | 748 | 873 |
| MST-1 printed | 373 | 431.86 | 577.31 | 711.54 | 836.54 |

Findings:
- The printed last interval is consistent with the flux condition: T₅ − T₄ = 125 K = q''Δx/k.
- With q''' = 0 and constant k, the governing equation gives a **linear** profile, so every interval must be 125 K.
- The printed intervals (58.86, 145.45, 134.23, 125) cannot come from any single uniform stencil.
- Using the diagonal 2.015 that appears in the printed matrix gives 373, 471.87, 577.81, 692.43, 817.43. That is close to T₃, but not to the other nodes.

The document most likely contains an arithmetic or transcription slip in its intermediate matrix or RHS. ThermoSolve does **not** force these numbers. The UI labels the example "Reference Example" and shows the printed values next to the computed ones with their differences. The solver never reads the reference values.

## 10. API

Base URL: `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness |
| GET | `/api/examples` | Example catalogue, with reference values |
| POST | `/api/solve/steady` | Solve a steady problem |
| POST | `/api/solve/transient` | Solve a transient problem |
| POST | `/api/validate` | Validate inputs only: `{problem_type, steady \| transient}` |
| POST | `/api/report/pdf` | PDF report: `{problem_type, steady \| transient}` |

Steady request:

```json
{
  "length": 0.5, "length_unit": "m", "nodes": 5,
  "thermal_conductivity": 1.0, "heat_generation": 0,
  "left_boundary":  { "type": "temperature", "value": 373 },
  "right_boundary": { "type": "heat_flux",   "value": 1000 },
  "temperature_unit": "K", "boundary_scheme": "first_order"
}
```

Transient requests add `density`, `specific_heat`, `initial_condition` (`{type: "uniform", value}` or `{type: "custom", values: [...]}`), `time_step_mode` (`"fourier"` or `"manual"`), `fourier_number`, `time_step`, `num_steps` or `final_time`, and `output_interval`.

Response fields: `success`, `dx`, `x`, `nodes`, `solution`, `matrix` (dense when N ≤ 100), `bands`, `rhs`, `equations`, `gaussian` (steps, operations, back substitution), `validation`, `explanation`, `summary`, `warnings`, `execution_time_ms`. Transient responses also include `alpha`, `dt`, `fourier`, `times`, `snapshots`, `initial` and `first_step_solution`.

Errors come back as HTTP 422 with `{success: false, error, details, kind}`, for example "Length must be greater than zero." Stack traces are never exposed.

## 11. Project structure

```
app/
  main.py                    FastAPI app factory, CORS, error handlers
  config.py                  environment settings (THERMOSOLVE_*)
  models.py                  Pydantic request models
  api/routes.py              HTTP routes (all under /api)
  assets/logo.png            logo used in the PDF report
  solver/
    gaussian_elimination.py  GaussianEliminationSolver (pivoting, recording, replay)
    finite_difference.py     Mesh, MatrixAssembler, FiniteDifferenceSolver
    boundary_conditions.py   BoundaryCondition → matrix rows
    steady_state.py          SteadyStateSolver
    transient.py             TransientSolver
    validation.py            InputValidator, ValidationEngine
    explanation.py           deterministic step-by-step derivation
    results.py               SimulationResult + serialisation
    examples.py              ExampleManager
    visualization.py         Matplotlib figures (PDF)
    report.py                PDF report
    utils.py                 shared helpers and error types
tests/                       pytest suite
requirements.txt
.env.example
```

## 12. Testing

```bash
.venv/bin/pytest -q
```

The suite covers:
- Δx calculation
- Dirichlet–Dirichlet and Dirichlet–Neumann cases
- the heat-flux sign convention
- Neumann–Neumann compatibility detection, for both incompatible and non-unique cases
- heat generation against the analytic parabola, and the ghost-node scheme's exactness
- transient marching, steady-state convergence and energy conservation
- Gaussian elimination tested on its own: pivoting, singular matrices, NaN input, comparison with NumPy, and factorisation replay
- input validation
- N = 2 … 1000
- both MST reference examples
- API response shape and error handling
- PDF generation

## 13. Deployment

The API is deployed as a Render Web Service with this repository at the root:

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |
| Environment | `PYTHON_VERSION=3.12.3`, `THERMOSOLVE_CORS_ORIGINS=<UI origin>` |

See `DEPLOYMENT.md` in the project workspace for the full walkthrough, including the Vercel side and troubleshooting.
