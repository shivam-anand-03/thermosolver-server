"""HTTP routes. All numerical work is delegated to :mod:`app.solver`."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..models import ReportRequest, SolveEnvelope, SteadyRequest, TransientRequest
from ..solver.examples import ExampleManager
from ..solver.report import build_pdf_report
from ..solver.steady_state import SteadyStateSolver
from ..solver.transient import TransientSolver
from ..solver.validation import InputValidator

router = APIRouter(prefix="/api")
examples = ExampleManager()


@router.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "thermosolve"}


@router.get("/examples")
def list_examples() -> list[dict]:
    """Pre-loaded example problems, including the MST-1 reference problems."""
    return examples.list()


@router.post("/solve/steady")
def solve_steady(req: SteadyRequest) -> dict:
    """Solve d²T/dx² + q'''/k = 0 with finite differences + Gaussian elimination."""
    return SteadyStateSolver(req).solve().to_dict()


@router.post("/solve/transient")
def solve_transient(req: TransientRequest) -> dict:
    """Solve ∂T/∂t = α ∂²T/∂x² with the fully implicit scheme."""
    return TransientSolver(req).solve().to_dict()


@router.post("/validate")
def validate(env: SolveEnvelope) -> dict:
    """Check inputs without solving; returns errors and warnings."""
    v = InputValidator()
    if env.problem_type == "steady":
        if env.steady is None:
            raise HTTPException(422, "Missing 'steady' problem definition.")
        return v.validate_steady(env.steady).to_dict()
    if env.transient is None:
        raise HTTPException(422, "Missing 'transient' problem definition.")
    return v.validate_transient(env.transient).to_dict()


@router.post("/report/pdf")
def report_pdf(req: ReportRequest) -> Response:
    """Re-solve the problem and return a PDF solution report."""
    if req.problem_type == "steady" and req.steady is not None:
        result = SteadyStateSolver(req.steady).solve().to_dict()
    elif req.problem_type == "transient" and req.transient is not None:
        result = TransientSolver(req.transient).solve().to_dict()
    else:
        raise HTTPException(422, "Missing problem definition for the report.")
    pdf = build_pdf_report(result)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="thermosolve-report.pdf"'})
