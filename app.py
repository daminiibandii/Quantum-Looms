from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Classical (CP-SAT) from OR-Tools
from ortools.sat.python import cp_model

# QUBO tools
import dimod
try:
    import neal  # simulated annealing (local, no cloud needed)
    HAVE_NEAL = True
except Exception:
    HAVE_NEAL = False

app = FastAPI(title="Quantum Assignment API", version="1.0")

# Allow local dev origins (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class OptimizeRequest(BaseModel):
    cost_matrix: List[List[float]]  # rows = orders, cols = weavers
    penalty: Optional[float] = 10.0  # constraint penalty for QUBO
    solver: Optional[str] = "quantum_local"  # 'classical' | 'quantum_local'

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Classical solver (CP-SAT) ----------------
def solve_classical(cost_matrix: np.ndarray):
    n_orders, n_weavers = cost_matrix.shape
    model = cp_model.CpModel()
    x = {}
    for i in range(n_orders):
        for j in range(n_weavers):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # Each order assigned exactly once
    for i in range(n_orders):
        model.Add(sum(x[i, j] for j in range(n_weavers)) == 1)

    # Objective: minimize total cost
    model.Minimize(sum(int(cost_matrix[i, j] * 1000) * x[i, j]  # scale to int
                       for i in range(n_orders)
                       for j in range(n_weavers)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    assignment = []
    total_cost = 0.0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n_orders):
            for j in range(n_weavers):
                if solver.Value(x[i, j]) == 1:
                    c = float(cost_matrix[i, j])
                    assignment.append({"order": i, "weaver": j, "cost": c})
                    total_cost += c
    return assignment, total_cost, status

# ---------------- QUBO builder ----------------
def build_bqm(cost_matrix: np.ndarray, A: float = 10.0):
    n_orders, n_weavers = cost_matrix.shape
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    # Objective linear terms (costs)
    for i in range(n_orders):
        for j in range(n_weavers):
            var = (i, j)
            bqm.add_variable(var, float(cost_matrix[i, j]))

    # Constraint: each order assigned exactly once -> A*(sum_j x_ij - 1)^2
    for i in range(n_orders):
        # Linear part: -A * sum_j x_ij
        for j in range(n_weavers):
            var = (i, j)
            bqm.add_linear(var, -A)

        # Quadratic part: 2A * sum_{j<k} x_ij x_ik
        for j in range(n_weavers):
            for k in range(j + 1, n_weavers):
                bqm.add_quadratic((i, j), (i, k), 2.0 * A)

        # Offset A*1 is constant; no need to add

    return bqm

def decode_assignment(sample, cost_matrix: np.ndarray):
    n_orders, n_weavers = cost_matrix.shape
    assignment = []
    total_cost = 0.0

    for i in range(n_orders):
        # find chosen j (prefer j with bit=1; repair if none/multiple)
        chosen = [j for j in range(n_weavers) if sample.get((i, j), 0) == 1]
        if len(chosen) == 1:
            j = chosen[0]
        else:
            # repair: pick cheapest j
            j = int(np.argmin(cost_matrix[i, :]))
        c = float(cost_matrix[i, j])
        assignment.append({"order": i, "weaver": j, "cost": c})
        total_cost += c

    return assignment, total_cost

def solve_qubo_local(cost_matrix: np.ndarray, A: float = 10.0):
    bqm = build_bqm(cost_matrix, A=A)
    if HAVE_NEAL:
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=200)
    else:
        # fallback: exact (only for tiny problems)
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)

    best = sampleset.first.sample
    return decode_assignment(best, cost_matrix)

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    cm = np.array(req.cost_matrix, dtype=float)

    # Classical baseline
    classical_assign, classical_cost, status = solve_classical(cm)

    # Quantum-ish (local simulated annealing) unless user forces classical
    if req.solver == "classical":
        quantum_assign, quantum_cost = classical_assign, classical_cost
    else:
        quantum_assign, quantum_cost = solve_qubo_local(cm, A=float(req.penalty))

    return {
        "classical": {
            "assignment": classical_assign,
            "total_cost": classical_cost,
            "status": int(status),
        },
        "quantum": {
            "assignment": quantum_assign,
            "total_cost": quantum_cost,
            "solver": req.solver,
        },
    }
