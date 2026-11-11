"""Legacy wrappers for SimplexSolver and GraphicalORSolver.

Maintains backward compatibility while providing auto-scaling,
exact corner-point enumeration and standard interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from polytech.optimisation.orsolver.core.model import Problem, Sense
from polytech.optimisation.orsolver.lp.simplex import TwoPhaseSimplexSolver
from polytech.optimisation.orsolver.visualizer.graphical_2d import DynamicGraphicalSolver


class SimplexSolver:
    """Class to solve linear programming problems using the Simplex method."""

    def __init__(
        self,
        objective: List[float],
        constraints: List[Tuple[List[float], str, float]],
        bounds: Optional[Any] = None,
    ):
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
        self.solver = TwoPhaseSimplexSolver()

    def solve(self) -> Dict[str, Any]:
        p = Problem(name="simplex_problem", sense=Sense.MAXIMIZE)
        num_vars = len(self.objective)
        var_names = [f"x{i+1}" for i in range(num_vars)]

        for v in var_names:
            p.add_variable(v, lower_bound=0.0)

        obj_dict = {f"x{i+1}": float(c) for i, c in enumerate(self.objective)}
        p.set_objective(obj_dict, sense=Sense.MAXIMIZE)

        for i, (coeffs, op, val) in enumerate(self.constraints):
            c_dict = {f"x{j+1}": float(coeffs[j]) for j in range(len(coeffs))}
            p.add_constraint(c_dict, op, float(val), name=f"c{i+1}")

        sol = self.solver.solve(p)

        solution_list = [sol.variable_values.get(v, 0.0) for v in var_names]
        obj_val = sol.objective_value if sol.objective_value is not None else float("nan")

        return {"solution": solution_list, "objective_value": obj_val}


class GraphicalORSolver:
    """Class to visualize linear programming problems and calculate solutions graphically."""

    def __init__(
        self,
        constraints: List[Tuple[List[float], str, float]],
        bounds: Optional[Any] = None,
        objective: Optional[List[float]] = None,
    ):
        self.constraints = constraints
        self.bounds = bounds
        self.objective = objective or [1.0, 1.0]
        self.solver = DynamicGraphicalSolver()
        self._problem = self._build_problem()

    def _build_problem(self) -> Problem:
        p = Problem(name="graphical_problem", sense=Sense.MAXIMIZE)
        p.add_variable("x1", lower_bound=0.0)
        p.add_variable("x2", lower_bound=0.0)

        p.set_objective({"x1": float(self.objective[0]), "x2": float(self.objective[1])})

        for i, (coeffs, op, val) in enumerate(self.constraints):
            p.add_constraint(
                {"x1": float(coeffs[0]), "x2": float(coeffs[1])},
                op,
                float(val),
                name=f"c{i+1}",
            )
        return p

    def calculate_graphical_solution(self) -> Dict[str, Any]:
        res = self.solver.solve(self._problem)
        opt_pt = res["optimal_point"]
        sol_arr = np.array(opt_pt) if opt_pt is not None else None
        return {
            "solution": sol_arr,
            "objective_value": res["optimal_value"],
        }

    def plot(self, solution=None):
        """Plots the solution."""
        self.solver.plot_and_save(self._problem, filepath="solution_graphical.png")

    def save_plot(self, filename: str = "solution.png", format: str = "png"):
        self.solver.plot_and_save(self._problem, filepath=filename)
