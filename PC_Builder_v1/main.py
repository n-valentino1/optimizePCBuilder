import pandas as pd
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Integer
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

import os

# Get the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    mobos = pd.read_csv(os.path.join(script_dir, "motherboards.csv"))
    ram = pd.read_csv(os.path.join(script_dir, "ram.csv"))
    ssds = pd.read_csv(os.path.join(script_dir, "ssds.csv"))
    gpus = pd.read_csv(os.path.join(script_dir, "gpus.csv"))
    cpus = pd.read_csv(os.path.join(script_dir, "cpus.csv"))

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()


class PCBuilderProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=5,                       # 5 design variables
            xl=np.array([0, 0, 0, 0, 0]),  # Lower bounds for all variables
            xu=np.array([5, 5, 5, 5, 5]),  # Upper bounds for all variables
            n_obj=2,
            n_ieq_constr=1,
            n_eq_constr=2
        )

        self.max_budget = 5000
        self.gpu_weight = 0.7 #Gaming focused 0.7GPU, 0.3 CPU
        self.cpu_weight = 0.3
    
    def _evaluate(self, x, out, *args, **kwargs):
        m_idx = int(x[0])
        r_idx = int(x[1])
        s_idx = int(x[2])
        g_idx = int(x[3])
        c_idx = int(x[4])

        #performance
        gpu_perf = gpus.loc[g_idx, "gaming"]
        cpu_perf = cpus.loc[c_idx, "gaming"]

        total_performance = (gpu_perf * self.gpu_weight) + ( cpu_perf * self.cpu_weight)

        #price
        total_price = (
            mobos.loc[m_idx, "price"]
            + ram.loc[r_idx, "price"]
            + ssds.loc[s_idx, "price"]
            + gpus.loc[g_idx, "price"]
            + cpus.loc[c_idx, "price"]
        )
        out["F"] = [total_price, -total_performance]

        budget_constraint = total_price - self.max_budget

        socket_constraint = (
            0 if mobos.loc[m_idx, "socket"] == cpus.loc[c_idx, "socket"] else 1
        )

        ram_constraint = (
            0 if mobos.loc[m_idx, "ram_type"] == ram.loc[r_idx, "ram_type"] else 1
        )

        out["G"] = [budget_constraint]
        out["H"]  = [ram_constraint, socket_constraint]

problem = PCBuilderProblem()

algorithm = NSGA2(pop_size = 100)

res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 50), # Run for 50 generations
        seed=1,
        verbose=False
        )

plot = Scatter(title="Approximated Pareto Front", xlabel="Price", ylabel="Performance")
plot.add(res.F, color="red", marker="o")
plot.show()
