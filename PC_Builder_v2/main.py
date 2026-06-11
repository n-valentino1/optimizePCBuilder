import pandas as pd
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Integer
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

from sklearn.preprocessing import MinMaxScaler

import os

# Get the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    mobos = pd.read_csv("motherboard.csv")
    ram = pd.read_csv( "memory.csv")
    ssds = pd.read_csv( "internal-hard-drive.csv")
    gpus = pd.read_csv("video-card.csv")
    cpus = pd.read_csv( "cpu.csv")

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

cpu_cols = ['core_count', 'core_clock', 'price']
scaler = MinMaxScaler()
cpus[cpu_cols] = scaler.fit_transform(cpus[cpu_cols])


print(cpus[cpu_cols])
class PCBuilder(ElementwiseProblem):
    
    def __init__(self):
        super.__init__(
            n_var = 5,
            xl = np.array([0, 0, 0, 0, 0]),
            xu = np.array([len(mobos) -1, 
                           len(ram) -1, 
                           len(ssds) - 1, 
                           len(gpus) -1, 
                           len(cpus) - 1
                           ]),
            n_obj = 2
        )

        self.max_budget = 1200
        self.gpu_weight = 0.7
        self.cpu_weight = 0.3

    def _evaluate(self, x, out, *args, **kwargs):
        m_idx = int(x[0])
        r_idx = int(x[1])
        s_idx = int(x[2])
        g_idx = int(x[3])
        c_idx = int(x[4])

        #price
        total_price = (
            mobos.loc[m_idx, "price"]
            + ram.loc[r_idx, "price"]
            + ssds.loc[s_idx, "price"]
            + gpus.loc[g_idx, "price"]
            + cpus.loc[c_idx, "price"]
        )

        #performance
        cpu_perf = cpus.loc[c_idx, 'core_count'] + cpus.loc['core_clock']

        total_performance = cpu_perf * self.cpu_weight


        out["F"] = [total_price, -total_performance]

problem = PCBuilder()

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
