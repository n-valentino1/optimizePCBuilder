import pandas as pd
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Integer
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

import matplotlib.pyplot as plt


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

mobos = mobos.dropna(subset=['price', 'socket']).reset_index(drop=True)
ram   = ram.dropna(subset=['price']).reset_index(drop=True)
ssds  = ssds.dropna(subset=['price']).reset_index(drop=True)
gpus  = gpus.dropna(subset=['price', 'core_clock', 'memory']).reset_index(drop=True)
cpus  = cpus.dropna(subset=['price', 'boost_clock', 'core_clock']).reset_index(drop=True)

cpu_cols = ['core_count', 'core_clock', 'boost_clock']
scaler = MinMaxScaler()
cpus[cpu_cols] = scaler.fit_transform(cpus[cpu_cols])

gpu_cols = ['memory', 'core_clock']
scaler = MinMaxScaler()
gpus[gpu_cols] = scaler.fit_transform(gpus[gpu_cols])

arch_to_socket = {
    'Zen 5': 'AM5',
    'Zen 4': 'AM5',
    'Zen 3': 'AM4',
    'Zen 2': 'AM4',
    'Zen': 'AM4',
    'Raptor Lake': 'LGA1700',
    'Alder Lake': 'LGA1700',
    'Rocket Lake': 'LGA1200',
    'Comet Lake': 'LGA1200',
}



cpus['socket']  = cpus['microarchitecture'].map(arch_to_socket)
cpus = cpus.dropna(subset=['socket']).reset_index(drop=True)

class PCBuilder(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(
            n_var = 5,
            xl = np.array([0, 0, 0, 0, 0]),
            xu = np.array([len(mobos) -1, 
                           len(ram) -1, 
                           len(ssds) - 1, 
                           len(gpus) -1, 
                           len(cpus) - 1
                           ]),
            n_obj = 6,
            n_ieq_constr = 2
        )

        self.max_budget = 4000
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
        cpu_perf = cpus.loc[c_idx, 'core_count'] + cpus.loc[c_idx, 'core_clock'] + cpus.loc[c_idx, 'boost_clock']
        gpu_perf = gpus.loc[g_idx, 'memory'] + gpus.loc[g_idx, 'core_clock']

        cpu_clock = cpus.loc[c_idx, 'boost_clock']          
        cpu_cores = cpus.loc[c_idx, 'core_count']          
        gpu_clock = gpus.loc[g_idx, 'core_clock']           
        gpu_memory = gpus.loc[g_idx, 'memory']                        
        ssd_capacity = ssds.loc[s_idx, 'capacity']   
        total_performance = (cpu_perf * self.cpu_weight) + (gpu_perf * self.gpu_weight)

        socket_constraint = (
            0 if mobos.loc[m_idx, "socket"] == cpus.loc[c_idx, 'socket'] else 1
        )


        out["F"] = [
            total_price,           
            -cpu_clock,            
            -cpu_cores,           
            -gpu_memory,           
            -gpu_clock,
            -ssd_capacity            
        ]

        budget_constraint = total_price - self.max_budget
        out["G"] = [budget_constraint, socket_constraint]

problem = PCBuilder()

algorithm = NSGA2(pop_size = 1500)

res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 100),
        seed=1,
        verbose=False
        )

labels = ["Price", "-CPU Clock", "-CPU Cores", "-GPU VRAM", "-GPU Clock", "-SSD Size"]
plot = PCP(title="PC Builds - Pareto Front", labels=labels)
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(res.F, color="grey", alpha=0.15)
plot.show()

#Validate your solutions by making your own choice with common sense and see if it's better or worse
#https://pymoo.org/visualization/pcp.html