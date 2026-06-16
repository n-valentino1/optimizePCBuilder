import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

from sklearn.preprocessing import MinMaxScaler

# Data Loading and Cleaning
try:
    mobos = pd.read_csv("motherboard.csv")
    ram   = pd.read_csv("memory.csv")
    ssds  = pd.read_csv("internal-hard-drive.csv")
    gpus  = pd.read_csv("video-card.csv")
    cpus  = pd.read_csv("cpu.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

mobos = mobos.dropna(subset=['price', 'socket', 'max_memory']).reset_index(drop=True)
ram   = ram.dropna(subset=['price', 'speed', 'modules']).reset_index(drop=True)
ssds  = ssds.dropna(subset=['price', 'capacity']).reset_index(drop=True)
gpus  = gpus.dropna(subset=['price', 'core_clock', 'memory']).reset_index(drop=True)
cpus  = cpus.dropna(subset=['price', 'boost_clock', 'core_clock', 'core_count', 'microarchitecture']).reset_index(drop=True)

def parse_ram_gen(speed_str):
    speed_str = str(speed_str).upper()
    if 'DDR5' in speed_str: return 5
    if 'DDR4' in speed_str: return 4
    if 'DDR3' in speed_str: return 3
    
    # Fallback: If your CSV just lists speeds like "3200" or "5600 MT/s"
    try:
        speed_num = int(''.join(filter(str.isdigit, speed_str)))
        if speed_num >= 4800: return 5
        if speed_num >= 2133: return 4
        return 3
    except ValueError:
        return 4 # Default to DDR4 as a safe middle ground if text is unreadable

def parse_ram_cap(modules_str):
    try:
        modules_str = str(modules_str).upper()
        # Handles standard PCPartPicker format: "2 x 8 GB" or "2x8GB"
        if 'X' in modules_str:
            parts = modules_str.split('X')
            count = int(''.join(filter(str.isdigit, parts[0])))
            size = int(''.join(filter(str.isdigit, parts[1])))
            return count * size
        # Handles your original comma split format: "2,8"
        elif ',' in modules_str:
            count, size = modules_str.split(',')
            return int(count) * int(size)
        # Fallback if it's just a raw number like "16" or "16GB"
        else:
            return int(''.join(filter(str.isdigit, modules_str)))
    except Exception:
        return 16 # Default to a standard 16GB kit if parsing fails
    
def parse_ssd_cap(cap_str):
    try:
        cap_str = str(cap_str).upper()
        if 'TB' in cap_str:
            return float(cap_str.replace('TB', '').strip()) * 1000
        return float(cap_str.replace('GB', '').strip())
    except Exception:
        return np.nan

ram['ram_gen'] = ram['speed'].apply(parse_ram_gen)
ram['capacity_gb'] = ram['modules'].apply(parse_ram_cap)
ram = ram.dropna(subset=['ram_gen', 'capacity_gb']).reset_index(drop=True)
ram['ram_gen'] = ram['ram_gen'].astype(int)
ram['capacity_gb'] = ram['capacity_gb'].astype(int)

ssds['capacity_gb'] = ssds['capacity'].apply(parse_ssd_cap)
ssds = ssds.dropna(subset=['capacity_gb']).reset_index(drop=True)

socket_to_ram_gen = {
    'AM5': 5,
    'AM4': 4,
    'LGA1700': 5,
    'LGA1200': 4,
}

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

mobos['ram_gen'] = mobos['socket'].map(socket_to_ram_gen)
mobos = mobos.dropna(subset=['ram_gen']).reset_index(drop=True)
mobos['ram_gen'] = mobos['ram_gen'].astype(int)

cpus['socket']  = cpus['microarchitecture'].map(arch_to_socket)
cpus = cpus.dropna(subset=['socket']).reset_index(drop=True)

# Min-Max Scaling Configuration
scaler = MinMaxScaler()
cpus['scaled_clk']   = scaler.fit_transform(cpus[['boost_clock']])
cpus['scaled_cores'] = scaler.fit_transform(cpus[['core_count']])
gpus['scaled_mem']   = scaler.fit_transform(gpus[['memory']])
gpus['scaled_clk']   = scaler.fit_transform(gpus[['core_clock']])
ssds['scaled_cap']   = scaler.fit_transform(ssds[['capacity_gb']])


class PCBuilder(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(
            n_var=5,
            xl = np.zeros(5),
            xu = np.array([
                len(mobos) - 1,
                len(ram) - 1,
                len(ssds) - 1,
                len(gpus) - 1,
                len(cpus) - 1
        ]),
            n_obj=6,         
            n_ieq_constr=8, 
        )

        self.max_budget = 4000
        self.min_ram_gb = 16
        self.gpu_weight = 0.7
        self.cpu_weight = 0.3

        self.min_cores = 5
        self.min_vram = 11
        self.min_ssd_gb = 900

    def _evaluate(self, x, out, *args, **kwargs):
        m_idx = int(np.round(x[0]))
        r_idx = int(np.round(x[1]))
        s_idx = int(np.round(x[2]))
        g_idx = int(np.round(x[3]))
        c_idx = int(np.round(x[4]))

        #price
        total_price = (
            mobos.loc[m_idx, "price"]
            + ram.loc[r_idx, "price"]
            + ssds.loc[s_idx, "price"]
            + gpus.loc[g_idx, "price"]
            + cpus.loc[c_idx, "price"]
        )

        # Retrieve pre-scaled MinMax attributes for the optimization matrix
        cpu_clock_scaled = cpus.loc[c_idx, 'scaled_clk']          
        cpu_cores_scaled = cpus.loc[c_idx, 'scaled_cores']          
        gpu_clock_scaled = gpus.loc[g_idx, 'scaled_clk']           
        gpu_memory_scaled = gpus.loc[g_idx, 'scaled_mem']                        
        ssd_capacity_scaled = ssds.loc[s_idx, 'scaled_cap']   

        socket_constraint = (
            0 if mobos.loc[m_idx, "socket"] == cpus.loc[c_idx, 'socket'] else 1
        )

        memory_type_constraint = (
            0 if ram.loc[r_idx, "ram_gen"] == mobos.loc[m_idx, "ram_gen"] else 1
        )

        memory_capacity_constraint = (
            ram.loc[r_idx, "capacity_gb"] - mobos.loc[m_idx, "max_memory"]
        )

        min_ram_constraint = self.min_ram_gb - ram.loc[r_idx, "capacity_gb"]
        min_cores_constraint = self.min_cores - cpus.loc[c_idx, "core_count"]
        min_vram_constraint = self.min_vram - gpus.loc[g_idx, "memory"]
        min_ssd_constraint = self.min_ssd_gb - ssds.loc[s_idx, "capacity_gb"]

        out["F"] = [
            total_price,          
            -cpu_clock_scaled,            
            -cpu_cores_scaled,           
            -gpu_memory_scaled,           
            -gpu_clock_scaled,
            -ssd_capacity_scaled            
        ]

        budget_constraint = total_price - self.max_budget
        out["G"] = [
            budget_constraint,
            socket_constraint,
            memory_type_constraint,
            memory_capacity_constraint,
            min_ram_constraint,
            min_cores_constraint,
            min_vram_constraint,
            min_ssd_constraint
        ]

problem = PCBuilder()
algorithm = NSGA2(pop_size = 250)

res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 250),
        seed=1,
        verbose=False
        )

labels = ["Price", "-CPU Clk", "-Cores", "-VRAM", "-GPU Clk", "-SSD Size"]
plot = PCP(title="PC Builds - Pareto Front", labels=labels)
plot.set_axis_style(color="grey", alpha=0.5)
plot.add(res.F, color="blue", alpha=0.1) # Changed color to blue for better line tracking

# Clean up the matplotlib layout before showing it
plt.tight_layout()
fig = plt.gcf()
fig.subplots_adjust(bottom=0.2, top=0.85) # Adds breathing room for labels and titles
plt.xticks(rotation=15) # Rotates labels slightly so they don't crash into each other
plot.show()

for i, x in enumerate(res.X):
    m_idx = int(np.clip(x[0], 0, len(mobos) - 1))
    r_idx = int(np.clip(x[1], 0, len(ram) - 1))
    s_idx = int(np.clip(x[2], 0, len(ssds) - 1))
    g_idx = int(np.clip(x[3], 0, len(gpus) - 1))
    c_idx = int(np.clip(x[4], 0, len(cpus) - 1))

    print(f"--- Build {i+1} ---")
    print(f"Motherboard: {mobos.loc[m_idx, 'name']}")
    print(f"RAM:         {ram.loc[r_idx, 'name']}")
    print(f"SSD:         {ssds.loc[s_idx, 'name']}")
    print(f"GPU:         {gpus.loc[g_idx, 'name']}")
    print(f"CPU:         {cpus.loc[c_idx, 'name']}")
    print(f"Total Price: ${res.F[i][0]:.2f}")
    print()

#Validate your solutions by making your own choice with common sense and see if it's better or worse
#https://pymoo.org/visualization/pcp.html