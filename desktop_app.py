import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Matplotlib integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Simulation Logic (Ported)
# -----------------------------------------------------------------------------
def generate_targets(n, size):
    return np.random.rand(n, 2) * size

def solve_tsp_greedy(points, start_pos):
    if len(points) == 0:
        return np.array([start_pos]), 0.0

    curr = start_pos
    path = [curr]
    unvisited = list(points)
    total_dist = 0.0

    while unvisited:
        dist_matrix = cdist([curr], unvisited)
        nearest_idx = np.argmin(dist_matrix)
        nearest_node = unvisited[nearest_idx]
        
        path.append(nearest_node)
        total_dist += dist_matrix[0, nearest_idx]
        
        curr = nearest_node
        unvisited.pop(nearest_idx)

    dist_to_home = np.linalg.norm(curr - start_pos)
    path.append(start_pos)
    total_dist += dist_to_home
    
    return np.array(path), total_dist

class DroneSwarmSimulation:
    def __init__(self, n_drones, n_targets, map_size, speed):
        self.n = n_drones
        self.m = n_targets
        self.size = map_size
        self.speed = speed
        self.depot = np.array([map_size/2, map_size/2])
        self.targets = generate_targets(n_targets, map_size)
        
        self.clusters = [[] for _ in range(n_drones)]
        self.routes = [None] * n_drones
        self.times = np.zeros(n_drones)
        self.history_max_time = []
    
    def initial_clustering(self):
        kmeans = KMeans(n_clusters=self.n, n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.targets)
        for i in range(self.n):
            self.clusters[i] = [self.targets[j] for j, label in enumerate(labels) if label == i]

    def update_routes(self):
        for i in range(self.n):
            cluster_points = np.array(self.clusters[i])
            path, dist = solve_tsp_greedy(cluster_points, self.depot)
            self.routes[i] = path
            self.times[i] = dist / self.speed

    def step_optimize(self, threshold):
        """Performs one step of optimization. Returns True if converged."""
        max_idx = np.argmax(self.times)
        min_idx = np.argmin(self.times)
        max_time = self.times[max_idx]
        min_time = self.times[min_idx]
        
        self.history_max_time.append(max_time)
        
        deviation = (max_time - min_time) / max_time if max_time > 0 else 0
        if deviation < threshold:
            return True, deviation # Converged
            
        if len(self.clusters[max_idx]) <= 1:
            return False, deviation
            
        max_cluster = np.array(self.clusters[max_idx])
        if len(self.clusters[min_idx]) > 0:
            min_centroid = np.mean(self.clusters[min_idx], axis=0)
        else:
            min_centroid = self.depot
            
        # Find move candidate
        dists = cdist(max_cluster, [min_centroid])
        candidate_local_idx = np.argmin(dists)
        candidate_target = max_cluster[candidate_local_idx]
        
        # Tentative Move
        self.clusters[max_idx].pop(candidate_local_idx)
        self.clusters[min_idx].append(candidate_target)
        
        # Check
        path_max, dist_max = solve_tsp_greedy(self.clusters[max_idx], self.depot)
        path_min, dist_min = solve_tsp_greedy(self.clusters[min_idx], self.depot)
        
        new_time_max = dist_max / self.speed
        new_time_min = dist_min / self.speed
        
        # Stability Check
        if np.std([new_time_max, new_time_min]) < np.std([max_time, min_time]):
            self.routes[max_idx] = path_max
            self.times[max_idx] = new_time_max
            self.routes[min_idx] = path_min
            self.times[min_idx] = new_time_min
        else:
            # Revert
            self.clusters[min_idx].pop()
            self.clusters[max_idx].insert(candidate_local_idx, candidate_target)
            
        return False, deviation

# -----------------------------------------------------------------------------
# GUI Application
# -----------------------------------------------------------------------------
class DroneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drone Swarm: Time Deviation Optimization")
        self.geometry("1400x900")
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # 1. Sidebar (Left)
        self.sidebar = ttk.Frame(self, width=300, padding="10")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Parameters
        self._add_label("Number of Drones:")
        self.n_drones_var = tk.IntVar(value=5)
        ttk.Scale(self.sidebar, from_=2, to=20, variable=self.n_drones_var, command=lambda v: self._update_label("drones_lbl", int(float(v)))).pack(fill=tk.X)
        self.drones_lbl = ttk.Label(self.sidebar, text="5")
        self.drones_lbl.pack()

        self._add_label("Number of Targets:")
        self.n_targets_var = tk.IntVar(value=50)
        ttk.Scale(self.sidebar, from_=10, to=200, variable=self.n_targets_var, command=lambda v: self._update_label("targets_lbl", int(float(v)))).pack(fill=tk.X)
        self.targets_lbl = ttk.Label(self.sidebar, text="50")
        self.targets_lbl.pack()
        
        self._add_label("Map Size (m):")
        self.map_size_var = tk.IntVar(value=1000)
        ttk.Entry(self.sidebar, textvariable=self.map_size_var).pack(fill=tk.X)
        
        self._add_label("Drone Speed (m/s):")
        self.speed_var = tk.IntVar(value=10)
        ttk.Scale(self.sidebar, from_=1, to=50, variable=self.speed_var, command=lambda v: self._update_label("speed_lbl", int(float(v)))).pack(fill=tk.X)
        self.speed_lbl = ttk.Label(self.sidebar, text="10")
        self.speed_lbl.pack()
        
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self._add_label("Max Iterations:")
        self.max_iter_var = tk.IntVar(value=100)
        ttk.Entry(self.sidebar, textvariable=self.max_iter_var).pack(fill=tk.X)
        
        self._add_label("Threshold (%):")
        self.thresh_var = tk.DoubleVar(value=1.0)
        ttk.Scale(self.sidebar, from_=0.1, to=20, variable=self.thresh_var, command=lambda v: self._update_label("thresh_lbl", f"{float(v):.1f}%")).pack(fill=tk.X)
        self.thresh_lbl = ttk.Label(self.sidebar, text="1.0%")
        self.thresh_lbl.pack()
        
        tk.Label(self.sidebar, text="").pack(pady=5)
        
        self.run_btn = ttk.Button(self.sidebar, text="🚀 Run Simulation", command=self.start_simulation)
        self.run_btn.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.sidebar, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X)
        
        self.status_lbl = ttk.Label(self.sidebar, text="Ready", wraplength=280)
        self.status_lbl.pack(pady=5)

        # 2. Main Content (Right)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="🗺️ Mission Map")
        self.notebook.add(self.tab2, text="📊 Analytics")
        
        # Setup Plots
        self._init_plots()

    def _add_label(self, text):
        ttk.Label(self.sidebar, text=text, font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
        
    def _update_label(self, attr_name, val):
        getattr(self, attr_name).config(text=str(val))

    def _init_plots(self):
        # Map Plot
        self.fig_map = Figure(figsize=(8, 8), dpi=100)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("Mission Map")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab1)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Analytics Plots (Grid Layout)
        self.fig_anal = Figure(figsize=(8, 8), dpi=100)
        self.ax_gantt = self.fig_anal.add_subplot(211) # Top: Gantt
        self.ax_conv = self.fig_anal.add_subplot(212) # Bottom: Convergence
        self.fig_anal.tight_layout(pad=3.0)
        
        self.canvas_anal = FigureCanvasTkAgg(self.fig_anal, master=self.tab2)
        self.canvas_anal.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        self.run_btn.state(['disabled'])
        self.progress_var.set(0)
        self.status_lbl.config(text="Initializing...")
        
        # Run in thread
        t = threading.Thread(target=self.run_logic)
        t.daemon = True
        t.start()

    def run_logic(self):
        try:
            # Gather params
            n_drones = self.n_drones_var.get()
            n_targets = self.n_targets_var.get()
            map_size = float(self.map_size_var.get())
            speed = float(self.speed_var.get())
            max_iter = int(self.max_iter_var.get())
            thresh = self.thresh_var.get() / 100.0
            
            sim = DroneSwarmSimulation(n_drones, n_targets, map_size, speed)
            
            # Step 1: Init
            self.update_status("Steps 1-2: Clustering & Initial Routing...")
            sim.initial_clustering()
            sim.update_routes()
            
            # Draw initial state
            self.update_plots(sim)
            
            # Step 2: Optimize
            self.update_status("Step 3: Optimizing Time Deviation...")
            for i in range(max_iter):
                converged, dev = sim.step_optimize(thresh)
                
                # Update UI occasionally
                if i % 5 == 0 or converged:
                    self.update_progress((i+1)/max_iter*100, f"Iter {i}: Deviation {dev:.1%}")
                    self.update_plots(sim) # Update graphs live
                
                if converged:
                    self.update_status(f"Converged! Deviation {dev:.1%}")
                    break
                time.sleep(0.01) # Small sleep to keep UI responsive
                
            self.update_plots(sim)
            if not converged:
                 self.update_status(f"Finished max iterations. Deviation {dev:.1%}")
                 
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(e)
        finally:
            self.run_btn.state(['!disabled'])

    def update_status(self, text):
        self.after(0, lambda: self.status_lbl.config(text=text))
        
    def update_progress(self, val, text):
        self.after(0, lambda: [self.progress_var.set(val), self.status_lbl.config(text=text)])

    def update_plots(self, sim):
        def _update():
            # 1. Map Update
            self.ax_map.clear()
            self.ax_map.set_title("Mission Map")
            self.ax_map.plot(sim.depot[0], sim.depot[1], 'y*', markersize=15, label='Depot')
            
            colors = plt.cm.tab20(np.linspace(0, 1, sim.n))
            for i in range(sim.n):
                if sim.routes[i] is None: continue
                path = sim.routes[i]
                self.ax_map.plot(path[:,0], path[:,1], '-o', color=colors[i], markersize=4, label=f'D{i}')
            
            self.ax_map.set_xlim(0, sim.size)
            self.ax_map.set_ylim(0, sim.size)
            self.canvas_map.draw()
            
            # 2. Gantt Update
            self.ax_gantt.clear()
            self.ax_gantt.set_title("Timeline (Gantt)")
            y_pos = np.arange(sim.n)
            self.ax_gantt.barh(y_pos, sim.times, color=colors)
            self.ax_gantt.set_yticks(y_pos)
            self.ax_gantt.set_yticklabels([f"Drone {i}" for i in range(sim.n)])
            self.ax_gantt.set_xlabel("Time (s)")
            self.ax_gantt.invert_yaxis()
            
            # 3. Convergence Update
            self.ax_conv.clear()
            self.ax_conv.set_title("Convergence (Max Time)")
            if sim.history_max_time:
                self.ax_conv.plot(sim.history_max_time, 'r-')
            self.ax_conv.set_xlabel("Iteration")
            self.ax_conv.set_ylabel("Max Time (s)")
            
            self.canvas_anal.draw()
            
        self.after(0, _update)

if __name__ == "__main__":
    app = DroneApp()
    app.mainloop()
