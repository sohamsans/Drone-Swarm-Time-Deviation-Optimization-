import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Simulation Logic
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
        max_idx = np.argmax(self.times)
        min_idx = np.argmin(self.times)
        max_time = self.times[max_idx]
        min_time = self.times[min_idx]
        
        self.history_max_time.append(max_time)
        
        deviation = (max_time - min_time) / max_time if max_time > 0 else 0
        if deviation < threshold:
            return True, deviation
            
        if len(self.clusters[max_idx]) <= 1:
            return False, deviation
            
        max_cluster = np.array(self.clusters[max_idx])
        if len(self.clusters[min_idx]) > 0:
            min_centroid = np.mean(self.clusters[min_idx], axis=0)
        else:
            min_centroid = self.depot
            
        dists = cdist(max_cluster, [min_centroid])
        candidate_local_idx = np.argmin(dists)
        candidate_target = max_cluster[candidate_local_idx]
        
        self.clusters[max_idx].pop(candidate_local_idx)
        self.clusters[min_idx].append(candidate_target)
        
        path_max, dist_max = solve_tsp_greedy(self.clusters[max_idx], self.depot)
        path_min, dist_min = solve_tsp_greedy(self.clusters[min_idx], self.depot)
        
        new_time_max = dist_max / self.speed
        new_time_min = dist_min / self.speed
        
        if np.std([new_time_max, new_time_min]) < np.std([max_time, min_time]):
            self.routes[max_idx] = path_max
            self.times[max_idx] = new_time_max
            self.routes[min_idx] = path_min
            self.times[min_idx] = new_time_min
            return False, deviation
        else:
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
        
        # Current Logic Object
        self.sim = None
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # 1. Sidebar (Left)
        self.sidebar = ttk.Frame(self, width=300, padding="15")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(self.sidebar, text="🎮 Parameters", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 15))

        # --- Precise Inputs ---
        self._add_input("Number of Drones:", "n_drones", 5, widget_type="spinbox", from_=2, to=50)
        self._add_input("Number of Targets:", "n_targets", 50)
        self._add_input("Map Size (m):", "map_size", 1000)
        self._add_input("Drone Speed (m/s):", "speed", 10)
        
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=15)
        
        ttk.Label(self.sidebar, text="⚙️ Logic", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self._add_input("Max Iterations:", "max_iter", 100)
        self._add_input("Threshold (%):", "thresh", 1.0)
        
        self.stop_at_thresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Run Until Threshold Met", variable=self.stop_at_thresh_var).pack(anchor="w", pady=5)
        
        tk.Label(self.sidebar, text="").pack(pady=5)
        
        # Run Button
        self.run_btn = ttk.Button(self.sidebar, text="🚀 Run Simulation", command=self.start_simulation)
        self.run_btn.pack(fill=tk.X, pady=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.sidebar, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)
        self.status_lbl = ttk.Label(self.sidebar, text="Ready", wraplength=280)
        self.status_lbl.pack(pady=5)
        
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # --- Exports ---
        ttk.Label(self.sidebar, text="💾 Export", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.btn_csv = ttk.Button(self.sidebar, text="Save Data (.csv)", command=self.save_csv, state="disabled")
        self.btn_csv.pack(fill=tk.X, pady=2)
        
        self.btn_img = ttk.Button(self.sidebar, text="Save Graphs (.png)", command=self.save_graphs, state="disabled")
        self.btn_img.pack(fill=tk.X, pady=2)

        # 2. Main Content (Right)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="🗺️ Mission Map")
        self.notebook.add(self.tab2, text="📊 Analytics")
        
        self._init_plots()

    def _add_input(self, label_text, var_name, default_val, widget_type="entry", **kwargs):
        container = ttk.Frame(self.sidebar)
        container.pack(fill=tk.X, pady=2)
        
        ttk.Label(container, text=label_text).pack(anchor="w")
        
        if widget_type == "spinbox":
            var = tk.IntVar(value=default_val)
            widget = ttk.Spinbox(container, from_=kwargs.get('from_'), to=kwargs.get('to'), textvariable=var)
        else:
            var = tk.StringVar(value=str(default_val))
            widget = ttk.Entry(container, textvariable=var)
            
        widget.pack(fill=tk.X)
        setattr(self, var_name, var)

    def _init_plots(self):
        # Map
        self.fig_map = Figure(figsize=(8, 8), dpi=100)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("Mission Map")
        self.ax_map.set_xlabel("X (m)") 
        self.ax_map.set_ylabel("Y (m)")
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab1)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Analytics
        self.fig_anal = Figure(figsize=(8, 8), dpi=100)
        self.ax_gantt = self.fig_anal.add_subplot(211)
        self.ax_conv = self.fig_anal.add_subplot(212)
        self.fig_anal.tight_layout(pad=3.0)
        self.canvas_anal = FigureCanvasTkAgg(self.fig_anal, master=self.tab2)
        self.canvas_anal.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        try:
            # Validate Inputs
            n_drones = int(self.n_drones.get())
            n_targets = int(self.n_targets.get())
            map_size = float(self.map_size.get())
            speed = float(self.speed.get())
            max_iter = int(self.max_iter.get())
            thresh = float(self.thresh.get()) / 100.0
            run_until_thresh = self.stop_at_thresh_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all parameters are valid numbers.")
            return

        self.run_btn.state(['disabled'])
        self.btn_csv.state(['disabled'])
        self.btn_img.state(['disabled'])
        
        self.progress_var.set(0)
        self.status_lbl.config(text="Initializing...")
        
        t = threading.Thread(target=self.run_logic, args=(n_drones, n_targets, map_size, speed, max_iter, thresh, run_until_thresh))
        t.daemon = True
        t.start()

    def run_logic(self, n_drones, n_targets, map_size, speed, max_iter, thresh, run_until_thresh):
        try:
            self.sim = DroneSwarmSimulation(n_drones, n_targets, map_size, speed)
            sim = self.sim
            
            # Step 1
            self.update_status("Clustering & Routing...")
            sim.initial_clustering()
            sim.update_routes()
            self.update_plots(sim)
            
            # Step 2
            self.update_status("Optimizing...")
            converged = False
            converged = False
            
            # Determine loop limit
            loop_limit = 1000000 if run_until_thresh else max_iter
            
            for i in range(loop_limit):
                converged, dev = sim.step_optimize(thresh)
                if i % 5 == 0 or converged:
                    progress_val = (i+1) / max_iter * 100 if not run_until_thresh else 0
                    if run_until_thresh: progress_val = min((i % 100), 100) # Indeterminate spinner effect
                    
                    self.update_progress(progress_val, f"Iter {i}: Deviation {dev:.1%}")
                    self.update_plots(sim)
                
                if converged:
                    self.update_status(f"Converged! Deviation {dev:.1%}")
                    break
                time.sleep(0.005)
                
            self.update_plots(sim)
            if not converged:
                 msg = "Finished max iterations." if not run_until_thresh else "Stopped (Hit Safety Limit)."
                 self.update_status(msg)
            
            # Enable Exports
            self.after(0, lambda: [self.btn_csv.state(['!disabled']), self.btn_img.state(['!disabled'])])

        except Exception as e:
            self.update_status(f"Error: {e}")
            print(e)
        finally:
            self.run_btn.state(['!disabled'])

    def update_status(self, text):
        self.after(0, lambda: self.status_lbl.config(text=text))
        
    def update_progress(self, val, text):
        self.after(0, lambda: [self.progress_var.set(val), self.status_lbl.config(text=text)])

    def update_plots(self, sim):
        def _update():
            # Map
            self.ax_map.clear()
            self.ax_map.set_title(f"Mission Map (Bias={np.std(sim.times):.2f})")
            self.ax_map.plot(sim.depot[0], sim.depot[1], 'y*', markersize=15, label='Depot')
            colors = plt.cm.tab20(np.linspace(0, 1, sim.n))
            for i in range(sim.n):
                path = sim.routes[i]
                if path is None: continue
                self.ax_map.plot(path[:,0], path[:,1], '-o', color=colors[i], markersize=4)
            self.canvas_map.draw()
            
            # Gantt
            self.ax_gantt.clear()
            self.ax_gantt.set_title("Flight Durations")
            y_pos = np.arange(sim.n)
            self.ax_gantt.barh(y_pos, sim.times, color=colors)
            self.ax_gantt.set_yticks(y_pos)
            self.ax_gantt.set_yticklabels([f"D{i}" for i in range(sim.n)])
            self.ax_gantt.invert_yaxis()
            self.ax_gantt.set_xlabel("Time (s)")
            
            # Conv
            self.ax_conv.clear()
            self.ax_conv.set_title("Optimization Convergence")
            self.ax_conv.plot(sim.history_max_time, 'r-')
            self.ax_conv.set_ylabel("Max Time (s)")
            self.canvas_anal.draw()
            
        self.after(0, _update)

    def save_csv(self):
        if not self.sim: return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path: return
        
        data = []
        for i in range(self.sim.n):
            data.append({
                "Drone ID": i,
                "Targets Assigned": len(self.sim.clusters[i]),
                "Flight Time (s)": self.sim.times[i],
                "Distance (m)": self.sim.times[i] * self.sim.speed
            })
        
        pd.DataFrame(data).to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Data saved to {file_path}")

    def save_graphs(self):
        if not self.sim: return
        base_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not base_path: return
        
        path_map = base_path.replace(".png", "_map.png")
        path_anal = base_path.replace(".png", "_analytics.png")
        
        self.fig_map.savefig(path_map)
        self.fig_anal.savefig(path_anal)
        
        messagebox.showinfo("Success", f"Graphs saved:\n{path_map}\n{path_anal}")

if __name__ == "__main__":
    app = DroneApp()
    app.mainloop()
