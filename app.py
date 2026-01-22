
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Drone Swarm Time Deviation",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 Drone Swarm: Time Deviation Optimization")
st.markdown("""
This simulation minimizes the **variance** in flight times between drones in a swarm. 
Unlike standard TSP which minimizes total distance, this method prioritizes **simultaneous completion**.
""")

# -----------------------------------------------------------------------------
# 1. Sidebar & Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎮 Mission Parameters")
    
    # Mission Settings
    n_drones = st.slider("Number of Drones", 2, 20, 5)
    n_targets = st.slider("Number of Targets", 10, 200, 50)
    map_size = st.number_input("Map Size (m)", 100, 10000, 1000)
    drone_speed = st.slider("Drone Speed (m/s)", 1, 50, 10)
    
    st.divider()
    
    # Algorithm Tuners
    st.subheader("⚙️ Algorithm Tuning")
    max_iterations = st.slider("Max Iterations", 10, 500, 100)
    convergence_threshold = st.slider("Convergence Threshold (%)", 0.0, 20.0, 1.0, step=0.5) / 100.0
    
    run_btn = st.button("🚀 Run Simulation", type="primary")

# -----------------------------------------------------------------------------
# 2. Logic: Classes & Functions
# -----------------------------------------------------------------------------

def generate_targets(n, size):
    """Generate random (x, y) coordinates for targets."""
    return np.random.rand(n, 2) * size

def solve_tsp_greedy(points, start_pos):
    """
    Greedy Nearest Neighbor TSP solver.
    Args:
        points: Array of target coordinates [N, 2]
        start_pos: Coordinate of the depot [1, 2]
    Returns:
        ordered_points: Array of coordinates in visitation order (including start)
        total_dist: Total path distance
    """
    if len(points) == 0:
        return np.array([start_pos]), 0.0

    curr = start_pos
    path = [curr]
    unvisited = list(points)
    total_dist = 0.0

    while unvisited:
        # Find closest unvisited point
        dist_matrix = cdist([curr], unvisited)
        nearest_idx = np.argmin(dist_matrix)
        nearest_node = unvisited[nearest_idx]
        
        path.append(nearest_node)
        total_dist += dist_matrix[0, nearest_idx]
        
        curr = nearest_node
        unvisited.pop(nearest_idx)

    # Return to depot (optional for mTSP generally, but let's assume one-way or 
    # if it's a loop. Requirement says 'Simultaneous completion', usually implies 
    # finishing the last task. Let's assume they DO NOT return to depot to keep it simple,
    # or if they do, we add the distance back.
    # User Context: "Travel Salesman Drone Swarm" usually implies loops.
    # Let's add return to depot for standard TSP behavior.
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
        
        # Assignments: list of arrays, where each array is the targets assigned to drone i
        self.clusters = [[] for _ in range(n_drones)]
        self.routes = [None] * n_drones
        self.times = np.zeros(n_drones)
        self.history_max_time = []
    
    def initial_clustering(self):
        """Step 1: K-Means Clustering"""
        kmeans = KMeans(n_clusters=self.n, n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.targets)
        
        for i in range(self.n):
            self.clusters[i] = [self.targets[j] for j, label in enumerate(labels) if label == i]

    def update_routes(self):
        """Step 2: Solve TSP for each cluster"""
        for i in range(self.n):
            cluster_points = np.array(self.clusters[i])
            path, dist = solve_tsp_greedy(cluster_points, self.depot)
            self.routes[i] = path
            self.times[i] = dist / self.speed
            
    def optimize_balance(self, max_iter, threshold, progress_bar):
        """Step 3: Time Deviation Loop"""
        
        for iteration in range(max_iter):
            # 1. Identify Overworked and Idle drones
            max_idx = np.argmax(self.times)
            min_idx = np.argmin(self.times)
            
            max_time = self.times[max_idx]
            min_time = self.times[min_idx]
            
            # Record metric
            self.history_max_time.append(max_time)
            
            # Convergence Check
            deviation = (max_time - min_time) / max_time if max_time > 0 else 0
            
            if deviation < threshold:
                break
                
            # Update Progress
            progress_bar.progress(min((iteration + 1) / max_iter, 1.0), text=f"Iteration {iteration}: Deviation {deviation:.1%}")
            
            # 2. Optimization: Try to move a target from Max -> Min
            if len(self.clusters[max_idx]) <= 1:
                # Can't take from a drone with 0 or 1 target
                continue
                
            # Find the target in Max's cluster that is closest to Min's Centroid (or depot if empty)
            max_cluster = np.array(self.clusters[max_idx])
            
            if len(self.clusters[min_idx]) > 0:
                min_centroid = np.mean(self.clusters[min_idx], axis=0)
            else:
                min_centroid = self.depot
                
            dists = cdist(max_cluster, [min_centroid])
            candidate_local_idx = np.argmin(dists)
            candidate_target = max_cluster[candidate_local_idx]
            
            # 3. Stability Check (Hysteresis-lite)
            # Temporarily move
            self.clusters[max_idx].pop(candidate_local_idx)
            self.clusters[min_idx].append(candidate_target)
            
            # Recalculate routes FOR ONLY THESE TWO
            path_max, dist_max = solve_tsp_greedy(self.clusters[max_idx], self.depot)
            path_min, dist_min = solve_tsp_greedy(self.clusters[min_idx], self.depot)
            
            new_time_max = dist_max / self.speed
            new_time_min = dist_min / self.speed
            
            # Check if global std dev would likely decrease (simplified: did the gap close?)
            # Actually, standard deviation is the user req, but gap (max-min) is a good proxy.
            # If the new setup makes the previous 'min' drone worse than the previous 'max' drone, revert.
            
            if np.std([new_time_max, new_time_min]) < np.std([max_time, min_time]):
                # Commit: update specific routes/times
                self.routes[max_idx] = path_max
                self.times[max_idx] = new_time_max
                self.routes[min_idx] = path_min
                self.times[min_idx] = new_time_min
            else:
                # Revert
                self.clusters[min_idx].pop()
                self.clusters[max_idx].insert(candidate_local_idx, candidate_target)

# -----------------------------------------------------------------------------
# 3. Main Execution & Visualization
# -----------------------------------------------------------------------------

if run_btn:
    st.success("Analysis Started...")
    
    sim = DroneSwarmSimulation(n_drones, n_targets, map_size, drone_speed)
    
    # 1. Initialize
    with st.spinner("Clustering..."):
        sim.initial_clustering()
        sim.update_routes()
    
    # 2. Optimize
    bar = st.progress(0, text="Optimizing Route Balance...")
    sim.optimize_balance(max_iterations, convergence_threshold, bar)
    bar.empty()
    
    # 3. Results Layout
    tab1, tab2, tab3 = st.tabs(["🗺️ Mission Map", "📊 Analytics & Gantt", "💾 Raw Data"])
    
    with tab1:
        st.subheader("Mission Visualization")
        
        # Prepare Data for Plotly
        fig_map = go.Figure()
        
        # Plot Depot
        fig_map.add_trace(go.Scatter(
            x=[sim.depot[0]], y=[sim.depot[1]],
            mode='markers',
            marker=dict(size=20, symbol='star', color='gold'),
            name='Depot'
        ))
        
        colors = px.colors.qualitative.Plotly
        
        for i in range(sim.n):
            path = sim.routes[i]
            if path is None: continue
            
            # Line
            fig_map.add_trace(go.Scatter(
                x=path[:, 0], y=path[:, 1],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=6),
                name=f'Drone {i} (T={sim.times[i]:.1f}s)'
            ))
            
            # Annotate end time
            # fig_map.add_annotation(
            #     x=path[-2, 0], y=path[-2, 1], # Last target before return
            #     text=f"{sim.times[i]:.1f}s",
            #     showarrow=False,
            #     yshift=10
            # )
            
        fig_map.update_layout(
            height=600,
            xaxis=dict(range=[0, map_size], title="X (m)"),
            yaxis=dict(range=[0, map_size], title="Y (m)"),
            title="Drone Flight Paths & Target Clusters"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimization Convergence")
            fig_conv = px.line(
                x=range(len(sim.history_max_time)), 
                y=sim.history_max_time,
                labels={'x': "Iteration", 'y': "Max Mission Time (s)"},
                title="Max Mission Time over Iterations"
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            
        with col2:
            st.subheader("Final load distribution")
            fig_bar = px.bar(
                x=[f"Drone {i}" for i in range(sim.n)],
                y=sim.times,
                color=[f"Drone {i}" for i in range(sim.n)],
                labels={'y': "Total Time (s)", 'x': "Drone ID"},
                title="Flight Time per Drone"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.subheader("Mission Gantt Chart")
        
        # Gantt Data Construction
        gantt_data = []
        for i in range(sim.n):
            gantt_data.append(dict(
                Task=f"Drone {i}", 
                Start=0, 
                Finish=sim.times[i], 
                Resource=f"Drone {i}"
            ))
            
        df_gantt = pd.DataFrame(gantt_data)
        fig_gantt = px.timeline(
            df_gantt, x_start="Start", x_end="Finish", y="Task", color="Resource",
            title="Drone Mission Timeline"
        )
        fig_gantt.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_gantt, use_container_width=True)

    with tab3:
        st.subheader("Simulation Statistics")
        st.metric("Total Mission Time (Slowest Drone)", f"{max(sim.times):.2f} s")
        st.metric("Fastest Drone", f"{min(sim.times):.2f} s")
        st.metric("Time Deviation (Max - Min)", f"{max(sim.times) - min(sim.times):.2f} s")
        
        st.dataframe(pd.DataFrame({
            "Drone ID": range(sim.n),
            "Targets Assigned": [len(c) for c in sim.clusters],
            "Flight Time (s)": sim.times,
            "Distance (m)": sim.times * sim.speed
        }))
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")

