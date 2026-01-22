# 🚁 Drone Swarm: Time Deviation Optimization

This project implements a simulation for optimizing the flight paths of a drone swarm. Unlike standard **Multiple Traveling Salesman Problems (mTSP)** which minimize total distance, this algorithm focuses on **minimizing time deviation** (variance) between drones to ensure simultaneous mission completion.

## 🎯 key Features
*   **Time Deviation Optimization**: specific logic to balance the load between "overworked" and "idle" drones.
*   **Interactive Simulation**: 
    *   **K-Means Clustering** for initial sector assignment.
    *   **Greedy TSP** for route planning within sectors.
    *   **Iterative Load Balancing** to transfer targets and harmonize flight times.
*   **Desktop GUI**: Built with **Tkinter** and **Matplotlib** for a native Windows experience.
*   **Visualization**:
    *   **Mission Map**: Real-time flight paths and target clusters.
    *   **Gantt Chart**: Compare flight durations of all drones.
    *   **Convergence Plot**: Watch the algorithm minimize the maximum mission time.

## 🚀 How to Run

### Prerequisites
*   Python 3.8+
*   Git (optional, for cloning)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/sohamsans/Drone-Swarm-Time-Deviation-Optimization-.git
    cd Drone-Swarm-Time-Deviation-Optimization-
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    *   **Windows**: Double-click `run_app.bat`.
    *   **Terminal**: 
        ```bash
        python desktop_app.py
        ```

## 🛠️ Usage
1.  **Configure Parameters** in the sidebar:
    *   **Number of Drones**: Swarm size (2-20).
    *   **Number of Targets**: Mission complexity.
    *   **Map Size & Speed**: Environment settings.
    *   **Max Iterations & Threshold**: Optimization aggressiveness.
2.  Click **Run Simulation**.
3.  View the **Mission Map** tab to see routes.
4.  View the **Analytics** tab to see the Gantt chart equalization.

## 📂 Project Structure
*   `desktop_app.py`: Main application code (Tkinter GUI + Logic).
*   `requirements.txt`: Python dependencies.
*   `run_app.bat`: Windows launcher script.

## 🧠 The Algorithm
1.  **Initialize**: Random targets + K-Means clustering.
2.  **Route**: Solve TSP for each cluster.
3.  **Optimize Loop**:
    *   Identify the drone with the **Max Flight Time** and **Min Flight Time**.
    *   Identify a "boundary target" in the Max drone's cluster that is closest to the Min drone.
    *   Tentatively move the target.
    *   **Hysteresis Check**: If the move reduces the standard deviation of flight times, commit it. Otherwise, revert.
    *   Repeat until converged or max iterations reached.
