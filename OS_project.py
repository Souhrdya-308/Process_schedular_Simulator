import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Scheduling Algorithms
def calculate_throughput(n, completion_array):
    return n / max(completion_array) if max(completion_array) > 0 else 0

def calculate_avg_waiting_time(waiting_array):
    return np.mean(waiting_array) if len(waiting_array) > 0 else 0

def first_come_first_serve(n, arrival_array, burst_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    
    processes = sorted(zip(arrival_array, burst_array, range(n)))
    current_time = 0
    
    for arrival, burst, i in processes:
        if current_time < arrival:
            current_time = arrival
        start_time = current_time
        current_time += burst
        completion_array[i] = current_time
        turnaround_array[i] = completion_array[i] - arrival
        waiting_array[i] = turnaround_array[i] - burst
        gantt_chart.append((process_array[i], start_time, current_time))
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

def shortest_job_first(n, arrival_array, burst_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    remaining = set(range(n))
    current_time = 0
    
    while remaining:
        available = [(i, burst_array[i]) for i in remaining if arrival_array[i] <= current_time]
        if not available:
            current_time += 1
            continue
        i, burst = min(available, key=lambda x: x[1])
        start_time = current_time
        current_time += burst
        completion_array[i] = current_time
        turnaround_array[i] = completion_array[i] - arrival_array[i]
        waiting_array[i] = turnaround_array[i] - burst_array[i]
        gantt_chart.append((process_array[i], start_time, current_time))
        remaining.remove(i)
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

def shortest_job_first_preemptive(n, arrival_array, burst_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    remaining_burst = burst_array.copy()
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    current_time = 0
    prev_process = None
    start_time = 0
    
    while np.any(remaining_burst > 0):
        available = [(i, remaining_burst[i]) for i in range(n) if arrival_array[i] <= current_time and remaining_burst[i] > 0]
        if not available:
            current_time += 1
            continue
        curr_process, _ = min(available, key=lambda x: x[1])
        if prev_process is not None and prev_process != curr_process and start_time < current_time:
            gantt_chart.append((process_array[prev_process], start_time, current_time))
        remaining_burst[curr_process] -= 1
        if prev_process != curr_process:
            start_time = current_time
        current_time += 1
        if remaining_burst[curr_process] == 0:
            completion_array[curr_process] = current_time
            turnaround_array[curr_process] = completion_array[curr_process] - arrival_array[curr_process]
            waiting_array[curr_process] = turnaround_array[curr_process] - burst_array[curr_process]
        prev_process = curr_process
    
    if prev_process is not None and start_time < current_time:
        gantt_chart.append((process_array[prev_process], start_time, current_time))
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

def priority_scheduling(n, arrival_array, burst_array, priority_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    remaining = set(range(n))
    current_time = 0
    
    while remaining:
        available = [(i, priority_array[i]) for i in remaining if arrival_array[i] <= current_time]
        if not available:
            current_time += 1
            continue
        i, _ = min(available, key=lambda x: x[1])
        start_time = current_time
        current_time += burst_array[i]
        completion_array[i] = current_time
        turnaround_array[i] = completion_array[i] - arrival_array[i]
        waiting_array[i] = turnaround_array[i] - burst_array[i]
        gantt_chart.append((process_array[i], start_time, current_time))
        remaining.remove(i)
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

def priority_scheduling_preemptive(n, arrival_array, burst_array, priority_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    remaining_burst = burst_array.copy()
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    current_time = 0
    prev_process = None
    start_time = 0
    
    while np.any(remaining_burst > 0):
        available = [(i, priority_array[i]) for i in range(n) if arrival_array[i] <= current_time and remaining_burst[i] > 0]
        if not available:
            current_time += 1
            continue
        curr_process, _ = min(available, key=lambda x: x[1])
        if prev_process is not None and prev_process != curr_process and start_time < current_time:
            gantt_chart.append((process_array[prev_process], start_time, current_time))
        remaining_burst[curr_process] -= 1
        if prev_process != curr_process:
            start_time = current_time
        current_time += 1
        if remaining_burst[curr_process] == 0:
            completion_array[curr_process] = current_time
            turnaround_array[curr_process] = completion_array[curr_process] - arrival_array[curr_process]
            waiting_array[curr_process] = turnaround_array[curr_process] - burst_array[curr_process]
        prev_process = curr_process
    
    if prev_process is not None and start_time < current_time:
        gantt_chart.append((process_array[prev_process], start_time, current_time))
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

def round_robin(n, arrival_array, burst_array, time_quantum):
    process_array = np.array(["P" + str(i) for i in range(n)])
    remaining_burst = burst_array.copy()
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    queue = []
    current_time = 0
    start_times = {}
    
    while np.any(remaining_burst > 0) or queue:
        for i in range(n):
            if arrival_array[i] <= current_time and remaining_burst[i] > 0 and i not in queue and i not in start_times:
                queue.append(i)
        if not queue:
            current_time += 1
            continue
        curr_process = queue.pop(0)
        if curr_process not in start_times:
            start_times[curr_process] = current_time
        exec_time = min(time_quantum, remaining_burst[curr_process])
        gantt_chart.append((process_array[curr_process], current_time, current_time + exec_time))
        remaining_burst[curr_process] -= exec_time
        current_time += exec_time
        for i in range(n):
            if arrival_array[i] <= current_time and remaining_burst[i] > 0 and i not in queue and i != curr_process:
                queue.append(i)
        if remaining_burst[curr_process] > 0:
            queue.append(curr_process)
        else:
            completion_array[curr_process] = current_time
            turnaround_array[curr_process] = completion_array[curr_process] - arrival_array[curr_process]
            waiting_array[curr_process] = turnaround_array[curr_process] - burst_array[curr_process]
    
    return process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart

# Helper functions for visualization
def plot_gantt_chart(frame, gantt_chart):
    fig, ax = plt.subplots(figsize=(10, 4))
    for process, start, end in gantt_chart:
        ax.barh(process, end - start, left=start, height=0.5, color="#CC5500")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Process", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

def plot_histogram(frame, process_array, waiting_array):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(process_array, waiting_array, color="#CC5500")
    ax.set_xlabel("Process", fontsize=12)
    ax.set_ylabel("Waiting Time", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

# Visualization Function for Single Algorithm
def visualize_scheduling(n, process_array, arrival_array, burst_array, completion_array, turnaround_array, waiting_array, gantt_chart, algorithm_name):
    root = tk.Tk()
    root.title(f"{algorithm_name} Scheduling Visualization")
    root.configure(bg="#3f3f3f")
    root.state('zoomed')

    main_frame = tk.Frame(root, bg="#3f3f3f")
    main_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(main_frame, bg="#3f3f3f", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#3f3f3f")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Title
    algo_message = tk.Label(
        scrollable_frame,
        text=f"Visual Representation of Processes Running and Scheduling with {algorithm_name}",
        font=("Arial", 24, "bold"),
        bg="#3f3f3f",
        fg="white"
    )
    algo_message.pack(pady=20)

    # DataFrame Section (Moved Before Gantt Chart)
    df_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    df_frame.pack(pady=20, padx=(root.winfo_screenwidth()-900)//2)

    data = {
        "Process": process_array,
        "Arrival Time": arrival_array,
        "Burst Time": burst_array,
        "Completion Time": completion_array,
        "Turnaround Time": turnaround_array,
        "Waiting Time": waiting_array
    }
    df = pd.DataFrame(data)
    
    table_title = tk.Label(df_frame, text="Process Scheduling Metrics", font=("Arial", 26, "bold"), bg="#3f3f3f", fg="white")
    table_title.pack(pady=5)
    table = ttk.Treeview(df_frame, columns=list(df.columns), show="headings", height=min(10, n))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 18, "bold"))  # Increased font size
    style.configure("Treeview", font=("Arial", 16), rowheight=40)    # Increased font size and row height
    for col in df.columns:
        table.heading(col, text=col, anchor="center")
        table.column(col, width=180, anchor="center")  # Wider columns for readability
    for _, row in df.iterrows():
        table.insert("", "end", values=list(row))
    table.pack(fill="x")
    
    
    # Summary Section
    summary_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    summary_frame.pack(pady=20, padx=(root.winfo_screenwidth()-900)//2)
    throughput = calculate_throughput(n, completion_array)
    avg_wait = calculate_avg_waiting_time(waiting_array)
    avg_turn = calculate_avg_waiting_time(turnaround_array)
    tk.Label(summary_frame, text=f"Throughput: {throughput:.2f}", font=("Arial", 16), bg="#3f3f3f", fg="white").pack(pady=2)
    tk.Label(summary_frame, text=f"Avg Waiting Time: {avg_wait:.2f}", font=("Arial", 16), bg="#3f3f3f", fg="white").pack(pady=2)
    tk.Label(summary_frame, text=f"Avg Turnaround Time: {avg_turn:.2f}", font=("Arial", 16), bg="#3f3f3f", fg="white").pack(pady=2)

    message = tk.Label(
        scrollable_frame,
        text=f"Real Time Visualization Of Process Scheduling {algorithm_name}",
        font=("Arial", 24, "bold"),
        bg="#3f3f3f",
        fg="white"
    )
    message.pack(pady=20)
    # Animation Section
    canvas_width = 900
    bar_width = 400
    bar_height = 40
    spacing = 60
    animation_height = (n * spacing) + 150

    animation_canvas = tk.Canvas(scrollable_frame, width=canvas_width, height=animation_height, bg="#3f3f3f", highlightthickness=0)
    animation_canvas.pack(pady=10, padx=(root.winfo_screenwidth()-canvas_width)//2)

    process_bars = {}
    process_progress = {}
    process_status_texts = {}
    start_x = (canvas_width - bar_width) // 2

    total_burst_times = {}
    for i in range(n):
        process_name = process_array[i]
        total_burst_times[i] = sum(end - start for p, start, end in gantt_chart if p == process_name)
        process_progress[i] = 0

    for i in range(n):
        y1 = 100 + i * spacing
        y2 = y1 + bar_height
        animation_canvas.create_rectangle(start_x, y1, start_x + bar_width, y2, outline="white", width=2)
        bar = animation_canvas.create_rectangle(start_x, y1, start_x, y2, fill="gray", outline="white", width=2)
        text = animation_canvas.create_text(start_x - 50, (y1 + y2) // 2, text=process_array[i], font=("Arial", 14, "bold"), fill="white", anchor="e")
        status_text = animation_canvas.create_text(start_x + bar_width + 60, (y1 + y2) // 2, text="", font=("Arial", 12, "italic"), fill="#CC5500", anchor="w")
        process_bars[i] = (bar, start_x, y1, y2, bar_width)
        process_status_texts[i] = status_text

    # Additional UI Elements
    time_label = tk.Label(scrollable_frame, text="Time: 0", font=("Arial", 24, "bold"), bg="#3f3f3f", fg="white")
    time_label.pack(pady=10)
    completion_label = tk.Label(scrollable_frame, text="Completed Processes: ", font=("Arial", 16, "bold"), bg="#3f3f3f", fg="#00FF00")
    completion_label.pack(pady=10)
    timeline_label = tk.Label(scrollable_frame, text="Current Execution: None", font=("Arial", 16), bg="#3f3f3f", fg="white")
    timeline_label.pack(pady=10)

    # Gantt Chart and Histogram Frames
    gantt_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    gantt_frame.pack(pady=10, fill="x", padx=(root.winfo_screenwidth()-900)//2)
    hist_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    hist_frame.pack(pady=10, fill="x", padx=(root.winfo_screenwidth()-900)//2)

    # Buttons Frame
    buttons_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    buttons_frame.pack(pady=10)

    is_paused = tk.BooleanVar(value=False)
    ttk.Button(buttons_frame, text="Pause", command=lambda: is_paused.set(not is_paused.get()), style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, text="Export to CSV", command=lambda: export_to_csv(df, algorithm_name), style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, text="Back to Input", command=lambda: [root.destroy(), SchedulerVisualizer(tk.Tk()).show_input_page(algorithm_name)], style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, text="Back to Home", command=lambda: [root.destroy(), SchedulerVisualizer(tk.Tk())], style="Custom.TButton").pack(side="left", padx=10)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    completed_processes = []

    def update_visualization(current_time=0, gantt_idx=0):
        if gantt_idx >= len(gantt_chart):
            gantt_title = tk.Label(gantt_frame, text=f"Gantt Chart of {algorithm_name}", font=("Arial", 26, "bold"), bg="#3f3f3f", fg="white")
            gantt_title.pack(pady=5)
            plot_gantt_chart(gantt_frame, gantt_chart)

            hist_title = tk.Label(hist_frame, text="Histogram of Waiting Times per Process", font=("Arial", 26, "bold"), bg="#3f3f3f", fg="white")
            hist_title.pack(pady=5)
            plot_histogram(hist_frame, process_array, waiting_array)

            canvas.configure(scrollregion=canvas.bbox("all"))
            return

        process_name, start_time, end_time = gantt_chart[gantt_idx]
        process_idx = int(process_name[1:])
        bar, x1, y1, y2, width = process_bars[process_idx]
        burst_time = end_time - start_time
        
        while current_time < start_time:
            if is_paused.get():
                root.after(100, update_visualization, current_time, gantt_idx)
                return
            time_label.config(text=f"Time: {current_time}")
            timeline_label.config(text="Current Execution: Idle")
            root.update()
            time.sleep(0.8)
            current_time += 1

        total_burst = total_burst_times[process_idx]
        steps = 20
        animation_canvas.itemconfig(process_status_texts[process_idx], text="Running", fill="#CC5500")
        timeline_label.config(text=f"Current Execution: {process_name} (Start: {start_time}, End: {end_time})")
        
        for t in range(burst_time):
            if is_paused.get():
                root.after(100, update_visualization, current_time, gantt_idx)
                return
            process_progress[process_idx] += 1
            for step in range(steps):
                progress = min(process_progress[process_idx] / total_burst, 1.0)
                new_x2 = x1 + int(progress * width)
                animation_canvas.coords(bar, x1, y1, new_x2, y2)
                animation_canvas.itemconfig(bar, fill="#CC5500")
                root.update()
                time.sleep(0.8 / steps)
            current_time += 1
            time_label.config(text=f"Time: {current_time}")

        if process_progress[process_idx] >= total_burst:
            animation_canvas.itemconfig(bar, fill="blue")
            animation_canvas.itemconfig(process_status_texts[process_idx], text="Completed", fill="#00FF00", font=("Arial", 26))
            if process_name not in completed_processes:
                completed_processes.append(process_name)
                completion_label.config(text=f"Completed Processes: {', '.join(completed_processes)}")
        
        root.after(0, update_visualization, current_time, gantt_idx + 1)

    root.after(0, update_visualization)

    # Helper function for CSV export
    def export_to_csv(df, algo_name):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{algo_name}_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    root.mainloop()
    return process_array, completion_array, turnaround_array, waiting_array, gantt_chart
# Visualization Function for Comparison
def visualize_comparison(n, results):
    root = tk.Tk()
    root.title("Scheduling Algorithms Comparison")
    root.configure(bg="#3f3f3f")
    root.state('zoomed')

    main_frame = tk.Frame(root, bg="#3f3f3f")
    main_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(main_frame, bg="#3f3f3f", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#3f3f3f")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Title
    tk.Label(scrollable_frame, 
             text="Comparison of Scheduling Algorithms", 
             font=("Arial", 24, "bold"), 
             bg="#3f3f3f", 
             fg="white").pack(pady=20)

    # DataFrame Section
    metrics_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    metrics_frame.pack(pady=20, padx=50)

    tk.Label(metrics_frame, 
             text="Performance Metrics Comparison", 
             font=("Arial", 16, "bold"), 
             bg="#3f3f3f", 
             fg="white").pack(pady=5)

    data = {"Algorithm": [], "Throughput": [], "Avg Waiting Time": [], "Avg Turnaround Time": []}
    for algo, (_, _, _, completion_array, turnaround_array, waiting_array, _) in results.items():
        data["Algorithm"].append(algo)
        data["Throughput"].append(calculate_throughput(n, completion_array))
        data["Avg Waiting Time"].append(calculate_avg_waiting_time(waiting_array))
        data["Avg Turnaround Time"].append(calculate_avg_waiting_time(turnaround_array))

    df = pd.DataFrame(data)
    
    table = ttk.Treeview(metrics_frame, columns=list(df.columns), show="headings", height=len(results))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 18, "bold"))
    style.configure("Treeview", font=("Arial", 16), rowheight=40)
    for col in df.columns:
        table.heading(col, text=col, anchor="center")
        table.column(col, width=250, anchor="center")
    for _, row in df.iterrows():
        table.insert("", "end", values=list(row))
    table.pack(fill="x")

    # Highlight Best and Worst Performing Algorithms
    best_algo_idx = df["Avg Waiting Time"].idxmin()
    worst_algo_idx = df["Avg Waiting Time"].idxmax()
    best_algo = df.loc[best_algo_idx, "Algorithm"]
    worst_algo = df.loc[worst_algo_idx, "Algorithm"]
    performance_label = tk.Label(metrics_frame,
                                text=f"Best: {best_algo} (Lowest Avg Waiting Time) | Worst: {worst_algo} (Highest Avg Waiting Time)",
                                font=("Arial", 14, "italic"),
                                bg="#3f3f3f",
                                fg="#00FF00")
    performance_label.pack(pady=10)

    # Calculate max width (47% of screen width)
    screen_width = root.winfo_screenwidth()
    max_graph_width = screen_width * 0.47 / 100  # Convert to inches (assuming 100 DPI for simplicity)

    # Gantt Charts and Histograms Section (Side by Side)
    for algo, (process_array, arrival_array, burst_array, completion_array, 
              turnaround_array, waiting_array, gantt_chart) in results.items():
        algo_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        algo_frame.pack(pady=20, fill="x", padx=50)

        # Algorithm Name Centered Above Gantt Chart
        tk.Label(algo_frame, 
                 text=algo, 
                 font=("Arial", 18, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(pady=5)

        # Side-by-Side Layout
        chart_frame = tk.Frame(algo_frame, bg="#3f3f3f")
        chart_frame.pack(fill="x")

        # Gantt Chart (Left)
        gantt_subframe = tk.Frame(chart_frame, bg="#3f3f3f")
        gantt_subframe.pack(side="left", padx=10, fill="x", expand=True)
        tk.Label(gantt_subframe, 
                 text="Gantt Chart", 
                 font=("Arial", 14, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(pady=5)
        # Limit width to 47% of screen
        plot_gantt_chart(gantt_subframe, gantt_chart, max_width=max_graph_width)

        # Histogram (Right)
        hist_subframe = tk.Frame(chart_frame, bg="#3f3f3f")
        hist_subframe.pack(side="left", padx=10, fill="x", expand=True)
        tk.Label(hist_subframe, 
                 text="Waiting Time Histogram", 
                 font=("Arial", 14, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(pady=5)
        # Limit width to 47% of screen
        plot_histogram(hist_subframe, process_array, waiting_array, max_width=max_graph_width)

    # Summary Statistics Section
    summary_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    summary_frame.pack(pady=20, padx=50)

    tk.Label(summary_frame,
             text="Summary Statistics",
             font=("Arial", 16, "bold"),
             bg="#3f3f3f",
             fg="white").pack(pady=5)

    summary_text = tk.Text(summary_frame, height=4, width=60, font=("Arial", 14), bg="#2d2d2d", fg="white")
    summary_text.pack(pady=5)
    summary_text.insert(tk.END, f"Total Algorithms Compared: {len(results)}\n")
    summary_text.insert(tk.END, f"Average Throughput: {df['Throughput'].mean():.2f}\n")
    summary_text.insert(tk.END, f"Average Waiting Time Across All: {df['Avg Waiting Time'].mean():.2f}\n")
    summary_text.insert(tk.END, f"Average Turnaround Time Across All: {df['Avg Turnaround Time'].mean():.2f}")
    summary_text.configure(state="disabled")

    # Comparison Bar Chart
    comparison_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    comparison_frame.pack(pady=20, padx=50)
    tk.Label(comparison_frame, 
             text="Average Waiting Time Comparison", 
             font=("Arial", 16, "bold"), 
             bg="#3f3f3f", 
             fg="white").pack(pady=5)
    fig, ax = plt.subplots(figsize=(min(max_graph_width, 10), 4))  # Limit width to 47% of screen
    ax.bar(df["Algorithm"], df["Avg Waiting Time"], color="#CC5500")
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Avg Waiting Time", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=comparison_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill="x")

    # Buttons Frame
    buttons_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    buttons_frame.pack(pady=20)

    zoom_level = tk.DoubleVar(value=1.0)
    ttk.Button(buttons_frame, 
               text="Zoom In", 
               command=lambda: adjust_zoom(zoom_level, 1.2, results, scrollable_frame, max_graph_width),
               style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, 
               text="Zoom Out", 
               command=lambda: adjust_zoom(zoom_level, 0.8, results, scrollable_frame, max_graph_width),
               style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, 
               text="Export Detailed Results", 
               command=lambda: export_detailed_results(results),
               style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, 
               text="Back to Compare", 
               command=lambda: [root.destroy(), SchedulerVisualizer(tk.Tk()).show_compare_input_page()],
               style="Custom.TButton").pack(side="left", padx=10)
    ttk.Button(buttons_frame, 
               text="Back to Home", 
               command=lambda: [root.destroy(), SchedulerVisualizer(tk.Tk())], 
               style="Custom.TButton").pack(side="left", padx=10)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    root.mainloop()

# Modified plot_gantt_chart with max_width parameter
def plot_gantt_chart(frame, gantt_chart, max_width=None):
    if max_width is None:
        max_width = 10  # Default width in inches
    fig, ax = plt.subplots(figsize=(min(max_width, 10), 4))  # Limit width
    for process, start, end in gantt_chart:
        ax.barh(process, end - start, left=start, height=0.5, color="#CC5500")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Process", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)

# Modified plot_histogram with max_width parameter
def plot_histogram(frame, process_array, waiting_array, max_width=None):
    if max_width is None:
        max_width = 10  # Default width in inches
    fig, ax = plt.subplots(figsize=(min(max_width, 10), 4))  # Limit width
    ax.bar(process_array, waiting_array, color="#CC5500")
    ax.set_xlabel("Process", fontsize=12)
    ax.set_ylabel("Waiting Time", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)

# Helper function for exporting detailed results
def export_detailed_results(results):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"scheduling_detailed_comparison_{timestamp}.csv"
        detailed_data = []
        for algo, (process_array, arrival_array, burst_array, completion_array, 
                  turnaround_array, waiting_array, _) in results.items():
            for i in range(len(process_array)):
                detailed_data.append({
                    "Algorithm": algo,
                    "Process": process_array[i],
                    "Arrival Time": arrival_array[i],
                    "Burst Time": burst_array[i],
                    "Completion Time": completion_array[i],
                    "Turnaround Time": turnaround_array[i],
                    "Waiting Time": waiting_array[i]
                })
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Detailed results exported to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export: {str(e)}")

# Helper function for exporting metrics to CSV
def export_to_csv(df):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"scheduling_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Results exported to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export: {str(e)}")

# Modified adjust_zoom with max_width constraint
def adjust_zoom(zoom_level, factor, results, scrollable_frame, max_graph_width):
    new_zoom = zoom_level.get() * factor
    if 0.5 <= new_zoom <= 2.0:  # Limit zoom range
        zoom_level.set(new_zoom)
        for widget in scrollable_frame.winfo_children():
            if isinstance(widget, tk.Frame) and widget != buttons_frame:
                for child in widget.winfo_children():
                    if isinstance(child, FigureCanvasTkAgg):
                        child.get_tk_widget().pack_forget()
                        fig = child.figure
                        current_width = fig.get_size_inches()[0]
                        new_width = min(current_width * factor, max_graph_width)
                        fig.set_size_inches(new_width, 4 * new_zoom)
                        child.draw()
                        child.get_tk_widget().pack(fill="x")
        canvas.configure(scrollregion=canvas.bbox("all"))
# UI/UX Implementation
class SchedulerVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Process Scheduling Visualizer")
        self.root.geometry("1200x800")
        self.theme = "dark"
        self.algorithm = None
        self.arrival_array = None
        self.burst_array = None
        
        self.setup_styles()
        self.create_home_page()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 36, "bold"), padding=10)
        self.style.configure("Custom.TButton", font=("Helvetica", 24, "bold"), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 36))
        self.style.configure("Treeview.Heading", font=("Helvetica", 36, "bold"))

    def create_home_page(self):
        self.home_frame = tk.Frame(self.root, bg="#1E1E1E")  # Dark grey background
        self.home_frame.pack(fill="both", expand=True)

        # Top bar for toggle theme
        top_bar = tk.Frame(self.home_frame, bg="#2C2C2C", bd=0, relief="flat")
        top_bar.pack(fill="x", pady=(0, 2))

        self.theme_emoji = tk.StringVar(value="☾" if self.theme == "dark" else "☀")
        self.toggle_button = tk.Button(top_bar, textvariable=self.theme_emoji, 
                                       command=self.toggle_theme, 
                                       font=("Roboto", 20, "bold"), 
                                       bg="#3A3A3A", fg="#FFFFFF", 
                                       bd=0, relief="flat", 
                                       activebackground="#4A4A4A", 
                                       highlightthickness=0)
        self.toggle_button.place(x=20, y=10, width=50, height=50)
        self.toggle_button.bind("<Enter>", lambda e: self.toggle_button.config(bg="#4A4A4A"))
        self.toggle_button.bind("<Leave>", lambda e: self.toggle_button.config(bg="#3A3A3A"))

        # Main content area
        content_frame = tk.Frame(self.home_frame, bg="#040406")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Heading
        title = tk.Label(content_frame, text="Process Scheduling Visualizer", 
                         font=("Roboto", 48, "bold"), 
                         fg="#FFFFFF", bg="#040406")
        title.pack(pady=(20, 50))  # Increased padding below for separation
       
        # Container for buttons with spacing
        custom_text = tk.Label(content_frame, 
                       text="Choose a process scheduling algorithm and see it in action with real-time visualization. Compare multiple algorithms to analyze response time, waiting time, and CPU efficiency. Start exploring now!", 
                       font=("Helvetica", 24, "bold"), 
                       fg="white", 
                       bg="#040406",
                       wraplength=800,  # Adjust this value as needed
                       justify="center")
        custom_text.pack(pady=20)
        button_frame = tk.Frame(content_frame, bg="#040406")
        button_frame.pack(pady=20)

        # Select Algorithm Dropdown
        self.algo_var = tk.StringVar(value="Select an Algorithm")
        select_menu = ttk.OptionMenu(button_frame, self.algo_var, "Select an Algorithm", 
                                     "FCFS", "SJF (Non-Preemptive)", "SJF (Preemptive)", 
                                     "Priority (Non-Preemptive)", "Priority (Preemptive)", 
                                     "Round Robin", 
                                     command=self.handle_algo_selection)
        select_menu.config(width=25)
        select_menu.pack(side="left", padx=40)  # Increased padx for spacing
        style = ttk.Style()
        style.configure("TMenubutton", font=("Roboto", 20), 
                        background="white", foreground="black", 
                        padding=10)
        style.map("TMenubutton", background=[("active", "white")])

        # Compare Algorithms Button
        compare_button = ttk.Button(button_frame, 
                            text="Compare Algorithms", 
                            style="Custom.TButton", 
                            command=self.show_compare_input_page)
        style.configure("Custom.TButton", font=("Roboto", 20), 
                background="#3A3A3A", foreground="#FFFFFF", 
                padding=10)
        style.map("Custom.TButton", background=[("active", "#4A4A4A")])
        compare_button = ttk.Button(button_frame, 
                                    text="Compare Algorithms", 
                                    style="Custom.TButton", 
                                    command=self.show_compare_input_page)
        compare_button.pack(side="left", padx=40)  # Increased padx for spacing
        style.configure("Custom.TButton", font=("Roboto", 20), 
                        background="grey", foreground="black", 
                        padding=10)
        style.map("Custom.TButton", background=[("active", "grey")])

        # Subtle underline animation below title
        def animate_underline():
            underline = tk.Frame(content_frame, bg="#CC5500", height=3)
            underline.pack(fill="x", pady=(0, 20))
            for width in range(0, 600, 15):
                underline.config(width=width)
                self.root.update()
                time.sleep(0.01)

        self.root.after(100, animate_underline)
        self.apply_theme()
    def handle_algo_selection(self, algo):
        if algo == "Compare Algorithms":
            self.show_compare_input_page()
        elif algo != "Select Algorithm":
            self.show_input_page(algo)

    def show_input_page(self, algo):
        if algo == "Select Algorithm":
            return
        
        self.algorithm = algo
        self.home_frame.pack_forget()
        
        self.input_frame = tk.Frame(self.root, bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0")
        self.input_frame.pack(fill="both", expand=True, padx=40, pady=40)

        tk.Label(self.input_frame, 
                 text=f"{algo} Input", font=("Helvetica", 48, "bold"), 
                 bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                 fg="white" if self.theme == "dark" else "black").pack(pady=(20, 30))

        input_style = {"font": ("Helvetica", 36), 
                       "bg": "#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                       "fg": "white" if self.theme == "dark" else "black"}

        tk.Label(self.input_frame, text="Number of Processes:", **input_style).pack(pady=(10, 5))
        self.num_processes = tk.Entry(self.input_frame, font=("Helvetica", 32), width=20, 
                                      bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                      fg="white" if self.theme == "dark" else "black", 
                                      insertbackground="white" if self.theme == "dark" else "black")
        self.num_processes.pack(pady=10)

        tk.Label(self.input_frame, text="Arrival Times (space-separated):", **input_style).pack(pady=(10, 5))
        self.arrival_entry = tk.Entry(self.input_frame, font=("Helvetica", 32), width=20, 
                                      bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                      fg="white" if self.theme == "dark" else "black", 
                                      insertbackground="white" if self.theme == "dark" else "black")
        self.arrival_entry.pack(pady=10)

        tk.Label(self.input_frame, text="Burst Times (space-separated):", **input_style).pack(pady=(10, 5))
        self.burst_entry = tk.Entry(self.input_frame, font=("Helvetica", 32), width=20, 
                                    bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                    fg="white" if self.theme == "dark" else "black", 
                                    insertbackground="white" if self.theme == "dark" else "black")
        self.burst_entry.pack(pady=10)

        dwindled = None

        if "Priority" in algo:
            tk.Label(self.input_frame, text="Priorities (space-separated):", **input_style).pack(pady=(10, 5))
            self.priority_entry = tk.Entry(self.input_frame, font=("Helvetica", 32), width=20, 
                                           bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                           fg="white" if self.theme == "dark" else "black", 
                                           insertbackground="white" if self.theme == "dark" else "black")
            self.priority_entry.pack(pady=10)
        elif algo == "Round Robin":
            tk.Label(self.input_frame, text="Time Quantum:", **input_style).pack(pady=(10, 5))
            self.quantum_entry = tk.Entry(self.input_frame, font=("Helvetica", 32), width=20, 
                                          bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                          fg="white" if self.theme == "dark" else "black", 
                                          insertbackground="white" if self.theme == "dark" else "black")
            self.quantum_entry.pack(pady=10)

        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=("Helvetica", 24, "bold"), padding=10)
        
        ttk.Button(self.input_frame, text="Run Simulation", style="Custom.TButton", 
                   command=self.run_simulation).pack(pady=20)
        ttk.Button(self.input_frame, text="Back to Home", style="Custom.TButton", 
                   command=self.back_to_home).pack(pady=10)

        self.apply_theme()

    def show_compare_input_page(self):
        self.home_frame.pack_forget()
        
        self.compare_input_frame = tk.Frame(self.root, bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0")
        self.compare_input_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(self.compare_input_frame, bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0")
        scrollbar = tk.Scrollbar(self.compare_input_frame, orient="vertical", command=canvas.yview)
        
        scrollable_frame = tk.Frame(canvas, bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
            canvas.yview_moveto(0)  # Ensure the page opens at the top
        
        canvas.bind("<Configure>", on_canvas_configure)
        
        container = tk.Frame(scrollable_frame, bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0")
        container.pack(expand=True)
        
        tk.Label(container, 
                 text="Compare Scheduling Algorithms", 
                 font=("Helvetica", 26, "bold"), 
                 bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                 fg="white" if self.theme == "dark" else "black").pack(pady=(20, 30))

        input_style = {"font": ("Helvetica", 26), 
                       "bg": "#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                       "fg": "white" if self.theme == "dark" else "black"}

        for text, var_name in [
            ("Number of Processes:", "num_processes"),
            ("Arrival Times (space-separated):", "arrival_entry"),
            ("Burst Times (space-separated):", "burst_entry"),
            ("Priorities (space-separated, optional):", "priority_entry"),
            ("Time Quantum (for Round Robin, optional):", "quantum_entry")
        ]:
            tk.Label(container, text=text, **input_style).pack(pady=(10, 5))
            setattr(self, var_name, tk.Entry(container, font=("Helvetica", 32), width=20, 
                                             bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                             fg="white" if self.theme == "dark" else "black", 
                                             insertbackground="white" if self.theme == "dark" else "black"))
            getattr(self, var_name).pack(pady=10)

        tk.Label(container, text="Select Algorithms to Compare:", **input_style).pack(pady=(20, 10))
        self.algo_vars = {}
        algorithms = ["FCFS", "SJF (Non-Preemptive)", "SJF (Preemptive)", 
                      "Priority (Non-Preemptive)", "Priority (Preemptive)", "Round Robin"]
        for algo in algorithms:
            var = tk.BooleanVar(value=False)
            self.algo_vars[algo] = var
            chk = tk.Checkbutton(container, text=algo, variable=var, 
                                font=("Helvetica", 28, "bold"), 
                                bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                                fg="white" if self.theme == "dark" else "black", 
                                selectcolor="#404040" if self.theme == "dark" else "#e0e0e0", 
                                padx=20, pady=10)
            chk.pack(pady=10)

        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=("Helvetica", 28, "bold"), padding=20)
        
        ttk.Button(container, text="Run Comparison", style="Custom.TButton", 
                   command=self.run_comparison).pack(pady=20)
        ttk.Button(container, text="Back to Home", style="Custom.TButton", 
                   command=self.back_to_home_from_compare).pack(pady=10)
        
        self.apply_theme()




    def back_to_home(self):
        self.input_frame.pack_forget()
        self.create_home_page()

    def back_to_home_from_compare(self):
        self.compare_input_frame.pack_forget()
        self.create_home_page()

    def run_simulation(self):
        try:
            n = int(self.num_processes.get())
            self.arrival_array = np.array([int(x) for x in self.arrival_entry.get().split()][:n])
            self.burst_array = np.array([int(x) for x in self.burst_entry.get().split()][:n])
            
            if len(self.arrival_array) != n or len(self.burst_array) != n:
                raise ValueError("Mismatch in number of processes and input arrays")

            if self.algorithm == "FCFS":
                result = first_come_first_serve(n, self.arrival_array, self.burst_array)
            elif self.algorithm == "SJF (Non-Preemptive)":
                result = shortest_job_first(n, self.arrival_array, self.burst_array)
            elif self.algorithm == "SJF (Preemptive)":
                result = shortest_job_first_preemptive(n, self.arrival_array, self.burst_array)
            elif self.algorithm == "Priority (Non-Preemptive)":
                priority_array = np.array([int(x) for x in self.priority_entry.get().split()][:n])
                result = priority_scheduling(n, self.arrival_array, self.burst_array, priority_array)
            elif self.algorithm == "Priority (Preemptive)":
                priority_array = np.array([int(x) for x in self.priority_entry.get().split()][:n])
                result = priority_scheduling_preemptive(n, self.arrival_array, self.burst_array, priority_array)
            elif self.algorithm == "Round Robin":
                time_quantum = int(self.quantum_entry.get())
                result = round_robin(n, self.arrival_array, self.burst_array, time_quantum)
            else:
                return

            visualize_scheduling(n, *result, self.algorithm)
            self.input_frame.pack_forget()
            self.display_results(*result)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_comparison(self):
        try:
            n = int(self.num_processes.get())
            arrival_array = np.array([int(x) for x in self.arrival_entry.get().split()][:n])
            burst_array = np.array([int(x) for x in self.burst_entry.get().split()][:n])
            priority_array = np.array([int(x) if x else 0 for x in self.priority_entry.get().split()][:n]) if self.priority_entry.get() else None
            time_quantum = int(self.quantum_entry.get()) if self.quantum_entry.get() else 4

            if len(arrival_array) != n or len(burst_array) != n:
                raise ValueError("Mismatch in number of processes and input arrays")

            selected_algos = [algo for algo, var in self.algo_vars.items() if var.get()]
            if len(selected_algos) < 2:
                raise ValueError("Select at least two algorithms to compare")

            results = {}
            for algo in selected_algos:
                if algo == "FCFS":
                    results[algo] = first_come_first_serve(n, arrival_array, burst_array)
                elif algo == "SJF (Non-Preemptive)":
                    results[algo] = shortest_job_first(n, arrival_array, burst_array)
                elif algo == "SJF (Preemptive)":
                    results[algo] = shortest_job_first_preemptive(n, arrival_array, burst_array)
                elif algo == "Priority (Non-Preemptive)":
                    results[algo] = priority_scheduling(n, arrival_array, burst_array, priority_array or np.zeros(n))
                elif algo == "Priority (Preemptive)":
                    results[algo] = priority_scheduling_preemptive(n, arrival_array, burst_array, priority_array or np.zeros(n))
                elif algo == "Round Robin":
                    results[algo] = round_robin(n, arrival_array, burst_array, time_quantum)

            visualize_comparison(n, results)
            self.compare_input_frame.pack_forget()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, process_array, completion_array, turnaround_array, waiting_array, gantt_chart):
        self.result_frame = tk.Frame(self.root, relief="raised", borderwidth=2)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(self.result_frame, text=f"{self.algorithm} Results", font=("Helvetica", 18, "bold")).pack(pady=15)

        data = pd.DataFrame({
            "Process": process_array,
            "Arrival": self.arrival_array,
            "Burst": self.burst_array,
            "Completion": completion_array,
            "Turnaround": turnaround_array,
            "Waiting": waiting_array
        })
        table_frame = tk.Frame(self.result_frame, relief="raised", borderwidth=2)
        table_frame.pack(fill="x", pady=10)
        table = ttk.Treeview(table_frame, columns=list(data.columns), show="headings")
        for col in data.columns:
            table.heading(col, text=col)
            table.column(col, width=120, anchor="center")
        for _, row in data.iterrows():
            table.insert("", "end", values=list(row))
        table.pack(fill="x")

        gantt_frame = tk.Frame(self.result_frame, relief="raised", borderwidth=2)
        gantt_frame.pack(fill="both", expand=True, pady=10)
        fig, ax = plt.subplots(figsize=(10, 4))
        for process, start, end in gantt_chart:
            ax.barh(process, end - start, left=start, height=0.5, color="#CC5500")
        ax.set_xlabel("Time")
        ax.set_ylabel("Process")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=gantt_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        metrics_frame = tk.Frame(self.result_frame, relief="raised", borderwidth=2)
        metrics_frame.pack(fill="x", pady=10)
        throughput = calculate_throughput(len(process_array), completion_array)
        avg_wait = calculate_avg_waiting_time(waiting_array)
        tk.Label(metrics_frame, text=f"Throughput: {throughput:.2f}", font=("Helvetica", 14)).pack(pady=5)
        tk.Label(metrics_frame, text=f"Avg Waiting Time: {avg_wait:.2f}", font=("Helvetica", 14)).pack(pady=5)

        ttk.Button(self.result_frame, text="Back to Home", command=self.back_to_home_from_results).pack(pady=15)

        self.apply_theme()

    def back_to_home_from_results(self):
        self.result_frame.pack_forget()
        self.create_home_page()

    def update_theme_button(self):
        self.theme_emoji.set("☾" if self.theme == "dark" else "☀")
        self.toggle_button.config(bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0",
                                 fg="white" if self.theme == "dark" else "black")

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.update_theme_button()
        self.apply_theme()

    def apply_theme(self):
        bg = "#040406" if self.theme == "dark" else "#f0f0f0"
        fg = "white" if self.theme == "dark" else "black"
        
        self.root.configure(bg=bg)
        for frame in [self.home_frame, getattr(self, 'input_frame', None), 
                      getattr(self, 'result_frame', None), getattr(self, 'compare_input_frame', None)]:
            if frame and frame.winfo_exists():
                frame.configure(bg=bg)
                for widget in frame.winfo_children():
                    if isinstance(widget, (tk.Label, tk.Text)):
                        widget.configure(bg=bg, fg=fg)
                    elif isinstance(widget, tk.Entry):
                        widget.configure(bg=bg, fg=fg, insertbackground=fg)
                    elif isinstance(widget, tk.Frame):
                        widget.configure(bg=bg)

if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerVisualizer(root)
    root.mainloop()