import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import seaborn as sns


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
    priority_array = ["-"] * n
    processes = [(arrival_array[i], i, burst_array[i]) for i in range(n)]
    processes.sort(key=lambda x: (x[0], x[1]))
    current_time = 0
    
    for arrival, i, burst in processes:
        if current_time < arrival:
            current_time = arrival
        start_time = current_time
        current_time += burst
        completion_array[i] = current_time
        turnaround_array[i] = completion_array[i] - arrival
        waiting_array[i] = turnaround_array[i] - burst
        gantt_chart.append((process_array[i], start_time, current_time))
    
    return process_array, arrival_array, burst_array,priority_array, completion_array, turnaround_array, waiting_array, gantt_chart

def shortest_job_first(n, arrival_array, burst_array):
    process_array = np.array(["P" + str(i) for i in range(n)])
    completion_array = np.zeros(n, dtype=int)
    turnaround_array = np.zeros(n, dtype=int)
    waiting_array = np.zeros(n, dtype=int)
    gantt_chart = []
    remaining = set(range(n))
    current_time = 0
    priority_array = ["-"] * n
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
    
    return process_array, arrival_array, burst_array,priority_array, completion_array, turnaround_array, waiting_array, gantt_chart

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
    priority_array = ["-"] * n
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
    
    return process_array, arrival_array, burst_array,priority_array, completion_array, turnaround_array, waiting_array, gantt_chart

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
    
    return process_array, arrival_array, burst_array, priority_array,completion_array, turnaround_array, waiting_array, gantt_chart

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
    
    return process_array, arrival_array, burst_array, priority_array,completion_array, turnaround_array, waiting_array, gantt_chart

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
    priority_array = ["-"] * n
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
    
    return process_array, arrival_array, burst_array,priority_array, completion_array, turnaround_array, waiting_array, gantt_chart

def plot_gantt_chart(frame, gantt_chart):
    fig, ax = plt.subplots(figsize=(7, 4))
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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(process_array, waiting_array, color="#CC5500")
    ax.set_xlabel("Process", fontsize=12)
    ax.set_ylabel("Waiting Time", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

comparison_metrics = {}

def visualize_scheduling(n, process_array, arrival_array, burst_array,priority_array, completion_array, turnaround_array, waiting_array, gantt_chart, algorithm_name):
    if isinstance(process_array, np.ndarray):
        process_array = process_array.tolist()
    if isinstance(arrival_array, np.ndarray):
        arrival_array = arrival_array.tolist()
    if isinstance(burst_array, np.ndarray):
        burst_array = burst_array.tolist()
    if isinstance(completion_array, np.ndarray):
        completion_array = completion_array.tolist()
    if isinstance(turnaround_array, np.ndarray):
        turnaround_array = turnaround_array.tolist()
    if isinstance(waiting_array, np.ndarray):
        waiting_array = waiting_array.tolist()
    if isinstance(gantt_chart, np.ndarray):
        gantt_chart = gantt_chart.tolist()

    gantt_chart = [(str(process), float(start), float(end)) for process, start, end in gantt_chart]

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

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def center_content(event):
        canvas_width = event.width
        frame_width = scrollable_frame.winfo_reqwidth()
        x_position = max(0, (canvas_width - frame_width) // 2)
        canvas.coords(canvas_window, x_position, 0)

    canvas.bind("<Configure>", center_content)

    algo_message = tk.Label(
        scrollable_frame,
        text=f"Visual Representation of Processes Running and Scheduling with {algorithm_name}",
        font=("Arial", 20, "bold"), 
        bg="#3f3f3f",
        fg="#03DAC6"
    )
    algo_message.pack(pady=20)  

    df_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    df_frame.pack(pady=10, fill="x")  

    data = {
        "Process": process_array,
        "Arrival Time": arrival_array,
        "Burst Time": burst_array,
        "Priority_array":priority_array,
        "Completion Time": completion_array,
        "Turnaround Time": turnaround_array,
        "Waiting Time": waiting_array
    }
    df = pd.DataFrame(data)
    
    table_title = tk.Label(df_frame, text="Process Scheduling Metrics", font=("Arial", 18, "bold"), bg="#3f3f3f", fg="#BB86FC")  
    table_title.pack(pady=5, expand=True)  # Reduced from 10 to 5
    table = ttk.Treeview(df_frame, columns=list(df.columns), show="headings", height=min(10, n))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 14, "bold"), foreground="#FFFFFF", background="#4A4A4A") 
    style.configure("Treeview", font=("Arial", 12), rowheight=30, foreground="#FFFFFF", background="#2D2D2D")  
    for col in df.columns:
        table.heading(col, text=col, anchor="center")
        table.column(col, width=180, anchor="center")
    for _, row in df.iterrows():
        table.insert("", "end", values=list(row))
    table.pack(expand=True)

    summary_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    summary_frame.pack(pady=10, fill="x")  # Reduced from 20 to 10
    throughput = calculate_throughput(n, completion_array)
    avg_wait = calculate_avg_waiting_time(waiting_array)
    avg_turn = calculate_avg_waiting_time(turnaround_array)
    total_time = max(completion_array) if completion_array else 1
    energy_result = calculate_energy(gantt_chart, total_time)
    total_energy = round(energy_result["total_energy"], 2)

    tk.Label(summary_frame, text=f"Throughput: {throughput:.2f}", font=("Arial", 14), bg="#3f3f3f", fg="#FFFF00").pack(pady=5, expand=True) 
    tk.Label(summary_frame, text=f"Avg Waiting Time: {avg_wait:.2f}", font=("Arial", 14), bg="#3f3f3f", fg="#FFA500").pack(pady=5, expand=True)  
    tk.Label(summary_frame, text=f"Avg Turnaround Time: {avg_turn:.2f}", font=("Arial", 14), bg="#3f3f3f", fg="#FF0000").pack(pady=5, expand=True)  
    tk.Label(summary_frame, text=f"Total Energy Consumption: {total_energy:.2f} J", font=("Arial", 14), bg="#3f3f3f", fg="#00FF00").pack(pady=5, expand=True)  

    comparison_metrics[algorithm_name] = {
        "Avg Waiting Time": avg_wait,
        "Avg Turnaround Time": avg_turn,
        "Throughput": throughput,
        "Energy Consumption (J)": total_energy
    }

    energy_button_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    energy_button_frame.pack(pady=5, fill="x")
    ttk.Button(energy_button_frame, 
               text="View Energy Consumption", 
               command=lambda: show_energy_consumption_view(algorithm_name, gantt_chart, total_time),
               style="Custom.TButton").pack(expand=True)
    style.configure("Custom.TButton", font=("Arial", 12), padding=8)  

    save_metrics_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    save_metrics_frame.pack(pady=10, fill="x")
    ttk.Button(save_metrics_frame, 
               text="Save Comparison Metrics", 
               command=lambda: save_comparison_metrics(),
               style="Custom.TButton").pack(expand=True)

    def save_comparison_metrics():
        if not comparison_metrics:
            messagebox.showwarning("Warning", "No metrics to save.")
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_metrics_{timestamp}.csv"
            df_metrics = pd.DataFrame(comparison_metrics).T
            df_metrics.to_csv(filename)
            messagebox.showinfo("Success", f"Comparison metrics saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save metrics: {str(e)}")

    message = tk.Label(
        scrollable_frame,
        text=f"Real Time Visualization Of Process Scheduling {algorithm_name}",
        font=("Arial", 18, "bold"),  
        bg="#3f3f3f",
        fg="#03DAC6"
    )
    message.pack(pady=0)  

    # Animation Section
    canvas_width = 1200
    bar_width = 400
    bar_height = 40
    spacing = 55 
    animation_height = (n * spacing) + 10

    animation_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    animation_frame.pack(pady=2, fill="x")  
    animation_canvas = tk.Canvas(animation_frame, width=canvas_width, height=animation_height, bg="#3f3f3f", highlightthickness=0)
    animation_canvas.pack()

    process_bars = {}
    process_progress = {}
    process_status_texts = {}
    process_segments = {}
    start_x = (canvas_width - bar_width) // 2

    total_burst_times = {}
    process_map = {p: i for i, p in enumerate(process_array)}
    for i in range(n):
        process_name = process_array[i]
        total_burst = sum(end - start for p, start, end in gantt_chart if p == process_name)
        total_burst_times[i] = total_burst
        process_progress[i] = 0
        num_segments = int(total_burst)
        segment_width = bar_width / num_segments if num_segments > 0 else bar_width
        process_segments[i] = (num_segments, segment_width)

    for i in range(n):
        y1 = 10 + i * spacing
        y2 = y1 + bar_height
        animation_canvas.create_rectangle(start_x, y1, start_x + bar_width, y2, outline="white", width=2)
        num_segments, segment_width = process_segments[i]
        for s in range(1, num_segments):
            x = start_x + s * segment_width
            animation_canvas.create_line(x, y1, x, y2, fill="white", dash=(2, 2))
        bar = animation_canvas.create_rectangle(start_x, y1, start_x, y2, fill="gray", outline="white", width=2)
        text = animation_canvas.create_text(start_x - 50, (y1 + y2) // 2, text=process_array[i], font=("Arial", 12, "bold"), fill="white", anchor="e")  
        status_text = animation_canvas.create_text(start_x + bar_width + 60, (y1 + y2) // 2, text="", font=("Arial", 16), fill="#CC5500", anchor="w") 
        process_bars[i] = (bar, start_x, y1, y2, bar_width)
        process_status_texts[i] = status_text

    timeline_height = 175
    timeline_canvas = tk.Canvas(animation_frame, width=canvas_width, height=timeline_height, bg="#3f3f3f", highlightthickness=0)
    timeline_canvas.pack(pady=0) 

    max_time = max(end for _, _, end in gantt_chart) if gantt_chart else max(completion_array) if completion_array else 1.0

   
    timeline_start_x = 50
    timeline_end_x = timeline_start_x + 1000  
    timeline_y = timeline_height // 2
    timeline_canvas.create_line(timeline_start_x, timeline_y, timeline_end_x, timeline_y, fill="white", width=2)
    
  
    time_interval = max_time / 15 if max_time > 0 else 1
    for t in range(16):  # 0 to 15 inclusive
        scaled_time = t * time_interval
        x = timeline_start_x + (scaled_time / max_time) * (timeline_end_x - timeline_start_x)
        timeline_canvas.create_line(x, timeline_y - 5, x, timeline_y + 5, fill="white")
        timeline_canvas.create_text(x, timeline_y + 20, text=f"{scaled_time:.1f}", font=("Arial", 8), fill="white")  # Reduced from 10 to 8

  
    process_colors = {}
    colors = ["#FF5555", "#55FF55", "#5555FF", "#FFFF55", "#FF55FF", 
              "#00FFFF", "#006400", "#FFA500", "#800080", "#FF4500"]
    unique_processes = set(process for process, _, _ in gantt_chart)
    for i, process in enumerate(unique_processes):
        process_colors[process] = colors[i % len(colors)]

   
    process_y_positions = {}
    y_spacing = 25  
    for i, process in enumerate(unique_processes):
        process_y_positions[process] = timeline_y - 50 + (i * y_spacing)

    for process, start, end in gantt_chart:
        x1 = timeline_start_x + (start / max_time) * (timeline_end_x - timeline_start_x)
        x2 = timeline_start_x + (end / max_time) * (timeline_end_x - timeline_start_x)
        y = process_y_positions[process]
        timeline_canvas.create_rectangle(x1, y - 5, x2, y + 5, fill=process_colors[process], outline="white")
        timeline_canvas.create_text(x1 - 30, y, text=process, font=("Arial", 8), fill="white", anchor="e")  

    time_label = tk.Label(scrollable_frame, text="Time: 0", font=("Arial", 18, "bold"), bg="#3f3f3f", fg="#FFFFFF") 
    time_label.pack(pady=5)  
    completion_label = tk.Label(scrollable_frame, text="Completed Processes: ", font=("Arial", 12, "bold"), bg="#3f3f3f", fg="#00FF00")  # Reduced from 16 to 12
    completion_label.pack(pady=5)  
    timeline_label = tk.Label(scrollable_frame, text="Current Execution: None", font=("Arial", 12), bg="#3f3f3f", fg="#FFFFFF")  
    timeline_label.pack(pady=5)  

    slider_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    slider_frame.pack(pady=5, fill="x") 
    time_slider = ttk.Scale(slider_frame, from_=0, to=max_time, orient="horizontal", length=400, command=lambda val: manual_time_update(float(val)))
    time_slider.pack(expand=True)

    is_running = tk.BooleanVar(value=False)
    is_completed = tk.BooleanVar(value=False)
    current_time = 0.0
    current_gantt_idx = 0
    completed_processes = []

    start_pause_label = tk.Label(
        scrollable_frame,
        text="  Start  ",
        font=("Arial", 16, "bold"),  
        bg="#040406",
        fg="white",
        cursor="hand2"
    )
    start_pause_label.pack(pady=10)  

    def toggle_start_pause(event=None):
        nonlocal current_time, current_gantt_idx
        if not is_running.get() and not is_completed.get():
            is_running.set(True)
            start_pause_label.config(text="  Pause  ")
            update_process_bars()
        elif is_running.get() and not is_completed.get():
            is_running.set(False)
            start_pause_label.config(text="  Start  ")
        elif is_completed.get():
            is_running.set(True)
            is_completed.set(False)
            current_time = 0.0
            current_gantt_idx = 0
            for i in range(n):
                process_progress[i] = 0
                bar, x1, y1, y2, _ = process_bars[i]
                animation_canvas.coords(bar, x1, y1, x1, y2)
                animation_canvas.itemconfig(bar, fill="gray")
                animation_canvas.itemconfig(process_status_texts[i], text="", fill="#CC5500", font=("Arial", 16))
            time_label.config(text="Time: 0")
            timeline_label.config(text="Current Execution: None")
            completion_label.config(text="Completed Processes: ")
            time_slider.set(0)
            completed_processes.clear()
            start_pause_label.config(text="  Pause  ")
            update_process_bars()

    start_pause_label.bind("<Button-1>", toggle_start_pause)

    def manual_time_update(val):
        if is_running.get():
            return
        nonlocal current_time
        current_time = float(val)
        time_label.config(text=f"Time: {int(current_time)}")
        for i in range(n):
            bar, x1, y1, y2, width = process_bars[i]
            animation_canvas.coords(bar, x1, y1, x1, y2)
            animation_canvas.itemconfig(bar, fill="gray")
            animation_canvas.itemconfig(process_status_texts[i], text="", fill="#CC5500", font=("Arial", 16))
            process_progress[i] = 0
        completed_processes.clear()
        completion_label.config(text="Completed Processes: ")

        for process_name, start, end in gantt_chart:
            if start > current_time:
                break
            process_idx = process_map.get(process_name, -1)
            if process_idx == -1:
                continue
            bar, x1, y1, y2, width = process_bars[process_idx]
            total_burst = total_burst_times[process_idx]
            num_segments, segment_width = process_segments[process_idx]
            executed_time = min(end, current_time) - start
            if executed_time <= 0:
                continue
            process_progress[process_idx] = sum(e - s for p, s, e in gantt_chart 
                                              if p == process_name and s <= current_time and e <= end)
            total_progress = min(process_progress[process_idx] / total_burst, 1.0)
            current_width = x1 + int(total_progress * width)
            animation_canvas.coords(bar, x1, y1, current_width, y2)
            animation_canvas.itemconfig(bar, fill=process_colors[process_name])
            if total_progress >= 1.0 and process_name not in completed_processes:
                animation_canvas.itemconfig(process_status_texts[process_idx], text="Completed", fill="#00FF00", font=("Arial", 16))
                completed_processes.append(process_name)
            elif end > current_time:
                animation_canvas.itemconfig(process_status_texts[process_idx], text="Running", fill="#CC5500", font=("Arial", 16))
                timeline_label.config(text=f"Current Execution: {process_name} (Start: {start}, End: {end})")
            else:
                animation_canvas.itemconfig(process_status_texts[process_idx], text="Preempted", fill="#FFFF00", font=("Arial", 16))
        completion_label.config(text=f"Completed Processes: {', '.join(completed_processes)}")
        if not any(start <= current_time < end for _, start, end in gantt_chart):
            timeline_label.config(text="Current Execution: Idle")

    def update_process_bars():
        nonlocal current_time, current_gantt_idx

        if not is_running.get():
            if animation_canvas.winfo_exists():
                animation_canvas.after(20, update_process_bars)
            return

        all_completed = all(process_progress[i] >= total_burst_times[i] for i in range(n))
        if all_completed and current_gantt_idx >= len(gantt_chart):
            is_running.set(False)
            is_completed.set(True)
            start_pause_label.config(text="  Replay  ")
            return

        current_time += 0.02
        time_label.config(text=f"Time: {current_time:.2f}")
        time_slider.set(current_time)

        if current_gantt_idx < len(gantt_chart):
            process_name, start, end = gantt_chart[current_gantt_idx]
            process_idx = process_map.get(process_name, -1)

            if process_idx == -1:
                current_gantt_idx += 1
                if animation_canvas.winfo_exists():
                    animation_canvas.after(20, update_process_bars)
                return

            bar, x1, y1, y2, width = process_bars[process_idx]
            total_burst = total_burst_times[process_idx]
            num_segments, segment_width = process_segments[process_idx]

            previous_process_name = timeline_label.cget("text").split(":")[-1].strip().split(" ")[0]
            if previous_process_name != "Idle" and previous_process_name != process_name:
                prev_process_idx = process_map.get(previous_process_name, -1)
                if prev_process_idx != -1 and previous_process_name not in completed_processes:
                    current_prev_progress = process_progress[prev_process_idx]
                    if current_prev_progress < total_burst_times[prev_process_idx]:
                        animation_canvas.itemconfig(process_status_texts[prev_process_idx], text="Preempted", fill="#FFFF00")

            if current_time < start:
                timeline_label.config(text="Current Execution: Idle")
            elif start <= current_time <= end:
                if process_progress[process_idx] < total_burst:
                    process_progress[process_idx] += 0.02
                    process_progress[process_idx] = min(process_progress[process_idx], total_burst)

                filled_width = x1 + (process_progress[process_idx] / total_burst) * width
                animation_canvas.coords(bar, x1, y1, filled_width, y2)
                animation_canvas.itemconfig(bar, fill=process_colors[process_name])
                animation_canvas.itemconfig(process_status_texts[process_idx], text="Running", fill="#CC5500")
                timeline_label.config(text=f"Current Execution: {process_name} (Start: {start}, End: {end})")

                if process_progress[process_idx] >= total_burst and process_name not in completed_processes:
                    animation_canvas.itemconfig(process_status_texts[process_idx], text="Completed", fill="#00FF00")
                    completed_processes.append(process_name)
                    completion_label.config(text=f"Completed Processes: {', '.join(completed_processes)}")

            elif current_time > end:
                executed_time = sum(e - s for p, s, e in gantt_chart[:current_gantt_idx + 1] 
                                   if p == process_name and e <= current_time)
                process_progress[process_idx] = min(executed_time, total_burst)
                total_progress = process_progress[process_idx] / total_burst
                current_width = x1 + total_progress * width
                animation_canvas.coords(bar, x1, y1, current_width, y2)
                animation_canvas.itemconfig(bar, fill=process_colors[process_name])

                if process_progress[process_idx] >= total_burst and process_name not in completed_processes:
                    animation_canvas.itemconfig(process_status_texts[process_idx], text="Completed", fill="#00FF00")
                    completed_processes.append(process_name)
                    completion_label.config(text=f"Completed Processes: {', '.join(completed_processes)}")
                elif process_progress[process_idx] < total_burst and process_name not in completed_processes:
                    animation_canvas.itemconfig(process_status_texts[process_idx], text="Preempted", fill="#FFFF00")

                current_gantt_idx += 1

        if animation_canvas.winfo_exists():
            animation_canvas.after(20, update_process_bars)

    graph_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
    graph_frame.pack(pady=10, fill="x")  

    gantt_frame = tk.Frame(graph_frame, bg="#3f3f3f")
    gantt_frame.pack(pady=5, fill="x")
    hist_frame = tk.Frame(graph_frame, bg="#3f3f3f")
    hist_frame.pack(pady=5, fill="x")  

    gantt_title = tk.Label(gantt_frame, text=f"Gantt Chart of {algorithm_name}", font=("Arial", 18, "bold"), bg="#3f3f3f", fg="#BB86FC")  
    gantt_title.pack(pady=5, expand=True)
    plot_gantt_chart(gantt_frame, gantt_chart)

    hist_title = tk.Label(hist_frame, text="Histogram of Waiting Times", font=("Arial", 18, "bold"), bg="#3f3f3f", fg="#BB86FC")  
    hist_title.pack(pady=5, expand=True)
    plot_histogram(hist_frame, process_array, waiting_array)

    if len(comparison_metrics) > 1:
        heatmap_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        heatmap_frame.pack(pady=10, fill="x")  # Reduced from 20 to 10
        tk.Label(heatmap_frame, 
                 text="Comparison Heatmap Across Algorithms", 
                 font=("Arial", 18, "bold"), bg="#3f3f3f", fg="#FFFFFF").pack(expand=True, pady=5)  # Reduced from 26 to 18
        heatmap_data = pd.DataFrame(comparison_metrics).T
        heatmap_data_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 4))
        sns.heatmap(heatmap_data_normalized, annot=heatmap_data.round(2), cmap="YlOrRd", ax=ax_heatmap, cbar_kws={'label': 'Normalized Value'})
        ax_heatmap.set_title("Metrics Comparison Across Algorithms", fontsize=12)  # Reduced from 14 to 12
        plt.tight_layout()
        canvas_heatmap = FigureCanvasTkAgg(fig_heatmap, master=heatmap_frame)
        canvas_heatmap.draw()
        canvas_heatmap.get_tk_widget().pack(expand=True, fill="both")

    go_back_label = tk.Label(
        scrollable_frame,
        text="  Go Back  ",
        font=("Arial", 16, "bold"),  
        bg="#040406",
        fg="white",
        cursor="hand2"
    )
    go_back_label.pack(pady=10)  
    go_back_label.bind("<Button-1>", lambda e: root.destroy())

    def on_mouse_wheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Button-4>", on_mouse_wheel)
    canvas.bind_all("<Button-5>", on_mouse_wheel)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

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
def calculate_energy(gantt_chart, total_time, active_power=5.0, idle_power=1.0, context_switch_energy=0.1):
    """
    Calculate energy consumption for a scheduling algorithm.
    
    Parameters:
    - gantt_chart: List of tuples (process, start_time, end_time) from the Gantt chart.
    - total_time: Total simulation time.
    - active_power: Power consumption in active state (W).
    - idle_power: Power consumption in idle state (W).
    - context_switch_energy: Energy per context switch (J).
    
    Returns:
    - dict: Energy breakdown (total, active, idle, context switch).
    """
   
    active_time = 0
    context_switches = 0
    last_process = None
    last_end = 0

    for process, start, end in gantt_chart:
        
        active_time += end - start
        
        
        if last_process is not None and start == last_end and process != last_process:
            context_switches += 1
        last_process = process
        last_end = end

   
    idle_time = total_time - active_time

    
    active_energy = active_time * active_power  # J
    idle_energy = idle_time * idle_power  # J
    context_switch_energy_total = context_switches * context_switch_energy  # J
    total_energy = active_energy + idle_energy + context_switch_energy_total

    return {
        "total_energy": total_energy,
        "active_energy": active_energy,
        "idle_energy": idle_energy,
        "context_switch_energy": context_switch_energy_total,
        "context_switches": context_switches
    }

def show_energy_consumption_view(algo, gantt_chart, total_time):
    try:
        # Convert gantt_chart to a list if it's a NumPy array
        if isinstance(gantt_chart, np.ndarray):
            gantt_chart = gantt_chart.tolist()
        # Ensure gantt_chart is a list of tuples
        gantt_chart = [(str(process), float(start), float(end)) for process, start, end in gantt_chart]

        # Create a new window
        energy_window = tk.Toplevel()
        energy_window.title(f"Energy Consumption Analysis - {algo}")
        energy_window.configure(bg="#3f3f3f")
        energy_window.state('zoomed')

        main_frame = tk.Frame(energy_window, bg="#3f3f3f")
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, bg="#3f3f3f", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#3f3f3f")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create the window in the canvas, initially at the top-left
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Function to center the scrollable_frame horizontally in the canvas
        def center_content(event):
            # Get the visible width of the canvas
            canvas_width = event.width
            # Get the width of the scrollable_frame
            frame_width = scrollable_frame.winfo_reqwidth()
            # Calculate the x-coordinate to center the frame
            x_position = max(0, (canvas_width - frame_width) // 2)
            # Update the position of the window in the canvas
            canvas.coords(canvas_window, x_position, 0)


        canvas.bind("<Configure>", center_content)

        # Title (Centered)
        title_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        title_frame.pack(fill="x", pady=20)
        tk.Label(title_frame, 
                 text=f"Energy Consumption Analysis for {algo}", 
                 font=("Arial", 24, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True)

        # Validate inputs
        if not gantt_chart or not isinstance(gantt_chart, (list, tuple)) or total_time <= 0:
            error_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
            error_frame.pack(fill="x", pady=10)
            tk.Label(error_frame, 
                     text="Error: Invalid Gantt chart or total time.", 
                     font=("Arial", 16), 
                     bg="#3f3f3f", 
                     fg="red").pack(expand=True)
            return

        # Calculate energy consumption
        energy_result = calculate_energy(gantt_chart, total_time)
        total_energy = round(energy_result["total_energy"], 2)
        active_energy = round(energy_result["active_energy"], 2)
        idle_energy = round(energy_result["idle_energy"], 2)
        context_switch_energy = round(energy_result["context_switch_energy"], 2)
        context_switches = energy_result["context_switches"]

        # Calculate active and idle times
        active_time = sum(end - start for _, start, end in gantt_chart)
        idle_time = total_time - active_time
        num_processes = len(set(process for process, _, _ in gantt_chart))

        # Energy Efficiency Score
        efficiency_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        efficiency_frame.pack(fill="x", pady=20)
        tk.Label(efficiency_frame, 
                 text="Energy Efficiency Score", 
                 font=("Arial", 18, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)
        
        # Calculate efficiency score (simplified: lower energy per process per unit time is better)
        efficiency_score = round(100 - (total_energy / (num_processes * total_time)) * 10, 2)
        efficiency_score = max(0, min(100, efficiency_score))  # Clamp between 0 and 100
        efficiency_text = tk.Text(efficiency_frame, height=3, width=80, font=("Arial", 14), bg="#2d2d2d", fg="white")
        efficiency_text.pack(expand=True, pady=5)
        efficiency_text.insert(tk.END, f"Efficiency Score: {efficiency_score}/100\n")
        efficiency_text.insert(tk.END, "   - This score reflects energy efficiency (higher is better).\n")
        efficiency_text.insert(tk.END, "   - Based on total energy per process per unit time.")
        efficiency_text.configure(state="disabled")

        # Factors Contributing to Energy Consumption (Text Explanation)
        factors_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        factors_frame.pack(fill="x", pady=20)
        tk.Label(factors_frame, 
                 text="Factors Contributing to Energy Consumption", 
                 font=("Arial", 18, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        factors_text = tk.Text(factors_frame, height=6, width=80, font=("Arial", 14), bg="#2d2d2d", fg="white")
        factors_text.pack(expand=True, pady=5)
        factors_text.insert(tk.END, f"1. Active State Energy ({active_energy} J):\n")
        factors_text.insert(tk.END, "   - This is the energy consumed while the CPU is actively executing processes.\n")
        factors_text.insert(tk.END, f"   - Contributes {active_energy/total_energy*100:.2f}% to the total energy.\n")
        factors_text.insert(tk.END, f"2. Idle State Energy ({idle_energy} J):\n")
        factors_text.insert(tk.END, "   - This is the energy consumed when the CPU is idle, waiting for processes.\n")
        factors_text.insert(tk.END, f"   - Contributes {idle_energy/total_energy*100:.2f}% to the total energy.\n")
        factors_text.insert(tk.END, f"3. Context Switch Energy ({context_switch_energy} J):\n")
        factors_text.insert(tk.END, f"   - This is the energy overhead from {context_switches} context switches.\n")
        factors_text.insert(tk.END, f"   - Contributes {context_switch_energy/total_energy*100:.2f}% to the total energy.")
        factors_text.configure(state="disabled")

        # Energy Breakdown (DataFrame)
        energy_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        energy_frame.pack(fill="x", pady=20)
        tk.Label(energy_frame, 
                 text="Energy Consumption Breakdown", 
                 font=("Arial", 18, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        energy_data = {
            "Component": ["Active State", "Idle State", "Context Switches", "Total"],
            "Energy (J)": [active_energy, idle_energy, context_switch_energy, total_energy],
            "Percentage (%)": [
                round(active_energy/total_energy*100, 2),
                round(idle_energy/total_energy*100, 2),
                round(context_switch_energy/total_energy*100, 2),
                100.0
            ]
        }
        df_energy = pd.DataFrame(energy_data)

        table = ttk.Treeview(energy_frame, columns=list(df_energy.columns), show="headings", height=len(df_energy))
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 16, "bold"))
        style.configure("Treeview", font=("Arial", 14), rowheight=30)
        style.configure("Vertical.TScrollbar", width=20)
        for col in df_energy.columns:
            table.heading(col, text=col, anchor="center")
            table.column(col, width=200, anchor="center")
        for _, row in df_energy.iterrows():
            table.insert("", "end", values=list(row))
        table.pack(expand=True)

        # Combined Frame for Energy Consumption Over Time and Context Switch Timeline (Side by Side)
        combined_graph_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        combined_graph_frame.pack(fill="x", pady=20)

        # Energy Consumption Over Time (Left)
        energy_over_time_frame = tk.Frame(combined_graph_frame, bg="#3f3f3f")
        energy_over_time_frame.pack(side="left", expand=True, fill="both", padx=10)
        tk.Label(energy_over_time_frame, 
                 text="Energy Consumption Over Time", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        # Calculate power over time
        time_points = []
        power_values = []
        current_time = 0
        active_power = 5.0  # Same as in calculate_energy
        idle_power = 1.0    # Same as in calculate_energy
        for process, start, end in gantt_chart:
            if current_time < start:
                time_points.extend([current_time, start])
                power_values.extend([idle_power, idle_power])
            time_points.extend([start, end])
            power_values.extend([active_power, active_power])
            current_time = end
        if current_time < total_time:
            time_points.extend([current_time, total_time])
            power_values.extend([idle_power, idle_power])

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.step(time_points, power_values, where='post', color="#00CC00", label="Power Consumption")
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.set_title("Power Consumption Over Time", fontsize=12)
        ax.legend()
        plt.tight_layout()
        canvas_fig = FigureCanvasTkAgg(fig, master=energy_over_time_frame)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(expand=True, fill="both")

        # Context Switch Timeline (Right)
        context_switch_frame = tk.Frame(combined_graph_frame, bg="#3f3f3f")
        context_switch_frame.pack(side="left", expand=True, fill="both", padx=10)
        tk.Label(context_switch_frame, 
                 text="Context Switch Timeline", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        # Identify context switch points
        context_switch_times = []
        last_process = None
        last_end = 0
        for process, start, end in gantt_chart:
            if last_process is not None and start == last_end and process != last_process:
                context_switch_times.append(start)
            last_process = process
            last_end = end

        # Plot context switch timeline
        fig_cs, ax_cs = plt.subplots(figsize=(6, 3))
        if context_switch_times:
            ax_cs.bar(context_switch_times, [1] * len(context_switch_times), width=0.1, color="#FF5555", label="Context Switch")
        ax_cs.set_xlabel("Time", fontsize=10)
        ax_cs.set_ylabel("Switch Event", fontsize=10)
        ax_cs.set_title("Context Switch Timeline", fontsize=12)
        ax_cs.set_ylim(0, 1.5)
        ax_cs.legend()
        plt.tight_layout()
        canvas_cs = FigureCanvasTkAgg(fig_cs, master=context_switch_frame)
        canvas_cs.draw()
        canvas_cs.get_tk_widget().pack(expand=True, fill="both")

        # Energy Consumption by Process (Table)
        process_energy_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        process_energy_frame.pack(fill="x", pady=20)
        tk.Label(process_energy_frame, 
                 text="Energy Consumption by Process", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        # Calculate energy per process
        process_energy = {}
        for process, start, end in gantt_chart:
            duration = end - start
            energy = duration * 5.0  # Active power = 5 W
            process_energy[process] = process_energy.get(process, 0) + energy

        process_data = {
            "Process": list(process_energy.keys()),
            "Energy (J)": [round(e, 2) for e in process_energy.values()],
            "Percentage (%)": [round(e/total_energy*100, 2) for e in process_energy.values()]
        }
        df_process = pd.DataFrame(process_data)

        process_table = ttk.Treeview(process_energy_frame, columns=list(df_process.columns), show="headings", height=len(df_process))
        for col in df_process.columns:
            process_table.heading(col, text=col, anchor="center")
            process_table.column(col, width=200, anchor="center")
        for _, row in df_process.iterrows():
            process_table.insert("", "end", values=list(row))
        process_table.pack(expand=True)

        # Combined Frame for Pie Chart and Donut Chart (Side by Side)
        combined_chart_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        combined_chart_frame.pack(fill="x", pady=20)

        # Pie Chart for Energy Breakdown (Left)
        pie_frame = tk.Frame(combined_chart_frame, bg="#3f3f3f")
        pie_frame.pack(side="left", expand=True, fill="both", padx=10)
        tk.Label(pie_frame, 
                 text="Energy Breakdown Distribution", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        labels = ["Active State", "Idle State", "Context Switches"]
        sizes = [active_energy, idle_energy, context_switch_energy]
        colors = ["#FF9999", "#66B2FF", "#99FF99"]
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=45)  # Changed startangle to 45
        ax_pie.axis('equal')
        plt.tight_layout()
        canvas_pie = FigureCanvasTkAgg(fig_pie, master=pie_frame)
        canvas_pie.draw()
        canvas_pie.get_tk_widget().pack(expand=True, fill="both")

        # Idle vs. Active Time Breakdown (Donut Chart, Right)
        time_breakdown_frame = tk.Frame(combined_chart_frame, bg="#3f3f3f")
        time_breakdown_frame.pack(side="left", expand=True, fill="both", padx=10)
        tk.Label(time_breakdown_frame, 
                 text="Active vs. Idle Time Breakdown", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        fig_time, ax_time = plt.subplots(figsize=(5, 5))
        labels = ["Active Time", "Idle Time"]
        sizes = [active_time, idle_time]
        colors = ["#FFCC99", "#99CCFF"]
        ax_time.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
        ax_time.axis('equal')
        plt.tight_layout()
        canvas_time = FigureCanvasTkAgg(fig_time, master=time_breakdown_frame)
        canvas_time.draw()
        canvas_time.get_tk_widget().pack(expand=True, fill="both")

        # Recommendations for Energy Optimization
        recommendations_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        recommendations_frame.pack(fill="x", pady=20)
        tk.Label(recommendations_frame, 
                 text="Recommendations for Energy Optimization", 
                 font=("Arial", 16, "bold"), 
                 bg="#3f3f3f", 
                 fg="white").pack(expand=True, pady=5)

        recommendations_text = tk.Text(recommendations_frame, height=5, width=80, font=("Arial", 14), bg="#2d2d2d", fg="white")
        recommendations_text.pack(expand=True, pady=5)
        if idle_time/total_time > 0.3:
            recommendations_text.insert(tk.END, "- High idle time detected. Consider scheduling processes more tightly to reduce idle periods.\n")
        if context_switches > num_processes:
            recommendations_text.insert(tk.END, "- High number of context switches. Consider increasing time quantum (if applicable) to reduce switches.\n")
        if active_energy/total_energy > 0.8:
            recommendations_text.insert(tk.END, "- Active state dominates energy use. Optimize process execution times if possible.\n")
        if efficiency_score < 50:
            recommendations_text.insert(tk.END, "- Low efficiency score. Review scheduling algorithm for better energy management.\n")
        recommendations_text.insert(tk.END, "- General tip: Use power-efficient hardware or dynamic voltage scaling to reduce energy consumption.")
        recommendations_text.configure(state="disabled")

        # Go Back Button (Smaller Size)
        button_frame = tk.Frame(scrollable_frame, bg="#3f3f3f")
        button_frame.pack(fill="x", pady=20)
        ttk.Button(button_frame, 
                   text="Go Back", 
                   command=energy_window.destroy,
                   style="Custom.TButton").pack(expand=True)
        # Customize button style to reduce size
        style.configure("Custom.TButton", font=("Arial", 12), padding=5)

        # Enable mouse wheel scrolling
        def on_mouse_wheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        canvas.bind_all("<Button-4>", on_mouse_wheel)
        canvas.bind_all("<Button-5>", on_mouse_wheel)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Ensure the window is brought to the front
        energy_window.lift()
        energy_window.focus_force()

    except Exception as e:
        # Display error message if something goes wrong
        error_window = tk.Toplevel()
        error_window.title("Error")
        error_window.configure(bg="#3f3f3f")
        tk.Label(error_window, 
                 text=f"Error displaying energy consumption view: {str(e)}", 
                 font=("Arial", 16), 
                 bg="#3f3f3f", 
                 fg="red").pack(pady=20, padx=20)
        ttk.Button(error_window, 
                   text="Close", 
                   command=error_window.destroy).pack(pady=10)

def visualize_comparison(n, results):
    root = tk.Tk()
    root.title("Scheduling Algorithms Comparison")
    root.configure(bg="#0C0C0C")
    root.state('zoomed')
    root.minsize(800, 600)

    main_frame = tk.Frame(root, bg="#0C0C0C")
    main_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(main_frame, bg="#0C0C0C", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#0C0C0C")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def center_content(event):
        canvas_width = event.width
        frame_width = scrollable_frame.winfo_reqwidth()
        x_position = max(0, (canvas_width - frame_width) // 2)
        canvas.coords(canvas_window, x_position, 0)

    canvas.bind("<Configure>", center_content)

    tk.Label(scrollable_frame, 
             text="Comparison of Scheduling Algorithms", 
             font=("Arial", 20, "bold"),
             bg="#0C0C0C", 
             fg="white").pack(pady=10, anchor="center")

    # DataFrame Section for Performance Metrics
    metrics_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
    metrics_frame.pack(pady=10, padx=20, anchor="center")

    tk.Label(metrics_frame, 
             text="Performance Metrics Comparison", 
             font=("Arial", 16, "bold"),
             bg="#0C0C0C", 
             fg="white").pack(pady=5, anchor="center")

    total_time = 1
    try:
        completion_times = []
        for algo, (_, _, _, _, completion_array, _, _, _) in results.items():
            if isinstance(completion_array, np.ndarray):
                completion_array = completion_array.tolist()
            if (isinstance(completion_array, (list, tuple)) and 
                len(completion_array) > 0 and 
                all(isinstance(x, (int, float)) for x in completion_array)):
                completion_times.append(max(completion_array))
            else:
                print(f"Warning: Invalid completion_array for {algo}: {completion_array}")
        if completion_times:
            total_time = max(completion_times)
        else:
            print("Warning: No valid completion times found. Using default total_time=1.")
    except Exception as e:
        print(f"Error calculating total_time: {str(e)}. Using default total_time=1.")

    data = {"Algorithm": [], "Throughput": [], "Avg Waiting Time": [], "Avg Turnaround Time": [], "Energy Consumption (J)": []}
    for algo, (process_array, arrival_array, burst_array, priority_array, completion_array, turnaround_array, waiting_array, gantt_chart) in results.items():
        if isinstance(process_array, np.ndarray):
            process_array = process_array.tolist()
        if isinstance(arrival_array, np.ndarray):
            arrival_array = arrival_array.tolist()
        if isinstance(burst_array, np.ndarray):
            burst_array = burst_array.tolist()
        if isinstance(priority_array, np.ndarray):
            priority_array = priority_array.tolist()
        if isinstance(completion_array, np.ndarray):
            completion_array = completion_array.tolist()
        if isinstance(turnaround_array, np.ndarray):
            turnaround_array = turnaround_array.tolist()
        if isinstance(waiting_array, np.ndarray):
            waiting_array = waiting_array.tolist()
        if isinstance(gantt_chart, np.ndarray):
            gantt_chart = gantt_chart.tolist()

        gantt_chart = [(str(process), float(start), float(end)) for process, start, end in gantt_chart]

        data["Algorithm"].append(algo)
        data["Throughput"].append(calculate_throughput(n, completion_array))
        data["Avg Waiting Time"].append(calculate_avg_waiting_time(waiting_array))
        data["Avg Turnaround Time"].append(calculate_avg_waiting_time(turnaround_array))
        data["Energy Consumption (J)"].append(round(calculate_energy(gantt_chart, total_time)["total_energy"], 2))

    df = pd.DataFrame(data)
    
    table = ttk.Treeview(metrics_frame, columns=list(df.columns), show="headings", height=len(results))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 16, "bold"))
    style.configure("Treeview", font=("Arial", 16), rowheight=25)
    style.configure("Vertical.TScrollbar", width=20)
    for col in df.columns:
        table.heading(col, text=col, anchor="center")
        table.column(col, width=150, anchor="center")
    for _, row in df.iterrows():
        table.insert("", "end", values=list(row))
    table.pack(fill="x", anchor="center")

    try:
        best_waiting_idx = df["Avg Waiting Time"].idxmin()
        worst_waiting_idx = df["Avg Waiting Time"].idxmax()
        best_waiting_algo = df.loc[best_waiting_idx, "Algorithm"]
        worst_waiting_algo = df.loc[worst_waiting_idx, "Algorithm"]

        best_energy_idx = df["Energy Consumption (J)"].idxmin()
        worst_energy_idx = df["Energy Consumption (J)"].idxmax()
        best_energy_algo = df.loc[best_energy_idx, "Algorithm"]
        worst_energy_algo = df.loc[worst_energy_idx, "Algorithm"]

        performance_label = tk.Label(metrics_frame,
                                    text=f"Least Avg Waiting Time: {best_waiting_algo} (Lowest Avg Waiting Time)   |   Worst Avg Waiting Time: {worst_waiting_algo} (Highest Avg Waiting Time)\n"
                                         f"Least Avg Energy: {best_energy_algo} (Lowest Energy Consumption)   |   Worst Energy: {worst_energy_algo} (Highest Energy Consumption)",
                                    font=("Arial", 16, "italic"),
                                    bg="#0C0C0C", 
                                    fg="#00FF00")  # Changed fg to green for best/worst avg time
        performance_label.pack(pady=5, anchor="center")
    except Exception as e:
        tk.Label(metrics_frame,
                 text="Error identifying best/worst algorithms: Data may be invalid.",
                 font=("Arial", 16, "italic"),
                 bg="#0C0C0C", 
                 fg="white").pack(pady=5, anchor="center")

    # Calculate max width (47% of screen width)
    screen_width = root.winfo_screenwidth()
    max_graph_width = screen_width * 0.40 / 100

    # Gantt Charts, Histograms, and Process Details Section
    for algo, (process_array, arrival_array, burst_array, priority_array, completion_array, 
              turnaround_array, waiting_array, gantt_chart) in results.items():
        if isinstance(process_array, np.ndarray):
            process_array = process_array.tolist()
        if isinstance(arrival_array, np.ndarray):
            arrival_array = arrival_array.tolist()
        if isinstance(burst_array, np.ndarray):
            burst_array = burst_array.tolist()
        if isinstance(priority_array, np.ndarray):
            priority_array = priority_array.tolist()
        if isinstance(completion_array, np.ndarray):
            completion_array = completion_array.tolist()
        if isinstance(turnaround_array, np.ndarray):
            turnaround_array = turnaround_array.tolist()
        if isinstance(waiting_array, np.ndarray):
            waiting_array = waiting_array.tolist()
        if isinstance(gantt_chart, np.ndarray):
            gantt_chart = gantt_chart.tolist()

        gantt_chart = [(str(process), float(start), float(end)) for process, start, end in gantt_chart]

        algo_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
        algo_frame.pack(pady=10, fill="x", padx=20, anchor="center")

        tk.Label(algo_frame, 
                 text=algo, 
                 font=("Arial", 16, "bold"),
                 bg="#0C0C0C", 
                 fg="white").pack(pady=5, anchor="center")

        chart_frame = tk.Frame(algo_frame, bg="#0C0C0C")
        chart_frame.pack(fill="x", anchor="center")

        gantt_subframe = tk.Frame(chart_frame, bg="#0C0C0C")
        gantt_subframe.pack(side="left", padx=5, fill="x", expand=True)
        tk.Label(gantt_subframe, 
                 text="Gantt Chart", 
                 font=("Arial", 16, "bold"),
                 bg="#0C0C0C", 
                 fg="white").pack(pady=5, anchor="center")
        plot_gantt_chart(gantt_subframe, gantt_chart, max_width=max_graph_width)

        hist_subframe = tk.Frame(chart_frame, bg="#0C0C0C")
        hist_subframe.pack(side="left", padx=5, fill="x", expand=True)
        tk.Label(hist_subframe, 
                 text="Waiting Time Histogram", 
                 font=("Arial", 16, "bold"),
                 bg="#0C0C0C", 
                 fg="white").pack(pady=5, anchor="center")
        plot_histogram(hist_subframe, process_array, waiting_array, max_width=max_graph_width)

        # Process Details DataFrame
        process_frame = tk.Frame(algo_frame, bg="#0C0C0C")
        process_frame.pack(pady=5, fill="x", anchor="center")
        tk.Label(process_frame, 
                 text="Process Details", 
                 font=("Arial", 16, "bold"),
                 bg="#0C0C0C", 
                 fg="white").pack(pady=5, anchor="center")

        process_data = {
            "Process": process_array,
            "Arrival Time": arrival_array,
            "Burst Time": burst_array,
            "Priority": priority_array,
            "Completion Time": completion_array,
            "Turn Around Time": turnaround_array,
            "Waiting Time": waiting_array
        }
        process_df = pd.DataFrame(process_data)
        
        process_table = ttk.Treeview(process_frame, columns=list(process_df.columns), show="headings", height=len(process_array))
        style.configure("Treeview", font=("Arial", 16), rowheight=25)
        for col in process_df.columns:
            process_table.heading(col, text=col, anchor="center")
            process_table.column(col, width=120, anchor="center")
        for _, row in process_df.iterrows():
            process_table.insert("", "end", values=list(row))
        process_table.pack(fill="x", anchor="center")

        # Buttons Frame
        button_frame = tk.Frame(algo_frame, bg="#0C0C0C")
        button_frame.pack(pady=5, anchor="center")

        ttk.Button(button_frame, 
                   text="View Energy Consumption", 
                   command=lambda a=algo, g=gantt_chart, t=total_time: show_energy_consumption_view(a, g, t),
                   style="Custom.TButton").pack(side="left", padx=5)

        style.configure("Visualize.TButton", 
                        font=("Arial", 16, "bold"),
                        background="#0C0C0C",
                        foreground="white",
                        borderwidth=1, 
                        relief="flat")
        style.map("Visualize.TButton",
                  background=[("active", "#777777")],
                  foreground=[("active", "#ffffff")])
        ttk.Button(button_frame, 
                   text="Visualize Scheduling", 
                   command=lambda a=algo, p=process_array, arr=arrival_array, b=burst_array, pp=priority_array,
                                 c=completion_array, t=turnaround_array, w=waiting_array, 
                                 g=gantt_chart: visualize_scheduling(n, p, arr, b,pp, c, t, w, g, a),
                   style="Visualize.TButton").pack(side="left", padx=5)

    # Summary Statistics Section
    summary_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
    summary_frame.pack(pady=10, padx=20, anchor="center")

    tk.Label(summary_frame,
             text="Summary Statistics",
             font=("Arial", 16, "bold"),
             bg="#0C0C0C", 
             fg="white").pack(pady=5, anchor="center")

    summary_text = tk.Text(summary_frame, height=5, width=50, font=("Arial", 16), bg="#0C0C0C", fg="white")
    summary_text.pack(pady=5, anchor="center")
    summary_text.insert(tk.END, f"Total Algorithms Compared: {len(results)}\n")
    summary_text.insert(tk.END, f"Average Throughput: {df['Throughput'].mean():.2f}\n")
    summary_text.insert(tk.END, f"Average Waiting Time Across All: {df['Avg Waiting Time'].mean():.2f}\n")
    summary_text.insert(tk.END, f"Average Turnaround Time Across All: {df['Avg Turnaround Time'].mean():.2f}\n")
    summary_text.insert(tk.END, f"Average Energy Consumption Across All: {df['Energy Consumption (J)'].mean():.2f} J\n")
    summary_text.configure(state="disabled")

    # Comparison Bar Chart for Waiting Time
    comparison_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
    comparison_frame.pack(pady=10, padx=20, anchor="center")
    tk.Label(comparison_frame, 
             text="Average Waiting Time Comparison", 
             font=("Arial", 16, "bold"),
             bg="#0C0C0C", 
             fg="white").pack(pady=5, anchor="center")
    fig, ax = plt.subplots(figsize=(min(max_graph_width, 10), 3))  # Increased height to 3
    ax.bar(df["Algorithm"], df["Avg Waiting Time"], color="#CC5500")
    ax.set_xlabel("Algorithm", fontsize=16)
    ax.set_ylabel("Avg Waiting Time", fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=comparison_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill="x", anchor="center")

    # Comparison Bar Chart for Energy Consumption (Normalized)
    energy_comparison_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
    energy_comparison_frame.pack(pady=10, padx=20, anchor="center")
    tk.Label(energy_comparison_frame, 
             text="Energy Consumption Comparison (Normalized)", 
             font=("Arial", 16, "bold"),
             bg="#0C0C0C", 
             fg="white").pack(pady=5, anchor="center")
    
    min_energy = df["Energy Consumption (J)"].min()
    normalized_energy = df["Energy Consumption (J)"] - min_energy
    
    fig_energy, ax_energy = plt.subplots(figsize=(min(max_graph_width, 10), 3))  # Increased height to 3
    ax_energy.bar(df["Algorithm"], normalized_energy, color="#00CC00")
    ax_energy.set_xlabel("Algorithm", fontsize=16)
    ax_energy.set_ylabel("Energy Consumption Above Minimum (J)", fontsize=16)
    ax_energy.tick_params(axis='x', rotation=45)
    
    ax_energy.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 2 decimal places
    
    plt.tight_layout()
    canvas_fig_energy = FigureCanvasTkAgg(fig_energy, master=energy_comparison_frame)
    canvas_fig_energy.draw()
    canvas_fig_energy.get_tk_widget().pack(fill="x", anchor="center")

    # Buttons Frame
    buttons_frame = tk.Frame(scrollable_frame, bg="#0C0C0C")
    buttons_frame.pack(pady=10, anchor="center")

    ttk.Button(buttons_frame, 
               text="Export Detailed Results", 
               command=lambda: export_detailed_results(results),
               style="Custom.TButton").pack(side="left", padx=5)
    ttk.Button(buttons_frame, 
               text="Go Back", 
               command=lambda: root.destroy(),
               style="Custom.TButton").pack(side="left", padx=5)

    def on_mouse_wheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Button-4>", on_mouse_wheel)
    canvas.bind_all("<Button-5>", on_mouse_wheel)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    root.mainloop()

# Modified plot_gantt_chart with max_width parameter
def plot_gantt_chart(frame, gantt_chart, max_width=None):
    if max_width is None:
        max_width = 8  # Default width in inches
    fig, ax = plt.subplots(figsize=(min(max_width, 8), 3))  # Limit width
    for process, start, end in gantt_chart:
        ax.barh(process, end - start, left=start, height=0.25, color="#CC5500")
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
        max_width = 8  # Default width in inches
    fig, ax = plt.subplots(figsize=(min(max_width, 8), 3))  # Limit width
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
        for algo, (process_array, arrival_array, burst_array,priority_array, completion_array, 
                  turnaround_array, waiting_array, _) in results.items():
            for i in range(len(process_array)):
                detailed_data.append({
                    "Algorithm     ": algo,
                    "Process       ": process_array[i],
                    "Arrival Time  ": arrival_array[i],
                    "Priority      ":priority_array[i],
                    "Burst Time    ": burst_array[i],
                    "Completion Time": completion_array[i],
                    "Turnaround Time": turnaround_array[i],
                    "Waiting Time   ": waiting_array[i]
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
def adjust_zoom(zoom_level, factor, results, scrollable_frame, max_graph_width, buttons_frame, canvas):
    new_zoom = zoom_level.get() * 0.05  # Fine-tuned zoom step
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
                        fig.set_size_inches(new_width, 2 * new_zoom)  # Controlled height scaling
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
        self.style.configure("TButton", font=("Helvetica", 24, "bold"), padding=10)
        self.style.configure("Custom.TButton", font=("Helvetica", 16, "bold"), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 24))
        self.style.configure("Treeview.Heading", font=("Helvetica", 16, "bold"))

    def create_home_page(self):
        self.home_frame = tk.Frame(self.root, bg="#1E1E1E")  # Dark grey background
        self.home_frame.pack(fill="both", expand=True)
        self.root.state('zoomed')  # For Windows, maximizes the window
        #self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
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
                       text="Choose a process scheduling algorithm and see it in action with real-time visualization.", 
                       font=("Helvetica", 24, "bold"), 
                       fg="white", 
                       bg="#040406",
                       wraplength=800,  # Adjust this value as needed
                       justify="center")
        custom_text.pack(pady=20)
        button_frame = tk.Frame(content_frame, bg="#040406")
        button_frame.pack(pady=20)
        
        custom_text2 = tk.Label(content_frame, 
                       text="Compare multiple algorithms to analyze response time, waiting time, and CPU efficiency.                Start exploring now!", 
                       font=("Helvetica", 24, "bold"), 
                       fg="white", 
                       bg="#040406",
                       wraplength=800,  # Adjust this value as needed
                       justify="center")
        custom_text2.pack(pady=20)
        button_frame2 = tk.Frame(content_frame, bg="#040406")
        button_frame2.pack(pady=20)

        # Select Algorithm Dropdown
        self.algo_var = tk.StringVar(value="Select an Algorithm")
        select_menu = ttk.OptionMenu(button_frame, self.algo_var, "Select an Algorithm", 
                             "FCFS", "SJF (Non-Preemptive)", "SJF (Preemptive)", 
                             "Priority (Non-Preemptive)                                             ", "Priority (Preemptive)", 
                             "Round Robin", 
                             command=self.handle_algo_selection)

# Increase width for better appearance and set a consistent width
        select_menu.config(width=18)  # Increased from 18 to 25 for better visibility and alignment
        select_menu.pack(side="left", padx=40, pady=10)

        # Update the style for dropdown box
        style = ttk.Style()
        style.configure("TMenubutton", 
                        font=("Roboto", 20),  # Increase font size of the main button
                        background="white", foreground="black", 
                        padding=15,  # Increased padding for a larger dropdown
                        relief="solid", borderwidth=2,  # More visible border
                        width=25)  # Match the button width (in characters)
        style.map("TMenubutton", background=[("active", "#E0E0E0")])  # Change background on hover

        # Fix for dropdown menu size and width
        self.root.option_add('*TMenubutton*Listbox.font', ("Roboto", 50))  # Makes dropdown list items bigger
        self.root.option_add('*TMenubutton*Listbox.background', 'white')  # Background color of the dropdown
        self.root.option_add('*TMenubutton*Listbox.foreground', 'black')  # Text color
        self.root.option_add('*TMenubutton*Listbox.activeBackground', '#E0E0E0')  # Hover background
        self.root.option_add('*TMenubutton*Listbox.activeForeground', 'black')  # Hover text color
        self.root.option_add('*TMenubutton*Listbox.width', 35)  # Attempt to set dropdown width to match button


        # Compare Algorithms Button
        compare_button = ttk.Button(button_frame, 
                            text="Compare Algorithms", 
                            style="Custom.TButton", 
                            command=self.show_compare_input_page)
        style.configure("Custom.TButton", font=("Roboto", 20), 
                background="white", foreground="#FFFFFF", 
                padding=10)
        style.map("Custom.TButton", background=[("active", "#4A4A4A")])
        compare_button = ttk.Button(button_frame, 
                                    text="Compare Algorithms", 
                                    style="Custom.TButton", 
                                    command=self.show_compare_input_page)
        compare_button.pack(side="left", padx=40)  # Increased padx for spacing
        style.configure("Custom.TButton", font=("Roboto", 20), 
                        background="white", foreground="black", 
                        padding=10)
        style.map("Custom.TButton", background=[("active", "white")])

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
                 text=f"{algo} Input", font=("Helvetica", 28, "bold"), 
                 bg="#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                 fg="white" if self.theme == "dark" else "black").pack(pady=(20, 30))

        input_style = {"font": ("Helvetica", 22), 
                       "bg": "#2d2d2d" if self.theme == "dark" else "#f0f0f0", 
                       "fg": "white" if self.theme == "dark" else "black"}

        # Validation function for positive real numbers
        def validate_positive_real(action, value_if_allowed, prior_value, text, widget_name):
            if action != '1':  # Allow deletion
                return True
            try:
                if value_if_allowed.strip() == "":
                    return True
                num = float(value_if_allowed)
                if num < 0:
                    messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                    return False
                return True
            except ValueError:
                messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                return False

        # Validation function for space-separated positive real numbers
        def validate_positive_real_list(action, value_if_allowed, prior_value, text, widget_name):
            if action != '1':  # Allow deletion
                return True
            try:
                if value_if_allowed.strip() == "":
                    return True
                numbers = value_if_allowed.split()
                for num in numbers:
                    val = float(num)
                    if val < 0:
                        messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                        return False
                return True
            except ValueError:
                messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                return False

        # Register validation functions
        validate_cmd = self.root.register(validate_positive_real)
        validate_list_cmd = self.root.register(validate_positive_real_list)

        tk.Label(self.input_frame, text="Number of Processes:", **input_style).pack(pady=(10, 5))
        self.num_processes = tk.Entry(self.input_frame, font=("Helvetica", 22), width=20, 
                                      bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                      fg="white" if self.theme == "dark" else "black", 
                                      insertbackground="white" if self.theme == "dark" else "black",
                                      validate="key", validatecommand=(validate_cmd, '%d', '%P', '%s', '%S', '%W'))
        self.num_processes.pack(pady=10)

        tk.Label(self.input_frame, text="Arrival Times (space-separated):", **input_style).pack(pady=(10, 5))
        self.arrival_entry = tk.Entry(self.input_frame, font=("Helvetica", 22), width=20, 
                                      bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                      fg="white" if self.theme == "dark" else "black", 
                                      insertbackground="white" if self.theme == "dark" else "black",
                                      validate="key", validatecommand=(validate_list_cmd, '%d', '%P', '%s', '%S', '%W'))
        self.arrival_entry.pack(pady=10)

        tk.Label(self.input_frame, text="Burst Times (space-separated):", **input_style).pack(pady=(10, 5))
        self.burst_entry = tk.Entry(self.input_frame, font=("Helvetica", 22), width=20, 
                                    bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                    fg="white" if self.theme == "dark" else "black", 
                                    insertbackground="white" if self.theme == "dark" else "black",
                                    validate="key", validatecommand=(validate_list_cmd, '%d', '%P', '%s', '%S', '%W'))
        self.burst_entry.pack(pady=10)

        dwindled = None

        if "Priority" in algo:
            tk.Label(self.input_frame, text="Priorities (space-separated):", **input_style).pack(pady=(10, 5))
            self.priority_entry = tk.Entry(self.input_frame, font=("Helvetica", 22), width=20, 
                                           bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                           fg="white" if self.theme == "dark" else "black", 
                                           insertbackground="white" if self.theme == "dark" else "black",
                                           validate="key", validatecommand=(validate_list_cmd, '%d', '%P', '%s', '%S', '%W'))
            self.priority_entry.pack(pady=10)
        elif algo == "Round Robin":
            tk.Label(self.input_frame, text="Time Quantum:", **input_style).pack(pady=(10, 5))
            self.quantum_entry = tk.Entry(self.input_frame, font=("Helvetica", 22), width=20, 
                                          bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                                          fg="white" if self.theme == "dark" else "black", 
                                          insertbackground="white" if self.theme == "dark" else "black",
                                          validate="key", validatecommand=(validate_cmd, '%d', '%P', '%s', '%S', '%W'))
            self.quantum_entry.pack(pady=10)

        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=("Helvetica", 16, "bold"), padding=10)

        ttk.Button(self.input_frame, text="Run Simulation", style="Custom.TButton", 
                   command=self.run_simulation).pack(pady=20)
        ttk.Button(self.input_frame, text="Back to Home", style="Custom.TButton", 
                   command=self.back_to_home).pack(pady=10)

        self.apply_theme()
    

    def show_compare_input_page(self):
        # Hide the home frame
        self.home_frame.pack_forget()

        # Create the comparison input frame
        self.compare_input_frame = tk.Frame(self.root, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
        self.compare_input_frame.pack(fill="both", expand=True)

        # Set up scrollable canvas
        canvas = tk.Canvas(self.compare_input_frame, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.compare_input_frame, orient="vertical", command=canvas.yview)

        scrollable_frame = tk.Frame(canvas, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")

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

        # Container for inputs
        container = tk.Frame(scrollable_frame, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
        container.pack(expand=True, padx=20, pady=20)

        # Title
        tk.Label(container, 
                 text="Compare Scheduling Algorithms", 
                 font=("Helvetica", 26, "bold"), 
                 bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0", 
                 fg="white" if self.theme == "dark" else "black").pack(pady=(10, 20))

        input_style = {"font": ("Helvetica", 16), 
                       "bg": "#0C0C0C" if self.theme == "dark" else "#f0f0f0", 
                       "fg": "white" if self.theme == "dark" else "black"}

        # Validation function for non-negative real numbers
        def validate_positive_real(action, value_if_allowed, prior_value, text, widget_name):
            if action != '1':  # Allow deletion
                return True
            try:
                if value_if_allowed.strip() == "":
                    return True
                num = float(value_if_allowed)
                if num < 0:
                    messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                    return False
                return True
            except ValueError:
                messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                return False

        # Validation function for space-separated non-negative real numbers
        def validate_positive_real_list(action, value_if_allowed, prior_value, text, widget_name):
            if action != '1':  # Allow deletion
                return True
            try:
                if value_if_allowed.strip() == "":
                    return True
                numbers = value_if_allowed.split()
                for num in numbers:
                    val = float(num)
                    if val < 0:
                        messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                        return False
                return True
            except ValueError:
                messagebox.showerror("Invalid Input", "Only positive real numbers are allowed.")
                return False

        # Register validation functions
        validate_cmd = self.root.register(validate_positive_real)
        validate_list_cmd = self.root.register(validate_positive_real_list)

        # Input fields
        for text, var_name in [
            ("Number of Processes:", "num_processes"),
            ("Arrival Times (space-separated):", "arrival_entry"),
            ("Burst Times (space-separated):", "burst_entry"),
            ("Priorities (space-separated, optional):", "priority_entry"),
            ("Time Quantum (for Round Robin):", "quantum_entry")
        ]:
            input_frame = tk.Frame(container, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
            input_frame.pack(pady=10, fill="x")

            tk.Label(input_frame, text=text, width=30, anchor="w", **input_style).pack(side="left", padx=(0, 10))
            entry = tk.Entry(input_frame, font=("Helvetica", 16), width=25, 
                             bg="#404040" if self.theme == "dark" else "#e0e0e0", 
                             fg="white" if self.theme == "dark" else "black", 
                             insertbackground="white" if self.theme == "dark" else "black",
                             relief="flat", borderwidth=2,
                             validate="key", 
                             validatecommand=(validate_cmd if var_name in ["num_processes", "quantum_entry"] else validate_list_cmd, 
                                            '%d', '%P', '%s', '%S', '%W'))
            setattr(self, var_name, entry)
            entry.pack(side="left")

        # Algorithm Selection Section
        tk.Label(container, text="Select Algorithms to Compare:", **{**input_style, "font": ("Helvetica", 24)}).pack(pady=(90, 10))

        self.algo_vars = {}
        algorithms = ["FCFS", "SJF (Non-Preemptive)", "SJF (Preemptive)", 
                      "Priority (Non-Preemptive)", "Priority (Preemptive)", "Round Robin"]

        # Split into two rows, 3 algorithms each
        algo_frame = tk.Frame(container, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
        algo_frame.pack(pady=10)

        # First row
        row1_frame = tk.Frame(algo_frame, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
        row1_frame.pack(pady=(0, 10))
        for algo in algorithms[:3]:
            var = tk.BooleanVar(value=False)
            self.algo_vars[algo] = var
            chk = tk.Checkbutton(row1_frame, text=algo, variable=var, 
                                 font=("Helvetica", 16), 
                                 bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0", 
                                 fg="white" if self.theme == "dark" else "black", 
                                 selectcolor="#404040" if self.theme == "dark" else "#e0e0e0", 
                                 width=20, anchor="w", 
                                 padx=10, pady=5,
                                 height=2,
                                 indicatoron=True)
            chk.pack(side="left")

        # Second row
        row2_frame = tk.Frame(algo_frame, bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0")
        row2_frame.pack(pady=(0, 10))
        for algo in algorithms[3:]:
            var = tk.BooleanVar(value=False)
            self.algo_vars[algo] = var
            chk = tk.Checkbutton(row2_frame, text=algo, variable=var, 
                                 font=("Helvetica", 16), 
                                 bg="#0C0C0C" if self.theme == "dark" else "#f0f0f0", 
                                 fg="white" if self.theme == "dark" else "black", 
                                 selectcolor="#404040" if self.theme == "dark" else "#e0e0e0", 
                                 width=20, anchor="w", 
                                 padx=10, pady=5,
                                 height=2,
                                 indicatoron=True)
            chk.pack(side="left")

        # Button Styling
        button_style = ttk.Style()
        button_style.configure("Custom.TButton", font=("Helvetica", 16, "bold"), padding=3, 
                               background="gray" if self.theme == "dark" else "gray", 
                               foreground="black")

        # Wrapper to collect and validate inputs before calling run_comparison
        def run_comparison_wrapper():
            try:
                # Validate and parse inputs
                num_processes = int(self.num_processes.get().strip())
                if num_processes <= 0:
                    raise ValueError("Number of processes must be positive")

                arrival_times = [int(x) for x in self.arrival_entry.get().strip().split()]
                burst_times = [int(x) for x in self.burst_entry.get().strip().split()]
                if len(arrival_times) != num_processes or len(burst_times) != num_processes:
                    raise ValueError("Arrival and burst times must match number of processes")

                # Handle optional priority input
                priority_str = self.priority_entry.get().strip()
                if priority_str:
                    priorities = [int(x) for x in priority_str.split()]
                    if len(priorities) != num_processes:
                        raise ValueError("Priorities must match number of processes")
                else:
                    priorities = [0] * num_processes  # Default to zeros if not provided

                # Handle time quantum
                quantum_str = self.quantum_entry.get().strip()
                time_quantum = int(quantum_str) if quantum_str else 0

                # Collect selected algorithms
                selected_algos = [algo for algo, var in self.algo_vars.items() if var.get()]
                if not selected_algos:
                    raise ValueError("At least one algorithm must be selected")

                # Call run_comparison with validated inputs
                self.run_comparison(num_processes, arrival_times, burst_times, priorities, time_quantum, selected_algos)

            except ValueError as e:
                tk.messagebox.showerror("Input Error", str(e))
            except Exception as e:
                tk.messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

        ttk.Button(container, text="Run Comparison", style="Custom.TButton", 
                   command=run_comparison_wrapper).pack(pady=5)
        ttk.Button(container, text="Back to Home", style="Custom.TButton", 
                   command=self.back_to_home_from_compare).pack(pady=5)

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

    def run_comparison(self, num_processes, arrival_times, burst_times, priorities, time_quantum, selected_algos):
        try:
            # Convert inputs to numpy arrays for consistency with existing logic
            n = num_processes
            arrival_array = np.array(arrival_times)
            burst_array = np.array(burst_times)
            priority_array = np.array(priorities)  # Use provided priorities directly
            time_quantum = time_quantum if time_quantum > 0 else 4  # Default to 4 if invalid

            # Validate input lengths
            if len(arrival_array) != n or len(burst_array) != n or len(priority_array) != n:
                raise ValueError("Mismatch in number of processes and input arrays")

            # Ensure at least two algorithms are selected
            if len(selected_algos) < 2:
                raise ValueError("Select at least two algorithms to compare")

            # Run selected algorithms
            results = {}
            for algo in selected_algos:
                if algo == "FCFS":
                    results[algo] = first_come_first_serve(n, arrival_array, burst_array)
                elif algo == "SJF (Non-Preemptive)":
                    results[algo] = shortest_job_first(n, arrival_array, burst_array)
                elif algo == "SJF (Preemptive)":
                    results[algo] = shortest_job_first_preemptive(n, arrival_array, burst_array)
                elif algo == "Priority (Non-Preemptive)":
                    results[algo] = priority_scheduling(n, arrival_array, burst_array, priority_array)
                elif algo == "Priority (Preemptive)":
                    results[algo] = priority_scheduling_preemptive(n, arrival_array, burst_array, priority_array)
                elif algo == "Round Robin":
                    if time_quantum <= 0:
                        raise ValueError("Time quantum must be positive for Round Robin")
                    results[algo] = round_robin(n, arrival_array, burst_array, time_quantum)

            # Visualize results and hide input frame
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
