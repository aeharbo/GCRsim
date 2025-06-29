import tkinter as tk
import h5py
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import threading
from GCRsim_v02f import CosmicRaySimulation

def is_delta_of_primary(pid, primary_pid):
    # True if pid is a delta ray of primary_pid
    sp_pid  = (pid >> (11+14)) & ((1<<7)-1)
    pr_pid  = (pid >> 14) & ((1<<11)-1)
    delta   = pid & ((1<<14)-1)
    sp_prim = (primary_pid >> (11+14)) & ((1<<7)-1)
    pr_prim = (primary_pid >> 14) & ((1<<11)-1)
    return (sp_pid, pr_pid) == (sp_prim, pr_prim) and delta > 0

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GCRsim(alpha-build)")
        self.geometry("980x700")

        # style for completed progress bar
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('default')
        except Exception:
            pass
        self.style.configure("green.Horizontal.TProgressbar", background='green')

        # placeholders
        self.sim = None
        self.current_heatmap = None
        self.current_streaks = None
        self.current_count = None
        self._current_movie_pid = None
        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Simulation", command=self.new_sim)
        filemenu.add_command(label="Load Simulation", command=self.load_sim)
        filemenu.add_command(label="Save Simulation", command=self.save_sim)
        filemenu.add_separator()
        filemenu.add_command(label="Save Forecast", command=self.save_forecast)
        filemenu.add_command(label="Load Forecast", command=self.load_forecast)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        self.tab_control = ttk.Frame(self.notebook)
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_3d = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_advanced = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)
        self.tab_movie = ttk.Frame(self.notebook)
        tabs = [
            (self.tab_control, "Simulation"),
            (self.tab_heatmap, "Heatmap"),
            (self.tab_3d, "3D Trajectory"),
            (self.tab_movie, "Movie Mode"),    # <<<<< NEW TAB
            (self.tab_analysis, "Analysis"),
            (self.tab_advanced, "Advanced Config"),
            (self.tab_log, "Log"),
        ]
        for tab, text in tabs:
            self.notebook.add(tab, text=text)

        self.create_control_tab()
        self.create_heatmap_tab()
        self.create_3d_tab()
        self.create_movie_tab()
        self.create_analysis_tab()
        self.create_advanced_tab()
        self.create_log_tab()

    def create_control_tab(self):
        frame = self.tab_control
        params = ttk.LabelFrame(frame, text="Simulation Parameters")
        params.pack(side='top', fill='x', padx=5, pady=5)

        labels = ["Grid Size:", "Exposure Time (dt):", "Date (fractional year):", "Max Workers:"]
        vars_ = [(tk.IntVar, 4088, 'grid_size_var'),
                 (tk.DoubleVar, 3.4, 'dt_var'),
                 (tk.DoubleVar, 2026.123, 'date_var'),
                 (tk.IntVar, 4, 'max_workers_var')]
        for i, (label, (vtype, default, name)) in enumerate(zip(labels, vars_)):
            ttk.Label(params, text=label).grid(row=i, column=0, sticky='e')
            setattr(self, name, vtype(value=default))
            ttk.Entry(params, textvariable=getattr(self, name), width=10).grid(row=i, column=1)

        self.full_sim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Run all species", variable=self.full_sim_var).grid(row=4, columnspan=2, pady=5)
        
            # Buttons and simulation progress bar (in Simulation Parameters)
        btns = ttk.Frame(params)
        btns.grid(row=5, column=0, columnspan=2, pady=(8, 2), sticky='w')
        self.run_button = ttk.Button(btns, text="Run", command=self.run_sim)
        self.run_button.pack(side='left', padx=(0, 2))
        ttk.Button(btns, text="Pause", command=self.pause_sim).pack(side='left', padx=(0, 2))
        ttk.Button(btns, text="Stop", command=self.stop_sim).pack(side='left', padx=(0, 6))
        self.progress = ttk.Progressbar(btns, orient='horizontal', length=200)
        self.progress.pack(side='left', padx=5)
        
        flux_forecast_frame = ttk.LabelFrame(frame, text="Primary GCR Flux Forecast")
        flux_forecast_frame.pack(side='top', fill='x', padx=5, pady=(8,0))

        ttk.Label(flux_forecast_frame, text="Start Year:").pack(side='left')
        self.flux_start_var = tk.DoubleVar(value=2021)
        ttk.Entry(flux_forecast_frame, textvariable=self.flux_start_var, width=7).pack(side='left')

        ttk.Label(flux_forecast_frame, text="End Year:").pack(side='left', padx=(8,0))
        self.flux_end_var = tk.DoubleVar(value=2022)
        ttk.Entry(flux_forecast_frame, textvariable=self.flux_end_var, width=7).pack(side='left')

        ttk.Label(flux_forecast_frame, text="Instances per Date:").pack(side='left', padx=(8,0))
        self.flux_ninst_var = tk.IntVar(value=2)
        ttk.Entry(flux_forecast_frame, textvariable=self.flux_ninst_var, width=4).pack(side='left')

        self.flux_button = ttk.Button(flux_forecast_frame, text="Predict Flux", command=self.predict_flux)
        self.flux_button.pack(side='top', pady=(3,0))
        self.flux_progress = ttk.Progressbar(flux_forecast_frame, orient='horizontal', length=200, style="green.Horizontal.TProgressbar")
        self.flux_progress.pack(side='top', fill='x', padx=5, pady=(2,8))
        self.flux_progress["mode"] = "determinate"

        self.flux_fig = Figure(figsize=(5,2.2), tight_layout=True)
        self.flux_ax = self.flux_fig.add_subplot(111)
        self.flux_canvas = FigureCanvasTkAgg(self.flux_fig, master=frame)
        self.flux_canvas.get_tk_widget().pack(fill='x', padx=10, pady=(10, 5))
        # Add navigation toolbar for interactivity
        self.flux_nav = NavigationToolbar2Tk(self.flux_canvas, frame)
        self.flux_nav.update()
        self.flux_nav.pack(side='top', fill='x', padx=10, pady=(0, 3))

    def create_heatmap_tab(self):
        fig = Figure(figsize=(5,5))
        self.heatmap_ax = fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        self.heatmap_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.heatmap_canvas, self.tab_heatmap)
        nav.update()
        nav.pack(side='bottom', fill='x')

    def create_3d_tab(self):
        frame = self.tab_3d

        control_row = ttk.Frame(frame)
        control_row.pack(side='top', fill='x', padx=5, pady=3)

        ttk.Label(control_row, text="Select Primary PID:").pack(side='left')
        self.selected_primary_pid = tk.StringVar()
        self.primary_pid_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_primary_pid, state='readonly'
        )
        self.primary_pid_combobox.pack(side='left')
        self.primary_pid_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_3d_delta_dropdown())
        self.primary_pid_combobox.state(['disabled'])

        ttk.Label(control_row, text="Delta Ray PID:").pack(side='left')
        self.selected_3d_delta_pid = tk.StringVar()
        self.combo_3d_delta_pid = ttk.Combobox(
            control_row, textvariable=self.selected_3d_delta_pid, state='readonly'
        )
        self.combo_3d_delta_pid.pack(side='left', padx=(5,0))
        self.combo_3d_delta_pid.state(['disabled'])
        self.combo_3d_delta_pid.bind('<<ComboboxSelected>>', lambda e: self._on_3d_delta_selection())

        fig = Figure(figsize=(5,5))
        self.traj_ax = fig.add_subplot(111, projection='3d')
        self.traj_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.traj_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.traj_canvas, frame)
        nav.update()
        nav.pack(side='bottom', fill='x')
        
    def create_movie_tab(self):
        frame = self.tab_movie

        # Dropdowns for PID selection
        control_row = ttk.Frame(frame)
        control_row.pack(side='top', fill='x', padx=5, pady=3)

        ttk.Label(control_row, text="Primary PID:").pack(side='left')
        self.selected_movie_primary = tk.StringVar()
        self.movie_primary_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_movie_primary, state='readonly'
        )
        self.movie_primary_combobox.pack(side='left', padx=(0, 10))
        self.movie_primary_combobox.state(['disabled'])
        self.movie_primary_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_movie_primary())

        ttk.Label(control_row, text="Delta Ray PID:").pack(side='left')
        self.selected_movie_delta = tk.StringVar()
        self.movie_delta_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_movie_delta, state='readonly'
        )
        self.movie_delta_combobox.pack(side='left')
        self.movie_delta_combobox.state(['disabled'])
        self.movie_delta_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_movie_delta())

        # 3D Figure
        self.movie_fig = Figure(figsize=(5, 5))
        self.movie_ax = self.movie_fig.add_subplot(111, projection='3d')
        self.movie_canvas = FigureCanvasTkAgg(self.movie_fig, master=frame)
        self.movie_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.movie_canvas, frame)
        nav.update()
        nav.pack(side='bottom', fill='x')

        # Movie controls
        movie_frame = ttk.Frame(frame)
        movie_frame.pack(side='bottom', fill='x', padx=8, pady=4)
        self.movie_playing = False
        self.movie_frame_idx = 0

        self.btn_play = ttk.Button(movie_frame, text="Play", width=7, command=self._movie_play)
        self.btn_pause = ttk.Button(movie_frame, text="Pause", width=7, command=self._movie_pause)
        self.btn_rewind = ttk.Button(movie_frame, text="Rewind", width=7, command=self._movie_rewind)
        self.movie_slider = ttk.Scale(movie_frame, from_=0, to=0, orient='horizontal', command=self._movie_slider_move)

        self.btn_play.pack(side='left')
        self.btn_pause.pack(side='left', padx=4)
        self.btn_rewind.pack(side='left', padx=4)
        self.movie_slider.pack(side='left', fill='x', expand=True, padx=(10,0))

    def _populate_movie_primary_dropdown(self):
        # Like _populate_primary_pid_dropdown, but for movie mode
        all_pids = {}
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    _, pid, *_ = streak
                    if (pid & ((1<<14)-1)) == 0:
                        all_pids[pid] = self.sim.decode_pid(pid)
        items = sorted(all_pids.items(), key=lambda x: x[1])
        self.movie_primaries_list = items
        self.movie_primary_combobox['values'] = [v for k,v in items]
        if items:
            self.selected_movie_primary.set(items[0][1])
            self.movie_primary_combobox.state(['!disabled'])
            self._update_movie_primary()
        else:
            self.selected_movie_primary.set('')
            self.movie_primary_combobox.state(['disabled'])
            self.movie_delta_combobox['values'] = []
            self.movie_delta_combobox.set('')
            self.movie_delta_combobox.state(['disabled'])

    def _update_movie_primary(self):
        selection = self.selected_movie_primary.get()
        pid_int = None
        # Find the PID int for the selected label
        for k, v in self.movie_primaries_list:
            if v == selection:
                pid_int = k
                break

        # Set the current movie PID, or clear if not found
        self._current_movie_pid = pid_int

        if pid_int is None:
            self.movie_delta_combobox['values'] = []
            self.movie_delta_combobox.set('')
            self.movie_delta_combobox.state(['disabled'])
            self._movie_clear()
            return

        # Find all delta rays of this primary
        children = []
        for species in (self.current_streaks or []):
            for bin in species:
                for stk in bin:
                    _, pid, *_ = stk
                    if (
                        ((pid >> (11+14)) == (pid_int >> (11+14))) and
                        (((pid >> 14) & ((1<<11)-1)) == ((pid_int >> 14) & ((1<<11)-1))) and
                        ((pid & ((1<<14)-1)) > 0)
                    ):
                        children.append((pid, self.sim.decode_pid(pid)))
        children = sorted(children, key=lambda x: x[1])
        self.movie_delta_children = children
        self.movie_delta_combobox['values'] = [c[1] for c in children]
        if children:
            self.movie_delta_combobox.state(['!disabled'])
            self.selected_movie_delta.set(children[0][1])
        else:
            self.movie_delta_combobox.state(['disabled'])
            self.selected_movie_delta.set('')
        # Show the selected streak by default
        streak = self._find_streak_by_pid(pid_int)
        if streak:
            positions = streak[0]
            if positions and len(positions) > 1:
                self._setup_movie_controls(positions, which='movie')

    def _update_movie_delta(self):
        selection = self.selected_movie_delta.get()
        pid_int = None
        for k, v in self.movie_delta_children:
            if v == selection:
                pid_int = k
                break
        if pid_int is not None:
            streak = self._find_streak_by_pid(pid_int)
            if streak:
                positions = streak[0]
                if positions and len(positions) > 1:
                    self._setup_movie_controls(positions, which='movie')

    def _movie_clear(self):
        self.movie_ax.clear()
        self.movie_canvas.draw()

    def _setup_movie_controls(self, positions, which='movie'):
        # For which='movie', use self.movie_ax etc
        if which == 'movie':
            self._movie_positions = positions
            self.movie_frame_idx = 0
            self.movie_slider.config(to=len(positions)-1)
            self.movie_slider.set(0)
            self.movie_playing = False
            self._movie_draw_frame(0)

    def _movie_draw_frame(self, idx):
        primary_positions = self._movie_positions  # (Already selected)
        idx = int(idx)
        xs, ys, zs = zip(*primary_positions[:idx+1])
        self.movie_ax.clear()
        # Draw primary
        self.movie_ax.plot(xs, ys, zs, '-o', color='royalblue', markersize=3, label='Primary')
        self.movie_ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='red', s=40, zorder=10)

        # --- Draw all associated delta rays
        primary_pid = self._current_movie_pid  # You'll need to track this on selection
        # Find all delta rays for this primary
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    positions, pid, *rest = streak
                    if is_delta_of_primary(pid, primary_pid):
                        # Optionally: determine at which primary step this delta is created (emission_step)
                        # For now, just draw all, up to current idx
                        n_draw = min(idx + 1, len(positions))
                        if n_draw > 1:
                            dx, dy, dz = zip(*positions[:n_draw])
                            self.movie_ax.plot(dx, dy, dz, '-', color='orange', alpha=0.6, label='Delta' if 'Delta' not in self.movie_ax.get_legend_handles_labels()[1] else None)
                            self.movie_ax.scatter([dx[-1]], [dy[-1]], [dz[-1]], color='gold', s=20, zorder=9)

        self.movie_ax.set_xlabel("X (μm)")
        self.movie_ax.set_ylabel("Y (μm)")
        self.movie_ax.set_zlabel("Z (μm)")
        self.movie_ax.set_title(f"Movie Mode: Step {idx+1}/{len(primary_positions)}")
        self.movie_ax.set_zlim(5, 0)
        self.movie_ax.legend()
        self.movie_canvas.draw()

    def _movie_play(self):
        if self.movie_playing: return
        self.movie_playing = True
        self._movie_animate()

    def _movie_pause(self):
        self.movie_playing = False

    def _movie_rewind(self):
        self.movie_playing = False
        self.movie_frame_idx = 0
        self.movie_slider.set(0)
        self._movie_draw_frame(0)

    def _movie_animate(self):
        if not self.movie_playing: return
        if self.movie_frame_idx < len(self._movie_positions)-1:
            self.movie_frame_idx += 1
            self.movie_slider.set(self.movie_frame_idx)
            self._movie_draw_frame(self.movie_frame_idx)
            self.after(60, self._movie_animate)  # 60 ms per frame ~16 FPS
        else:
            self.movie_playing = False  # stop at the end

    def _movie_slider_move(self, val):
        idx = int(float(val))
        self.movie_frame_idx = idx
        self._movie_draw_frame(idx)

    def predict_flux(self):
        self.flux_button.config(state='disabled')
        self.flux_progress['value'] = 0
        self._ensure_sim()
        start_date = self.flux_start_var.get()
        end_date = self.flux_end_var.get()
        n_instances = self.flux_ninst_var.get()
        df = self.sim.historic_df

        # Generate all dates at the spacing in your DataFrame (assume monthly spacing)
        date_min, date_max = df['date'].min(), df['date'].max()
        # Step size: median difference (should work for monthly/yearly)
        step = np.median(np.diff(df['date'].values))
        requested_dates = np.arange(start_date, end_date + step, step)
        
        # Map each requested date to a usable historical date (for M_value)
        mapped_dates = []
        for d in requested_dates:
            if d <= date_max:
                mapped_dates.append(d)
            else:
                # Wrap back by 22-year periodicity
                mapped_dates.append(d - 22)
        # Just as a tuple so you can track both real and mapped dates
        reference_dates = list(zip(requested_dates, mapped_dates))
        # Optionally: validate mapped_dates all in range
        mapped_dates = np.array(mapped_dates)
        if np.any(mapped_dates < date_min):
            messagebox.showwarning("Forecast Error", f"Some forecasted dates are out of range after mapping. Try a smaller date range.")
            self.flux_button.config(state='normal')
            return

        if n_instances < 1:
            messagebox.showwarning("Invalid Instances", "Instances per date must be at least 1.")
            self.flux_button.config(state='normal')
            return
        exposure_time = self.dt_var.get()
        grid_size = self.grid_size_var.get()

        threading.Thread(
            target=self._predict_flux_worker,
            args=(reference_dates, n_instances, exposure_time, grid_size),
            daemon=True
        ).start()

    def _predict_flux_worker(self, reference_dates, n_instances, exposure_time, grid_size):
        avg_particles = []
        std_particles = []
        all_particles = []
        progress_total = len(reference_dates) * n_instances
        self.flux_progress["maximum"] = progress_total

        for i, (real_date, mapped_date) in enumerate(reference_dates):
            counts = []
            for j in range(n_instances):
                # For each sim, use the mapped_date for M_value calculations
                sim = CosmicRaySimulation(
                    grid_size=grid_size,
                    dt=exposure_time,
                    date=mapped_date,     # <-- this is the only key difference!
                    historic_df=self.sim.historic_df,
                    progress_bar=True
                )
                _, _, particle_counts = sim.run_sim()
                if isinstance(particle_counts, int):
                    total_particles = particle_counts
                elif isinstance(particle_counts, (list, tuple)):
                    total_particles = sum(particle_counts)
                elif isinstance(particle_counts, dict):
                    total_particles = sum(particle_counts.values())
                else:
                    total_particles = 0
                counts.append(total_particles)
                del sim
                self.flux_progress["value"] += 1
                self.flux_progress.update()
            avg_particles.append(np.mean(counts))
            std_particles.append(np.std(counts))
            all_particles.append(counts)
        self._last_flux_dates = np.array(reference_dates)
        self._last_flux_avg = np.array(avg_particles)
        self._last_flux_std = np.array(std_particles)
        self._last_flux_all = np.array(all_particles)
        self._last_flux_meta = dict(grid_size=grid_size, dt=exposure_time,
                                    species_index=1, species_label="Hydrogen")

        # For plotting, use real_date
        real_dates = [rd for rd, _ in reference_dates]
        self.after(0, self._plot_flux_results, real_dates, avg_particles, std_particles)
        
    def _plot_flux_results(self, dates, avg_particles, std_particles, grid_size=None, dt=None, species=None):
        self.flux_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.flux_ax.clear()
        # Always prefer meta info
        if grid_size is None and hasattr(self, "_last_flux_meta"):
            grid_size = self._last_flux_meta.get("grid_size", "")
        if dt is None and hasattr(self, "_last_flux_meta"):
            dt = self._last_flux_meta.get("dt", "")
        if species is None and hasattr(self, "_last_flux_meta"):
            species = self._last_flux_meta.get("species_label", "Hydrogen")

        subtitle = f"Grid Size: {grid_size}x{grid_size} pixels, Exposure Time (dt): {dt} sec, Species: {species}+ ions"
        self.flux_ax.errorbar(
            dates, avg_particles, yerr=std_particles,
            fmt='o-', mfc='red', mec='black', ecolor='blue', alpha=0.75, capsize=3,
            label='Avg count per date'
        )
        if len(avg_particles) > 0:
            max_idx = int(np.argmax(avg_particles))
            max_date = dates[max_idx]
            max_value = avg_particles[max_idx]
            self.flux_ax.annotate(f"Max: {max_value:.0f}\nYear: {max_date:.3f}",
                xy=(max_date, max_value),
                xytext=(max_date, max_value + 0.25 * max_value),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                ha='center')
        self.flux_ax.set_xlabel("Date")
        self.flux_ax.set_ylabel("Predicted H+ GCR Count")
        self.flux_ax.set_title('Galactic Cosmic Ray Flux Forecast', pad=18)
        # Add subtitle just below the main title using ax.text
        self.flux_ax.text(0.5, 1.02, subtitle, ha='center', va='bottom', fontsize='medium', transform=self.flux_ax.transAxes)
        self.flux_ax.legend()
        self.flux_canvas.draw()
        self.flux_button.config(state='normal')
        self.flux_progress['value'] = 0  # Reset

    def _on_3d_delta_selection(self):
        self._plot_3d_for_primary_and_delta()

    def _populate_primary_pid_dropdown(self):
        all_pids = set()
        pid_to_human = {}
        for species in (self.current_streaks or []):
            for energy_bin in species:
                for streak in energy_bin:
                    positions, pid, *_ = streak
                    delta_idx = pid & ((1<<14)-1)
                    if delta_idx == 0:
                        all_pids.add(pid)
                        pid_to_human[pid] = self.sim.decode_pid(pid)
        sorted_pids = sorted(all_pids)
        self.primary_pid_choices = [(pid_to_human[pid], pid) for pid in sorted_pids]
        human_list = [s for s, p in self.primary_pid_choices]
        self.primary_pid_combobox['values'] = human_list
        if human_list:
            self.selected_primary_pid.set(human_list[0])
            self.primary_pid_combobox.state(['!disabled'])
            self._plot_3d_for_primary_and_delta()
        else:
            self.selected_primary_pid.set('')
            self.primary_pid_combobox.state(['disabled'])
        self._update_3d_delta_dropdown()

    def _update_3d_delta_dropdown(self):
        # Populate the delta ray dropdown based on selected primary
        selection = self.selected_primary_pid.get()
        pid_int = None
        for human, pid in self.primary_pid_choices:
            if human == selection:
                pid_int = pid
                break
        if pid_int is None:
            self.combo_3d_delta_pid['values'] = []
            self.combo_3d_delta_pid.set('')
            self.combo_3d_delta_pid.state(['disabled'])
            self._plot_3d_for_primary_and_delta()
            return

        # Find delta rays for this parent
        species_idx = (pid_int >> (11+14)) & ((1<<7)-1)
        primary_idx = (pid_int >> 14) & ((1<<11)-1)
        children = []
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    _, pid, *_ = streak
                    sp = (pid >> (11+14)) & ((1<<7)-1)
                    pr = (pid >> 14) & ((1<<11)-1)
                    delta = pid & ((1<<14)-1)
                    if (sp, pr) == (species_idx, primary_idx) and delta > 0:
                        children.append(pid)
        items = [("Show All", None)] + [(self.sim.decode_pid(pid), pid) for pid in children]
        self.delta_choices_3d = items
        self.combo_3d_delta_pid['values'] = [s for s, _ in items]
        self.combo_3d_delta_pid.set("Show All")
        self.combo_3d_delta_pid.state(['!disabled'])
        self._plot_3d_for_primary_and_delta()

    def _plot_3d_for_primary_and_delta(self):
        # Get selection
        primary_human = self.selected_primary_pid.get()
        delta_human = self.selected_3d_delta_pid.get()
        # Get the integer PID of the selected primary
        pid_int = None
        for human, pid in self.primary_pid_choices:
            if human == primary_human:
                pid_int = pid
                break
        if pid_int is None:
            self.traj_ax.clear(); self.traj_canvas.draw(); return

        species_idx = (pid_int >> (11+14)) & ((1<<7)-1)
        primary_idx = (pid_int >> 14) & ((1<<11)-1)
        parent_mask = (species_idx, primary_idx)

        # See if we are showing all, or a specific delta
        show_all = (delta_human == "Show All" or not delta_human)
        chosen_delta_pid = None
        if not show_all and hasattr(self, 'delta_choices_3d'):
            for s, pid in self.delta_choices_3d:
                if s == delta_human:
                    chosen_delta_pid = pid
                    break

        streaks_to_plot = []
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    positions, pid, *_ = streak
                    sp = (pid >> (11+14)) & ((1<<7)-1)
                    pr = (pid >> 14) & ((1<<11)-1)
                    delta = pid & ((1<<14)-1)
                    if (sp, pr) == parent_mask:
                        if show_all:
                            streaks_to_plot.append((positions, pid))
                        elif chosen_delta_pid is not None and pid == chosen_delta_pid:
                            streaks_to_plot.append((positions, pid))

        self.traj_ax.clear()
        for positions, pid in streaks_to_plot:
            if len(positions) < 2: continue
            xs, ys, zs = zip(*positions)
            col = self.sim.get_particle_color(pid)
            is_primary = (pid & ((1<<14)-1)) == 0
            alpha = 0.9 if is_primary else 0.4
            lw = 2.5 if is_primary else 1
            self.traj_ax.plot(xs, ys, zs, '-', color=col, alpha=alpha, linewidth=lw)
        self.traj_ax.set_xlabel("X (μm)")
        self.traj_ax.set_ylabel("Y (μm)")
        self.traj_ax.set_zlabel("Z (μm)")
        self.traj_ax.set_title(f"3D Trajectory: {primary_human}" +
                            ("" if show_all else f" > {delta_human}"))
        self.traj_ax.set_zlim(5, 0)
        self.traj_canvas.draw()

    def create_analysis_tab(self):
        frame = self.tab_analysis
        control_row = ttk.Frame(frame)
        control_row.pack(side='top', fill='x', padx=5, pady=3)

        # Dropdowns (as before)...
        ttk.Label(control_row, text="Primary PID:").pack(side='left')
        self.selected_analysis_primary = tk.StringVar()
        self.analysis_primary_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_analysis_primary, state='readonly'
        )
        self.analysis_primary_combobox.pack(side='left', padx=(0, 10))
        self.analysis_primary_combobox.state(['disabled'])
        self.analysis_primary_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_analysis_primary())

        ttk.Label(control_row, text="Delta Ray PID:").pack(side='left')
        self.selected_analysis_delta = tk.StringVar()
        self.analysis_delta_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_analysis_delta, state='readonly'
        )
        self.analysis_delta_combobox.pack(side='left')
        self.analysis_delta_combobox.state(['disabled'])
        self.analysis_delta_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_analysis_delta())

        # Info
        self.analysis_info = tk.Text(frame, height=11, width=80, wrap='word', font=('Consolas', 10))
        self.analysis_info.pack(fill='x', padx=10, pady=3)
        self.analysis_info.config(state='disabled')

        # Positions table (scrollable)
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='x', padx=10, pady=2)
        self.positions_table = tk.Text(table_frame, height=6, width=90, font=('Consolas', 9))
        self.positions_table.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.positions_table.yview)
        scrollbar.pack(side='left', fill='y')
        self.positions_table['yscrollcommand'] = scrollbar.set
        self.positions_table.config(state='disabled')

        # Export buttons
        export_frame = ttk.Frame(frame)
        export_frame.pack(fill='x', padx=10, pady=(0,5))
        ttk.Button(export_frame, text="Export Table (CSV)", command=self._export_positions_table).pack(side='left', padx=3)
        ttk.Button(export_frame, text="Export Energy Plot", command=self._export_energy_plot).pack(side='left', padx=3)
        ttk.Button(export_frame, text="Export Angles Plot", command=self._export_angles_plot).pack(side='left', padx=3)

        # Plots
        plot_frame = ttk.Frame(frame)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=2)

        # Energy loss plot
        self.energy_fig = Figure(figsize=(4,2.2))
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_canvas = FigureCanvasTkAgg(self.energy_fig, master=plot_frame)
        self.energy_canvas.get_tk_widget().pack(side='left', fill='both', expand=True)
        # Angles plot
        self.angles_fig = Figure(figsize=(4,2.2))
        self.angles_ax = self.angles_fig.add_subplot(111)
        self.angles_canvas = FigureCanvasTkAgg(self.angles_fig, master=plot_frame)
        self.angles_canvas.get_tk_widget().pack(side='left', fill='both', expand=True)


    def create_advanced_tab(self):
        ttk.Label(self.tab_advanced, text="Advanced configuration coming soon...").pack(pady=10)

    def create_log_tab(self):
        self.log_text = tk.Text(self.tab_log, wrap='none', height=10)
        self.log_text.pack(fill='both', expand=True)

    def new_sim(self):
        for attr in ('current_heatmap','current_streaks','current_count'):
            setattr(self, attr, None)
        self.sim = None
        self._ensure_sim()
        self.heatmap_ax.clear(); self.heatmap_canvas.draw()
        self.traj_ax.clear();   self.traj_canvas.draw()
        self.log_text.delete('1.0','end')
        self.analysis_primary_combobox.state(['disabled'])
        self.analysis_primary_combobox.set('')
        self.analysis_delta_combobox.state(['disabled'])
        self.analysis_delta_combobox.set('')
        self._clear_analysis_info()
        self.combo_3d_delta_pid.state(['disabled'])
        self.combo_3d_delta_pid.set('')
        self.delta_choices_3d = []

    def _ensure_sim(self):
        # Always make sure self.sim exists for any plotting or saving
        if self.sim is None:
            grid = self.grid_size_var.get()
            dt = self.dt_var.get()
            date = self.date_var.get()
            maxw = self.max_workers_var.get()
            self.sim = CosmicRaySimulation(grid_size=grid, dt=dt, date=date, progress_bar=True, max_workers=maxw)

    def load_sim(self):
        f = filedialog.askopenfilename(filetypes=[('HDF5 files','*.h5')])
        if not f: return
        hm, st, ct = CosmicRaySimulation.load_sim(f)
        self.current_heatmap, self.current_streaks, self.current_count = hm, st, ct
        self._ensure_sim()
        self._update_heatmap(hm)
        self._display_results()
        self.after(0, self._populate_primary_pid_dropdown)
        self.after(0, self._populate_analysis_primary_dropdown)
        self.after(0, self._populate_movie_primary_dropdown)
        messagebox.showinfo('Loaded', f'Simulation loaded from {f}')
        
    def save_sim(self):
        if self.current_heatmap is None:
            messagebox.showwarning('No Data', 'Run or load a simulation first.')
            return
        f = filedialog.asksaveasfilename(defaultextension='.h5', filetypes=[('HDF5 files', '*.h5')])
        if not f: return
        self._ensure_sim()
        # Standardize gcr_counts: if it's int, wrap as single-element list with name
        gcr_counts = self.current_count
        if isinstance(gcr_counts, int):
            # Try to get species name from the current sim
            try:
                idx = self.sim.species_index if hasattr(self.sim, 'species_index') else 0
                sp_name = self.sim.species_names.get(idx, f"Z={self.sim.Z_particle}")
            except Exception:
                sp_name = "Unknown"
            gcr_counts = [(sp_name, gcr_counts)]
        self.sim.save_sim(self.current_heatmap, self.current_streaks, gcr_counts, f)
        messagebox.showinfo('Saved', f'Simulation saved to {f}')

    def save_forecast(self):
        # Prompt for file location
        fpath = filedialog.asksaveasfilename(
            defaultextension='.h5',
            filetypes=[('HDF5 files', '*.h5')],
            title="Save Flux Forecast as HDF5"
        )
        if not fpath:
            return

        # These should be set at the end of _predict_flux_worker and _plot_flux_results
        try:
            dates = self._last_flux_dates
            # If dates is a list of pairs/tuples/2D array:
            dates_for_save = np.array(dates)
            # Store the whole array if you want both, but for plotting, you'll use only the first column
            with h5py.File(fpath, "w") as f:
                f.create_dataset("dates", data=dates_for_save)
            avg_particles = self._last_flux_avg
            std_particles = self._last_flux_std
            all_particles = self._last_flux_all
            grid_size = self.grid_size_var.get()
            dt = self.dt_var.get()
            species = getattr(self.sim, 'species_label', 'unknown')
        except Exception as e:
            messagebox.showerror("Error", "No forecast data to save.\n\n" + str(e))
            return

        with h5py.File(fpath, "w") as f:
            f.create_dataset("dates", data=dates)
            f.create_dataset("avg_particles", data=avg_particles)
            f.create_dataset("std_particles", data=std_particles)
            f.create_dataset("all_particles", data=all_particles)
            f.attrs["grid_size"] = grid_size
            f.attrs["dt"] = dt
            f.attrs["species_index"] = 1        # Always H
            f.attrs["species_label"] = "Hydrogen"
        messagebox.showinfo("Saved", f"Forecast saved to:\n{fpath}")

    def load_forecast(self):
        fpath = filedialog.askopenfilename(
            filetypes=[('HDF5 files', '*.h5')],
            title="Load Flux Forecast"
        )
        if not fpath:
            return

        with h5py.File(fpath, "r") as f:
            dates = f["dates"][:]
            avg_particles = f["avg_particles"][:]
            std_particles = f["std_particles"][:]
            all_particles = f["all_particles"][:]
            grid_size = f.attrs.get("grid_size", None)
            dt = f.attrs.get("dt", None)
            species_label = f.attrs.get("species_label", "Hydrogen")  # Default to H if missing
            species_index = f.attrs.get("species_index", 1)


        # Check shape and extract first column if 2D
        if dates.ndim == 2 and dates.shape[1] == 2:
            plot_dates = dates[:, 0]
        else:
            plot_dates = dates

        self._last_flux_dates = dates
        self._last_flux_avg = avg_particles
        self._last_flux_std = std_particles
        self._last_flux_all = all_particles
        self._last_flux_meta = dict(grid_size=grid_size, dt=dt,
                                    species_index=species_index, species_label=species_label)
        self._plot_flux_results(plot_dates, avg_particles, std_particles, grid_size, dt, species_label)
        messagebox.showinfo("Loaded", f"Forecast loaded from:\n{fpath}")

    def run_sim(self):
        self.run_button.config(state='disabled')
        self.heatmap_ax.clear(); self.traj_ax.clear(); self.heatmap_canvas.draw(); self.traj_canvas.draw()
        grid = self.grid_size_var.get(); dt = self.dt_var.get(); date = self.date_var.get(); maxw = self.max_workers_var.get()
        full = self.full_sim_var.get()
        self.sim = CosmicRaySimulation(grid_size=grid, dt=dt, date=date, progress_bar=True, max_workers=maxw)
        if full:
            total = len(CosmicRaySimulation.Z_list)
            self.progress.config(mode='determinate', maximum=total, value=0)
        else:
            self.progress.config(mode='indeterminate'); self.progress.start(10)
        threading.Thread(target=self._run_sim_thread, daemon=True).start()


    def _run_sim_thread(self):
        full = self.full_sim_var.get()
        if full:
            hm = np.zeros((self.sim.grid_size,self.sim.grid_size),int)
            streaks_all=[]; counts=[]
            for idx in range(len(CosmicRaySimulation.Z_list)):
                sim_i = CosmicRaySimulation(species_index=idx, grid_size=self.sim.grid_size,
                                            dt=self.sim.dt, date=self.sim.date,
                                            progress_bar=True, max_workers=self.sim.max_workers)
                h_i, s_i, c_i = sim_i.run_sim()
                hm += h_i; streaks_all.append(s_i); counts.append(c_i)
                self.after(0, self.progress.step, 1)
            self.current_heatmap, self.current_streaks, self.current_count = hm, streaks_all, sum(counts)
            self.after(0, self._update_heatmap, hm)
        else:
            h, s, c = self.sim.run_sim()
            self.current_heatmap, self.current_streaks, self.current_count = h, s, c
            self.after(0, self.progress.stop)
            self.after(0, lambda: self.progress.config(mode='determinate', value=self.progress['maximum']))
            self.after(0, self._update_heatmap, h)
        self.after(0, self._display_results)
        self.after(0, self.run_button.config, {'state':'normal'})
        maxv = self.progress['maximum']
        self.after(0, self.progress.config, {'style':'green.Horizontal.TProgressbar', 'value':maxv})
        self.after(0, self._populate_primary_pid_dropdown)
        self.after(0, self._populate_analysis_primary_dropdown)
        self.after(0, self._populate_movie_primary_dropdown)
        
    def _populate_analysis_primary_dropdown(self):
        # Find all primary PIDs (delta_idx==0)
        all_primaries = {}
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    _, pid, *_ = streak
                    if (pid & ((1<<14)-1)) == 0:  # delta_idx==0
                        all_primaries[pid] = self.sim.decode_pid(pid)
        items = sorted(all_primaries.items(), key=lambda x: x[1])
        self.analysis_primaries_list = items
        self.analysis_primary_combobox['values'] = [v for k,v in items]
        if items:
            self.selected_analysis_primary.set(items[0][1])
            self.analysis_primary_combobox.state(['!disabled'])
            self._update_analysis_primary()
        else:
            self.selected_analysis_primary.set('')
            self.analysis_primary_combobox.state(['disabled'])
            self.analysis_delta_combobox['values'] = []
            self.analysis_delta_combobox.set('')
            self.analysis_delta_combobox.state(['disabled'])
            self._clear_analysis_info()

    def _update_analysis_primary(self):
        selection = self.selected_analysis_primary.get()
        pid_int = None
        for k, v in self.analysis_primaries_list:
            if v == selection:
                pid_int = k
                break
        if pid_int is None:
            self.analysis_delta_combobox['values'] = []
            self.analysis_delta_combobox.set('')
            self.analysis_delta_combobox.state(['disabled'])
            self._clear_analysis_info()
            return

        # Find all streaks matching this primary PID
        streak = self._find_streak_by_pid(pid_int)
        if streak:
            self._display_streak_info(streak)
        else:
            self._clear_analysis_info()

        # Find all children delta rays
        children = []
        for species in (self.current_streaks or []):
            for bin in species:
                for stk in bin:
                    _, pid, *_ = stk
                    # Same species & primary idx, delta idx>0
                    if ((pid >> (11+14)) == (pid_int >> (11+14)) and
                        ((pid >> 14) & ((1<<11)-1)) == ((pid_int >> 14) & ((1<<11)-1)) and
                        (pid & ((1<<14)-1)) > 0):
                        children.append((pid, self.sim.decode_pid(pid)))
        children = sorted(children, key=lambda x: x[1])
        self.analysis_delta_children = children
        self.analysis_delta_combobox['values'] = [c[1] for c in children]
        if children:
            self.analysis_delta_combobox.state(['!disabled'])
            self.selected_analysis_delta.set(children[0][1])
        else:
            self.analysis_delta_combobox.state(['disabled'])
            self.selected_analysis_delta.set('')

    def _update_analysis_delta(self):
        selection = self.selected_analysis_delta.get()
        pid_int = None
        for k, v in self.analysis_delta_children:
            if v == selection:
                pid_int = k
                break
        if pid_int is not None:
            streak = self._find_streak_by_pid(pid_int)
            if streak:
                self._display_streak_info(streak)
            else:
                self._clear_analysis_info()

    def _find_streak_by_pid(self, pid_int):
        for species in (self.current_streaks or []):
            for bin in species:
                for streak in bin:
                    _, pid, *_ = streak
                    if pid == pid_int:
                        return streak
        return None

    def _display_streak_info(self, streak):
        (positions, pid, num_steps, theta_i, phi_i, theta_f, phi_f,
        theta0_vals, curr_vels, new_vels, energy_changes,
        start_pos, end_pos, init_en, final_en, delta_count, is_primary) = streak

        # Info box
        info = []
        info.append(f"PID: {self.sim.decode_pid(pid)}")
        info.append(f"Type: {'Primary' if (pid & ((1<<14)-1))==0 else 'Delta Ray'}")
        info.append(f"Steps: {num_steps}")
        info.append(f"Initial Position: {tuple(np.round(start_pos,3))}")
        info.append(f"Final Position:   {tuple(np.round(end_pos,3))}")
        info.append(f"Initial Energy:   {init_en:.3f} MeV")
        info.append(f"Final Energy:     {final_en:.3f} MeV")
        info.append(f"Initial θ:        {theta_i:.4f} rad ({np.degrees(theta_i):.2f}°)")
        info.append(f"Initial φ:        {phi_i:.4f} rad ({np.degrees(phi_i):.2f}°)")
        info.append(f"Final θ:          {theta_f:.4f} rad ({np.degrees(theta_f):.2f}°)")
        info.append(f"Final φ:          {phi_f:.4f} rad ({np.degrees(phi_f):.2f}°)")
        info.append(f"# Delta rays produced: {delta_count}")
        info.append(f"Number of recorded positions: {len(positions)}")
        self.analysis_info.config(state='normal')
        self.analysis_info.delete('1.0','end')
        self.analysis_info.insert('1.0', '\n'.join(info))
        self.analysis_info.config(state='disabled')

        # Table of positions
        self.positions_table.config(state='normal')
        self.positions_table.delete('1.0','end')
        header = f"{'Step':>4s} | {'X (μm)':>10s} | {'Y (μm)':>10s} | {'Z (μm)':>10s}\n"
        self.positions_table.insert('end', header)
        self.positions_table.insert('end', '-'*44+'\n')
        for i, (x,y,z) in enumerate(positions[:100]):  # limit to 100 for display
            self.positions_table.insert('end', f"{i:4d} | {x:10.3f} | {y:10.3f} | {z:10.3f}\n")
        if len(positions) > 100:
            self.positions_table.insert('end', f"... (truncated, total {len(positions)}) ...\n")
        self.positions_table.config(state='disabled')

        # Energy loss plot
        self.energy_ax.clear()
        dEs = [abs(de[0]) for de in energy_changes]
        if dEs:
            cumE = np.array([init_en] + list(np.array([init_en]) - np.cumsum(dEs)))
            self.energy_ax.plot(range(len(cumE)), cumE, '-o', ms=3)
            self.energy_ax.set_xlabel("Step")
            self.energy_ax.set_ylabel("Energy (MeV)")
            self.energy_ax.set_title("Cumulative Energy")
        else:
            self.energy_ax.set_title("No energy change data")
        self.energy_fig.tight_layout()
        self.energy_canvas.draw()

        # Angles plot
        self.angles_ax.clear()
        thetas = []
        phis = []
        # Extract angles from velocity vectors, or theta0_vals if available
        for v in new_vels:
            vx, vy, vz = v
            r = np.sqrt(vx**2 + vy**2 + vz**2)
            if r == 0: continue
            theta = np.arccos(np.clip(vz / r, -1, 1))
            phi = np.arctan2(vy, vx)
            thetas.append(theta)
            phis.append(phi)
        if thetas and phis:
            self.angles_ax.plot(thetas, label='θ (rad)')
            self.angles_ax.plot(phis, label='φ (rad)')
            self.angles_ax.set_xlabel("Step")
            self.angles_ax.set_ylabel("Angle (rad)")
            self.angles_ax.set_title("Trajectory Angles")
            self.angles_ax.legend()
        else:
            self.angles_ax.set_title("No angle data")
        self.angles_fig.tight_layout()
        self.angles_canvas.draw()

    def _clear_analysis_info(self):
        self.analysis_info.config(state='normal')
        self.analysis_info.delete('1.0','end')
        self.analysis_info.config(state='disabled')
        self.positions_table.config(state='normal')
        self.positions_table.delete('1.0','end')
        self.positions_table.config(state='disabled')
        self.energy_ax.clear()
        self.energy_ax.set_title("")
        self.energy_canvas.draw()
        self.angles_ax.clear()
        self.angles_ax.set_title("")
        self.angles_canvas.draw()

    def _update_heatmap(self, data):
        self.heatmap_ax.clear()
        # mask zeros so LogNorm and gray colormap don't show white for missing
        masked = np.ma.masked_equal(data, 0)
        norm = LogNorm(vmin=masked.min() if masked.min()>0 else 1, vmax=masked.max())
        im = self.heatmap_ax.imshow(
            masked,
            cmap='gray',
            origin='lower',
            norm=norm,
            extent=[0, self.sim.grid_size*self.sim.cell_size,
                    0, self.sim.grid_size*self.sim.cell_size]
        )
        im.cmap.set_bad(color='black')  # background for masked (zero) values
        if not hasattr(self, '_heat_cb'):
            self._heat_cb = self.heatmap_ax.figure.colorbar(im, ax=self.heatmap_ax)
        else:
            self._heat_cb.update_normal(im)

        # overlay trajectories
        legend_handles = {}
        for sp in self.current_streaks or []:
            for b in sp:
                for stk in b:
                    pos, pid, *_ = stk
                    if len(pos) < 2: continue
                    xs = [p[0] for p in pos]
                    ys = [p[1] for p in pos]
                    col = self.sim.get_particle_color(pid)
                    self.heatmap_ax.plot(xs, ys, '-', color=col, alpha=0.7)
                    idx = (pid >> (11+14)) & ((1<<7)-1)
                    lbl = self.sim.species_names.get(idx, f'Z={idx}')
                    if lbl not in legend_handles:
                        legend_handles[lbl] = Line2D([], [], color=col, label=lbl)
        if legend_handles:
            self.heatmap_ax.legend(
                handles=list(legend_handles.values()), title='Species',
                loc='upper right', fontsize='small', framealpha=0.5
            )
        self.heatmap_canvas.draw()

    def _export_positions_table(self):
        streak = self._get_current_analysis_streak()
        if not streak:
            messagebox.showwarning("No Data", "No streak data to export.")
            return
        positions = streak[0]
        fpath = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[('CSV files','*.csv')], title="Save positions as CSV")
        if not fpath:
            return
        import csv
        with open(fpath, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'X (um)', 'Y (um)', 'Z (um)'])
            for i, (x, y, z) in enumerate(positions):
                writer.writerow([i, x, y, z])
        messagebox.showinfo("Exported", f"Positions table saved to:\n{fpath}")

    def _export_energy_plot(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[('PNG Image','*.png')], title="Save energy plot as PNG")
        if not fpath:
            return
        self.energy_fig.savefig(fpath, dpi=150)
        messagebox.showinfo("Exported", f"Energy plot saved to:\n{fpath}")

    def _export_angles_plot(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[('PNG Image','*.png')], title="Save angles plot as PNG")
        if not fpath:
            return
        self.angles_fig.savefig(fpath, dpi=150)
        messagebox.showinfo("Exported", f"Angles plot saved to:\n{fpath}")

    def _get_current_analysis_streak(self):
        # Helper to get currently displayed streak (primary or delta)
        pid_str = self.selected_analysis_delta.get() or self.selected_analysis_primary.get()
        pid_int = None
        # Check in children first
        for k, v in getattr(self, 'analysis_delta_children', []):
            if v == pid_str:
                pid_int = k
                break
        # If not a child, check primaries
        if pid_int is None:
            for k, v in getattr(self, 'analysis_primaries_list', []):
                if v == pid_str:
                    pid_int = k
                    break
        if pid_int is not None:
            return self._find_streak_by_pid(pid_int)
        return None

    def _display_results(self):
        try:
            f = self.current_streaks[0][0][0]
            xs, ys, zs = zip(*f)
            self.traj_ax.clear()
            self.traj_ax.plot(xs, ys, zs, '-o', markersize=2)
            self.traj_canvas.draw()
        except:
            pass

    def pause_sim(self): pass
    def stop_sim(self): pass

if __name__ == "__main__":
    Application().mainloop()
