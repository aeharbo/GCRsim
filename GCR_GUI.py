import os
import csv
import h5py
import threading
import itertools
import numpy as np
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from GCRsim_v02f import CosmicRaySimulation
from electron_spread import process_electrons_to_DN

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
        self.create_histograms_tab()
        self.create_menu()
        self._ensure_sim()
        self.plot_wolf_number()
        self.current_dnmap = None

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
        filemenu.add_command(label="Convert CSV to DN Map", command=self.convert_to_dn_map)
        filemenu.add_command(label="Load DN Map", command=self.load_dnmap)
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
        self.tab_dnmap = ttk.Frame(self.notebook)
        self.tab_histogram = ttk.Frame(self.notebook)
        self.tab_advanced = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)
        self.tab_movie = ttk.Frame(self.notebook)
        tabs = [
            (self.tab_control, "Simulation"),
            (self.tab_heatmap, "Heatmap"),
            (self.tab_3d, "3D Trajectory"),
            (self.tab_movie, "Movie Mode"),    
            (self.tab_analysis, "Analysis"),
            (self.tab_dnmap, "Pixelated SCA"),
            (self.tab_histogram, "Histograms"),
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
        self.create_dnmap_tab()
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
        ttk.Button(btns, text="Export Energy Depositions", command=self._export_energy_depositions).pack(side='left', padx=(0, 6))
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

        self.flux_fig = Figure(figsize=(5,2), tight_layout=True)
        self.flux_ax = self.flux_fig.add_subplot(111)
        self.flux_canvas = FigureCanvasTkAgg(self.flux_fig, master=frame)
        self.flux_canvas.get_tk_widget().pack(fill='x', padx=10, pady=(10, 5))
        self.flux_nav = NavigationToolbar2Tk(self.flux_canvas, frame)
        self.flux_nav.update()
        self.flux_nav.pack(side='top', fill='x', padx=10, pady=(0, 3))
        # --- Add Wolf Number Plot below GCR Flux Forecast ---
        self.wolf_fig = Figure(figsize=(5, 2), tight_layout=True)
        self.wolf_ax = self.wolf_fig.add_subplot(111)
        self.wolf_canvas = FigureCanvasTkAgg(self.wolf_fig, master=frame)
        self.wolf_canvas.get_tk_widget().pack(fill='x', padx=10, pady=(0, 8))
        self.wolf_nav = NavigationToolbar2Tk(self.wolf_canvas, frame)
        self.wolf_nav.update()
        self.wolf_nav.pack(side='top', fill='x', padx=10, pady=(0, 3))

    def create_heatmap_tab(self):
        fig = Figure(figsize=(5,5))
        self.heatmap_ax = fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        self.heatmap_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.heatmap_canvas, self.tab_heatmap)
        nav.update()
        nav.pack(side='bottom', fill='x')
        self.heatmap_canvas.mpl_connect("button_press_event", self._on_heatmap_click)

    def _on_heatmap_click(self, event):
        """Handle click events on the heatmap to select and display info about a primary streak."""
        if event.inaxes != self.heatmap_ax or not self.current_streaks:
            return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        min_dist = float("inf")
        nearest_streak = None
        nearest_pid = None

        # Find the closest primary streak (delta_idx == 0)
        for sp in self.current_streaks:
            for b in sp:
                for streak in b:
                    positions, pid, *rest = streak
                    delta_idx = pid & ((1<<14)-1)
                    if delta_idx != 0 or len(positions) < 2:
                        continue  # Only primary streaks, skip single-point
                    # Find the nearest segment to the click
                    for (x0, y0, _), (x1, y1, _) in zip(positions[:-1], positions[1:]):
                        # Distance from point to segment
                        px, py = x_click, y_click
                        dx, dy = x1 - x0, y1 - y0
                        if dx == dy == 0:
                            dist = np.hypot(px - x0, py - y0)
                        else:
                            t = max(0, min(1, ((px - x0)*dx + (py - y0)*dy) / (dx*dx + dy*dy)))
                            proj_x, proj_y = x0 + t*dx, y0 + t*dy
                            dist = np.hypot(px - proj_x, py - proj_y)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_streak = streak
                            nearest_pid = pid

        # Pick a distance threshold: e.g., 25 microns (adjust to your cell size)
        threshold = 25  # microns
        if nearest_streak and min_dist < threshold:
            # Display the info (e.g., with a popup or a dedicated panel)
            info = self._format_primary_info(nearest_streak)
            messagebox.showinfo("Primary Info", info)
        # else: click missed all primaries; do nothing or clear panel

    def _format_primary_info(self, streak):
        positions, pid, num_steps, theta_i, phi_i, theta_f, phi_f, \
        theta0_vals, curr_vels, new_vels, energy_changes, \
        start_pos, end_pos, init_en, final_en, delta_count, is_primary = streak

        lines = [
            f"PID: {self.sim.decode_pid(pid)}",
            f"Steps: {num_steps}",
            f"Initial Position: {tuple(np.round(start_pos,3))}",
            f"Final Position:   {tuple(np.round(end_pos,3))}",
            f"Initial Energy:   {init_en:.3f} MeV",
            f"Final Energy:     {final_en:.3f} MeV",
            f"Initial θ:        {theta_i:.4f} rad ({np.degrees(theta_i):.2f}°)",
            f"Initial φ:        {phi_i:.4f} rad ({np.degrees(phi_i):.2f}°)",
            f"δ rays produced: {delta_count}",
        ]
        return "\n".join(lines)

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
        
    def create_dnmap_tab(self):
        self.dnmap_fig = Figure(figsize=(5, 5))
        self.dnmap_ax = self.dnmap_fig.add_subplot(111)
        self.dnmap_canvas = FigureCanvasTkAgg(self.dnmap_fig, master=self.tab_dnmap)
        self.dnmap_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.dnmap_canvas, self.tab_dnmap)
        nav.update()
        nav.pack(side='bottom', fill='x')
        btn_frame = ttk.Frame(self.tab_dnmap)
        btn_frame.pack(side='top', fill='x', padx=5, pady=5)
        ttk.Button(btn_frame, text="Export DN Map as NPY", command=self.export_dnmap_npy).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Export DN Map as PNG", command=self.export_dnmap_png).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Load DN Map", command=self.load_dnmap).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Highlight Affected Pixels", command=self.highlight_dn_pixels).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Reset View", command=self._update_dnmap).pack(side='left', padx=2)
        self.dnmap_canvas.mpl_connect("button_press_event", self._on_dnmap_click)

    def export_dnmap_npy(self):
        if self.current_dnmap is None:
            messagebox.showwarning("No Data", "No DN map to export.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".npy",
                filetypes=[('NumPy Array', '*.npy')], title="Save DN map as NPY")
        if not fpath: return
        np.save(fpath, self.current_dnmap)
        messagebox.showinfo("Exported", f"DN map saved to:\n{fpath}")

    def export_dnmap_png(self):
        if self.current_dnmap is None:
            messagebox.showwarning("No Data", "No DN map to export.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".png",
                filetypes=[('PNG Image','*.png')], title="Save DN map as PNG")
        if not fpath: return
        self.dnmap_fig.savefig(fpath, dpi=150)
        messagebox.showinfo("Exported", f"DN map saved to:\n{fpath}")

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

    def load_dnmap(self):
        fpath = filedialog.askopenfilename(
            filetypes=[('NumPy Array', '*.npy')],
            title="Load DN Map (.npy)"
        )
        if fpath:
            try:
                dn_map = np.load(fpath)
                self.current_dnmap = dn_map
                self._update_dnmap()
                self.notebook.select(self.tab_dnmap)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load DN map:\n{e}")

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

    def _on_dnmap_click(self, event):
        # Only handle double-click (ignore single clicks for pop-up)
        if not hasattr(event, 'dblclick') or not event.dblclick:
            return  # Only handle double-clicks
        
        if self.current_streaks is None or not self.current_streaks:
            messagebox.showinfo("No Data", "No simulation streaks are loaded. Run or load a simulation first.")
            return
        
        if event.inaxes != self.dnmap_ax or self.current_dnmap is None:
            return

        x_pix, y_pix = event.xdata, event.ydata
        if x_pix is None or y_pix is None:
            return

        grid_size = self.current_dnmap.shape[0]
        pixel_size_lo = 10.0  # or your actual low-res pixel size
        x_um = x_pix * pixel_size_lo
        y_um = y_pix * pixel_size_lo

        closest_pid, delta_pids, x_parent, y_parent = self._find_nearest_parent_pid(x_um, y_um)
        if closest_pid is None:
            messagebox.showinfo("No blob found", "No primary event found near this location.")
            return

        csvfile = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select Energy Loss CSV for High-Res View"
        )
        if not csvfile:
            return

        from electron_spread import process_pid_electrons_zoom
        H_zoom = process_pid_electrons_zoom(
            csvfile=csvfile,
            pid=closest_pid,
            delta_pids=delta_pids,
            x_center=x_parent,
            y_center=y_parent,
            region_size_um=20.0,     # Or whatever region size you want
            pixel_size_hi=0.1,
            kernel_size_hi=50,
            sigma=0.314
        )
        self._popup_zoom_dn_image(H_zoom, x_parent, y_parent, closest_pid)

    def _find_nearest_parent_pid(self, x_um, y_um, max_dist_um=50):
        """
        Finds the parent PID whose track passes closest to (x_um, y_um).
        Returns (parent_pid, list_of_delta_pids, nearest_x, nearest_y)
        """
        min_dist = float("inf")
        parent_pid = None
        parent_x = None
        parent_y = None
        # Search for the nearest primary (delta==0) streak
        for sp in (self.current_streaks or []):
            for b in sp:
                for streak in b:
                    positions, pid, *rest = streak
                    delta_idx = pid & ((1<<14)-1)
                    if delta_idx != 0 or len(positions) < 1:
                        continue  # Skip deltas
                    for (x, y, *_) in positions:
                        dist = np.hypot(x - x_um, y - y_um)
                        if dist < min_dist:
                            min_dist = dist
                            parent_pid = pid
                            parent_x = x
                            parent_y = y
        if parent_pid is None or min_dist > max_dist_um:
            return None, [], None, None
        # Find all delta PIDs for this parent
        delta_pids = []
        for sp in (self.current_streaks or []):
            for b in sp:
                for streak in b:
                    _, pid, *rest = streak
                    if is_delta_of_primary(pid, parent_pid):
                        delta_pids.append(pid)
        return parent_pid, delta_pids, parent_x, parent_y

    def _popup_zoom_dn_image(self, H_zoom, x_um, y_um, pid):
        """Shows the high-res zoom DN image in a pop-up matplotlib window."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            H_zoom,
            origin='lower',
            cmap='gray',
            interpolation='nearest',
            extent=[
                x_um - H_zoom.shape[1]/2 * 0.1,
                x_um + H_zoom.shape[1]/2 * 0.1,
                y_um - H_zoom.shape[0]/2 * 0.1,
                y_um + H_zoom.shape[0]/2 * 0.1,
            ]
        )
        ax.set_title(f"High-Res DN: PID {self.sim.decode_pid(pid)}\n(center={x_um:.1f}, {y_um:.1f} µm)")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=ax, label="Electrons (arbitrary units)")
        plt.show()

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
        
    def _update_dnmap(self):
        # Remove circles/patches and colorbar
        if hasattr(self, '_dnmap_cb') and self._dnmap_cb:
            try:
                self._dnmap_cb.remove()
            except Exception:
                pass
            self._dnmap_cb = None
        self.dnmap_ax.clear()
        
        if self.current_dnmap is None:
            self.dnmap_canvas.draw()
            return

        masked = np.ma.masked_invalid(self.current_dnmap)
        norm = LogNorm(
            vmin=np.nanmin(masked[masked > 0]) if np.any(masked > 0) else 1,
            vmax=np.nanmax(masked)
        )
        im = self.dnmap_ax.imshow(
            masked,
            cmap='gray',
            origin='lower',
            norm=norm
        )
        im.cmap.set_bad(color='black')
        self.dnmap_ax.set_title("Pixelated SCA image")
        self.dnmap_ax.set_xlabel("x (pixels)")
        self.dnmap_ax.set_ylabel("y (pixels)")
        self._dnmap_cb = self.dnmap_ax.figure.colorbar(im, ax=self.dnmap_ax)
        self._dnmap_cb.set_label("Digital Number (DN)")
        self.dnmap_canvas.draw()

    def highlight_dn_pixels(self):
        if hasattr(self, '_dnmap_cb') and self._dnmap_cb:
            try:
                self._dnmap_cb.remove()
            except Exception:
                pass
            self._dnmap_cb = None
        self.dnmap_ax.clear()

        if self.current_dnmap is None:
            self.dnmap_canvas.draw()
            return

        masked = np.ma.masked_invalid(self.current_dnmap)
        grid_size = self.current_dnmap.shape[0]
        extent = [0, grid_size, 0, grid_size ]
        norm = LogNorm(
            vmin=np.nanmin(masked[masked > 0]) if np.any(masked > 0) else 1,
            vmax=np.nanmax(masked)
        )
        im = self.dnmap_ax.imshow(
            masked,
            cmap='gray',
            origin='lower',
            norm=norm,
            extent=extent
        )
        im.cmap.set_bad(color='black')
        self.dnmap_ax.set_xlabel("x (pixels)")
        self.dnmap_ax.set_ylabel("y (pixels)")
        self.dnmap_ax.set_title("Pixelated SCA image")
        self._dnmap_cb = self.dnmap_ax.figure.colorbar(im, ax=self.dnmap_ax)
        self._dnmap_cb.set_label("Digital Number (DN)")

        # Add circles around affected pixels
        affected = np.argwhere(masked > 0)
        for y_pix, x_pix in affected:
            x_center = (x_pix + 0.5)
            y_center = (y_pix + 0.5)
            circle = Circle(
                (x_center, y_center),
                radius=0.5,
                edgecolor='limegreen',
                facecolor='none',
                linewidth=1.4
            )
            self.dnmap_ax.add_patch(circle)
        self.dnmap_canvas.draw()

    def convert_to_dn_map(self):
        # Prompt for input CSV
        csvfile = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            title="Select Energy Loss CSV"
        )
        if not csvfile:
            return

        # Prompt for gain map txt
        gain_txt = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")],
            title="Select Gain Map TXT"
        )
        if not gain_txt:
            return

        # Prompt for output .npy path
        dn_output = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy")],
            title="Save Output DN Map As"
        )
        if not dn_output:
            return

        # Run in a thread (to avoid freezing GUI)
        def do_conversion():
            try:
                # Optionally, show a progress dialog, or disable UI
                H_detector_DN = process_electrons_to_DN(
                    csvfile=csvfile,
                    gain_txt=gain_txt,
                    output_DN_path=dn_output
                )
                self.after(0, lambda: messagebox.showinfo("Success", f"DN Map saved to:\n{dn_output}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Conversion failed:\n{str(e)}"))
            self.current_dnmap = H_detector_DN
            self.after(0, self._update_dnmap)

        threading.Thread(target=do_conversion, daemon=True).start()

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
        self._populate_hist_species_dropdown()
        self.update_histogram()        
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
        self._populate_hist_species_dropdown()
        self.update_histogram()
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
            h, s, c = self.sim.run_sim(species_index=self.sim.species_index)
            self.current_heatmap, self.current_streaks, self.current_count = h, [s], c
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
        self._populate_hist_species_dropdown()
        self.update_histogram()
        
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
        self.heatmap_ax.set_title("GCR and δ ray tracks")
        self.heatmap_ax.set_xlabel("x (μm)")
        self.heatmap_ax.set_ylabel("y (μm)")
        self._heat_cb.set_label("Number of propagation events")    
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
        
        
    def _export_energy_depositions(self):
        # Check that you have streaks to export
        if self.current_streaks is None or not self.current_streaks:
            messagebox.showwarning("No Data", "No simulation data to export.")
            return

        # Compose default file name
        current_date = datetime.now()
        computer_friendly_date = current_date.strftime("%Y%m%d%H%M")
        default_file_name = computer_friendly_date + '_energy_loss.csv'

        # Ask where to save
        fpath = filedialog.asksaveasfilename(
            initialfile=default_file_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Energy Deposition Data"
        )
        if not fpath:
            return

        try:
            self._ensure_sim()  # Just in case
            # Call the CosmicRaySimulation method you fixed earlier
            self.sim.build_energy_loss_csv(self.current_streaks, fpath)
            messagebox.showinfo("Exported", f"Energy deposition data saved to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export energy depositions:\n{str(e)}")
            
        # After exporting the CSV
        if messagebox.askyesno("Convert", "CSV saved. Do you want to convert to a DN map now?"):
            self.convert_to_dn_map()

    def plot_wolf_number(self):
        df = self.sim.historic_df
        if df is None or df.empty:
            self.wolf_ax.clear()
            self.wolf_ax.set_title("No Wolf Number data loaded")
            self.wolf_canvas.draw()
            return

        cycle_max = df.loc[df.groupby("solar_cycle")["mean"].idxmax()]
        cycle_min = df.loc[df.groupby("solar_cycle")["mean"].idxmin()]

        self.wolf_ax.clear()
        self.wolf_ax.plot(df['date'], df['mean'], marker='o', linestyle='-', color='blue', alpha=0.25, label='Mean Wolf Number')

        cycle_change_dates = [1996.624, 2008.958, 2019.958]
        cycle_labels = ['Cycle 23 starts', 'Cycle 24 starts', 'Cycle 25 starts']
        ymin, ymax = self.wolf_ax.get_ylim()
        for x_val, label in zip(cycle_change_dates, cycle_labels):
            self.wolf_ax.axvline(x=x_val, color='red', linestyle='--', linewidth=2)
            self.wolf_ax.text(x_val, ymax - 0.05 * (ymax - ymin), label, rotation=90, color='red',
                            verticalalignment='top', horizontalalignment='right')

        self.wolf_ax.scatter(cycle_max['date'], cycle_max['mean'], color='green', s=60, marker='*', label='Cycle Max')
        self.wolf_ax.scatter(cycle_min['date'], cycle_min['mean'], color='orange', s=60, marker='D', label='Cycle Min')

        self.wolf_ax.set_xlabel('Date')
        self.wolf_ax.set_ylabel('Mean Wolf Number')
        self.wolf_ax.set_title('Wolf Number (Sunspot Number) History')
        self.wolf_ax.legend()
        self.wolf_ax.grid(True)
        self.wolf_fig.tight_layout()
        self.wolf_canvas.draw()

    def create_histograms_tab(self):
        frame = self.tab_histogram
        control_row = ttk.Frame(frame)
        control_row.pack(side='top', fill='x', padx=5, pady=3)

        # Controls for histogram selection
        controls = ttk.Frame(frame)
        controls.pack(side='top', fill='x', padx=6, pady=4)

        ttk.Label(controls, text="Histogram:").pack(side='left')
        self.histogram_type = tk.StringVar(value="Delta Ray Energies")
        self.histogram_options = ["Delta Ray Energies", "Primary Energy Distribution"]
        self.histogram_combobox = ttk.Combobox(
            controls, textvariable=self.histogram_type, state='readonly',
            values=self.histogram_options, width=28
        )
        
        self.hist_species_var = tk.StringVar()

        self.histogram_combobox.pack(side='left', padx=(5,10))
        self.histogram_combobox.bind('<<ComboboxSelected>>', lambda e: self.update_histogram())

        # If you want to select species for primary energies
        ttk.Label(controls, text="Species:").pack(side='left', padx=(10,0))
        self.hist_species_combobox = ttk.Combobox(
            controls, textvariable=self.hist_species_var, state='readonly', width=18
        )
        self.hist_species_combobox.pack(side='left')
        self.hist_species_combobox.bind('<<ComboboxSelected>>', lambda e: self.update_histogram())
        self.hist_species_combobox.state(['disabled'])
        ttk.Button(controls, text="Export as PNG", command=self.export_histogram).pack(side='left', padx=(20,0))

        # Matplotlib Figure for Histogram
        self.hist_fig = Figure(figsize=(5, 4))
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=frame)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Initialize species dropdown
        self._populate_hist_species_dropdown()
        self.update_histogram()

    def _populate_hist_species_dropdown(self):
        if not self.sim or not self.current_streaks:
            self.hist_species_combobox['values'] = []
            self.hist_species_var.set('')
            self.hist_species_combobox.state(['disabled'])
            return

        available_species_indices = [
            i for i, streak in enumerate(self.current_streaks)
            if streak and len(streak) > 0
        ]
        species_list = [self.sim.species_names[k] for k in available_species_indices]
        self.hist_species_combobox['values'] = species_list
        cur = self.hist_species_var.get()
        if not cur or cur not in species_list:
            if species_list:
                self.hist_species_var.set(species_list[0])
            else:
                self.hist_species_var.set('')

    def update_histogram(self):
        self.hist_ax.clear()
        hist_type = self.histogram_type.get()

        if hist_type == "Delta Ray Energies":
            self.hist_species_combobox.state(['disabled'])
            self.hist_species_var.set("All species")
            self.plot_delta_ray_energy_histogram()
        elif hist_type == "Primary Energy Distribution":
            self._populate_hist_species_dropdown()
            species_list = self.hist_species_combobox['values']
            # Only enable if there is a valid species selection
            if species_list:
                self.hist_species_combobox.state(['!disabled'])
            else:
                self.hist_species_combobox.state(['disabled'])
            self.plot_primary_energy_histogram()

        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

    def plot_delta_ray_energy_histogram(self):
        streaks = self.current_streaks
        if not streaks:
            self.hist_ax.set_title("No data loaded")
            return
        delta_mask = (1 << 14) - 1
        flat_streaks = list(itertools.chain.from_iterable(itertools.chain.from_iterable(streaks)))
        delta_ray_energies = [st[13] for st in flat_streaks if (st[1] & delta_mask) > 0]
        if not delta_ray_energies:
            self.hist_ax.set_title("No delta rays found")
            return

        bins = np.logspace(np.log10(min(delta_ray_energies)), np.log10(max(delta_ray_energies)), 100)
        hist_vals, bin_edges = np.histogram(delta_ray_energies, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        me = 0.511  # MeV
        beta2 = CosmicRaySimulation.beta(bin_centers, me)**2
        beta2 = np.clip(beta2, 1e-8, 1.0)
        K = 0.307075   # MeV*cm^2/g 
        Z = 58.88         # our weighted material average Z
        A = 144.47      # our weighted material average A
        z = -1.0       # electron charge in units of e
        prefactor = (K/2) * (Z/A) * (z**2)
        fit = prefactor / (bin_centers**2 * beta2)
        fit_scaled = fit * (hist_vals.max() / fit.max())

        self.hist_ax.hist(delta_ray_energies, bins=bins, color='orange', alpha=0.7, edgecolor='black', label="Delta Rays")
        self.hist_ax.plot(bin_centers, fit_scaled, 'k--', label=r'$\propto [T^2\beta^2]^{-1}$')
        self.hist_ax.set_xscale('log')
        self.hist_ax.set_yscale('log')
        self.hist_ax.set_xlabel(r"Initial $\delta$ ray energy (MeV)")
        self.hist_ax.set_ylabel(r"Count")
        self.hist_ax.set_title(r"Histogram of $\delta$ ray initial energies")
        self.hist_ax.legend()
        self.hist_ax.grid(True, which="both", ls="--", alpha=0.7)

    def plot_primary_energy_histogram(self):
        streaks = self.current_streaks
        if not streaks:
            self.hist_ax.set_title("No data loaded")
            return
        selected_species = self.hist_species_var.get()
        if not selected_species:
            self.hist_ax.set_title("No species selected")
            return

        # Map dropdown directly to streaks index
        species_list = self.hist_species_combobox['values']
        try:
            streaks_idx = species_list.index(selected_species)
        except ValueError:
            self.hist_ax.set_title("Species not found in current selection")
            return

        flat_streaks = list(itertools.chain.from_iterable(streaks[streaks_idx]))
        primaries = [st for st in flat_streaks if (st[1] & ((1<<14)-1)) == 0]
        primary_energies = [st[13] for st in primaries]
        if not primary_energies:
            self.hist_ax.set_title("No primaries found")
            return
        bins = np.logspace(np.log10(min(primary_energies)), np.log10(max(primary_energies)), 100)
        self.hist_ax.hist(primary_energies, bins=bins, color='royalblue', alpha=0.8, edgecolor='black')
        self.hist_ax.set_xscale('log')
        self.hist_ax.set_xlabel("Primary Initial Energy (MeV)")
        self.hist_ax.set_yscale('log')
        self.hist_ax.set_ylabel("Count")
        self.hist_ax.set_title(f"Primary Population vs. Initial Energy\nSpecies: {selected_species}")
        self.hist_ax.grid(True, which="both", ls="--", alpha=0.7)

    def export_histogram(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[('PNG Image','*.png')], title="Save histogram as PNG")
        if not fpath:
            return
        self.hist_fig.savefig(fpath, dpi=150)
        messagebox.showinfo("Exported", f"Histogram plot saved to:\n{fpath}")
            
    def create_advanced_tab(self):
        frame = self.tab_advanced
        # Clear frame if called multiple times
        for w in frame.winfo_children():
            w.destroy()

        params = ttk.LabelFrame(frame, text="Advanced Simulation Parameters")
        params.pack(side='top', fill='x', padx=10, pady=10)

        self._ensure_sim()
        species_names = [self.sim.species_names[k] for k in sorted(self.sim.species_names.keys())]
        ttk.Label(params, text="Species:").grid(row=0, column=0, sticky='e')
        self.adv_species_var = tk.StringVar()
        self.adv_species_combobox = ttk.Combobox(params, textvariable=self.adv_species_var, state='readonly')
        self.adv_species_combobox['values'] = species_names
        self.adv_species_combobox.grid(row=0, column=1, padx=4, pady=2)
        if species_names:
            self.adv_species_var.set(species_names[0])

        # Grid size
        ttk.Label(params, text="Grid Size:").grid(row=1, column=0, sticky='e')
        self.adv_grid_size_var = tk.IntVar(value=4088)
        ttk.Entry(params, textvariable=self.adv_grid_size_var, width=10).grid(row=1, column=1, padx=4, pady=2)

        # dt
        ttk.Label(params, text="Exposure Time (dt):").grid(row=2, column=0, sticky='e')
        self.adv_dt_var = tk.DoubleVar(value=3.4)
        ttk.Entry(params, textvariable=self.adv_dt_var, width=10).grid(row=2, column=1, padx=4, pady=2)

        # date
        ttk.Label(params, text="Date (fractional year):").grid(row=3, column=0, sticky='e')
        self.adv_date_var = tk.DoubleVar(value=2026.123)
        ttk.Entry(params, textvariable=self.adv_date_var, width=10).grid(row=3, column=1, padx=4, pady=2)

        # max_workers
        ttk.Label(params, text="Max Workers:").grid(row=4, column=0, sticky='e')
        self.adv_max_workers_var = tk.IntVar(value=4)
        ttk.Entry(params, textvariable=self.adv_max_workers_var, width=10).grid(row=4, column=1, padx=4, pady=2)

        # Progress Bar
        self.adv_progress = ttk.Progressbar(params, orient='horizontal', length=200, mode='determinate')
        self.adv_progress.grid(row=5, columnspan=2, pady=8)
        self.adv_progress['maximum'] = 100  # Set default; will update during run

        # Dedicated Run button
        self.adv_run_btn = ttk.Button(frame, text="Run Simulation", command=self.run_advanced_sim)
        self.adv_run_btn.pack(pady=15)

    def run_advanced_sim(self):
        species_index = self._get_selected_advanced_species_index()
        if species_index is None:
            messagebox.showerror("Error", "Please select a valid species.")
            return

        self.adv_progress.config(mode='indeterminate')
        self.adv_progress.start(10)
        self.adv_run_btn.config(state='disabled')

        # Pass species_index directly!
        threading.Thread(target=self._run_advanced_sim_thread, args=(species_index,), daemon=True).start()

    def _run_advanced_sim_thread(self, species_index):
        if species_index is None:
            self.after(0, lambda: messagebox.showerror("Error", "Please select a valid species."))
            self.after(0, lambda: self.adv_run_btn.config(state='normal'))
            return

        # Get simulation parameters from UI
        grid = self.adv_grid_size_var.get()
        dt = self.adv_dt_var.get()
        date = self.adv_date_var.get()
        maxw = self.adv_max_workers_var.get()
        self.sim = CosmicRaySimulation(
            species_index=species_index, grid_size=grid, dt=dt, date=date, progress_bar=True, max_workers=maxw
        )

        # Start the progress bar in indeterminate mode
        self.after(0, lambda: self.adv_progress.config(mode='indeterminate'))
        self.after(0, self.adv_progress.start)

        # Run the simulation
        h, s, c = self.sim.run_sim(species_index=species_index)
        self.current_heatmap, self.current_streaks, self.current_count = h, [s], c

        # After sim finishes...
        self.after(0, self.adv_progress.stop)
        self.after(0, lambda: self.adv_progress.config(mode='determinate', value=100, style="green.Horizontal.TProgressbar"))
        self.after(0, lambda: self.adv_run_btn.config(state='normal'))
        self.after(0, self._update_heatmap, h)
        self.after(0, self._display_results)
        self.after(0, self._populate_primary_pid_dropdown)
        self.after(0, self._populate_analysis_primary_dropdown)
        self.after(0, self._populate_movie_primary_dropdown)
        self._populate_hist_species_dropdown()
        self.update_histogram()

    def _get_selected_advanced_species_index(self):
        species_name = self.adv_species_var.get()
        for k, v in self.sim.species_names.items():
            if v == species_name:
                return k
        return None

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

if __name__ == "__main__":
    Application().mainloop()
