import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import numpy as np
import threading
from GCRsim_v02f import CosmicRaySimulation

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GCRsim(alpha-build)")
        self.geometry("900x600")

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

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Simulation", command=self.new_sim)
        filemenu.add_command(label="Load Simulation", command=self.load_sim)
        filemenu.add_command(label="Save Simulation", command=self.save_sim)
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

        tabs = [(self.tab_control, "Simulation"),
                (self.tab_heatmap, "Heatmap"),
                (self.tab_3d, "3D Trajectory"),
                (self.tab_analysis, "Analysis"),
                (self.tab_advanced, "Advanced Config"),
                (self.tab_log, "Log")]
        for tab, text in tabs:
            self.notebook.add(tab, text=text)

        self.create_control_tab()
        self.create_heatmap_tab()
        self.create_3d_tab()
        self.create_analysis_tab()
        self.create_advanced_tab()
        self.create_log_tab()

    def create_control_tab(self):
        frame = self.tab_control
        params = ttk.LabelFrame(frame, text="Simulation Parameters")
        params.pack(side='left', fill='y', padx=5, pady=5)

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

        btns = ttk.Frame(frame)
        btns.pack(side='top', fill='x', padx=5, pady=5)
        self.run_button = ttk.Button(btns, text="Run", command=self.run_sim)
        self.run_button.pack(side='left')
        ttk.Button(btns, text="Pause", command=self.pause_sim).pack(side='left')
        ttk.Button(btns, text="Stop", command=self.stop_sim).pack(side='left')
        self.progress = ttk.Progressbar(btns, orient='horizontal', length=200)
        self.progress.pack(side='left', padx=5)

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
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
                        from matplotlib.lines import Line2D
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
