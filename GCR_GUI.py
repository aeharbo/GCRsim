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
        self.title("Cosmic Ray Simulator")
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
        vars_ = [(tk.IntVar, 1024, 'grid_size_var'),
                 (tk.DoubleVar, 1.0, 'dt_var'),
                 (tk.DoubleVar, 2025.0, 'date_var'),
                 (tk.IntVar, 4, 'max_workers_var')]
        for i, (label, (vtype, default, name)) in enumerate(zip(labels, vars_)):
            ttk.Label(params, text=label).grid(row=i, column=0, sticky='e')
            setattr(self, name, vtype(value=default))
            ttk.Entry(params, textvariable=getattr(self, name), width=10).grid(row=i, column=1)

        self.full_sim_var = tk.BooleanVar(value=False)
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
        self.primary_pid_combobox.bind('<<ComboboxSelected>>', lambda e: self._plot_3d_for_primary_pid())
        self.primary_pid_combobox.state(['disabled'])

        fig = Figure(figsize=(5,5))
        self.traj_ax = fig.add_subplot(111, projection='3d')
        self.traj_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.traj_canvas.get_tk_widget().pack(fill='both', expand=True)
        nav = NavigationToolbar2Tk(self.traj_canvas, frame)
        nav.update()
        nav.pack(side='bottom', fill='x')

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
            self._plot_3d_for_primary_pid()
        else:
            self.selected_primary_pid.set('')
            self.primary_pid_combobox.state(['disabled'])

    def _plot_3d_for_primary_pid(self):
        selection = self.selected_primary_pid.get()
        pid_int = None
        for human, pid in self.primary_pid_choices:
            if human == selection:
                pid_int = pid
                break
        if pid_int is None:
            self.traj_ax.clear()
            self.traj_canvas.draw()
            return
        species_idx = (pid_int >> (11+14)) & ((1<<7)-1)
        primary_idx = (pid_int >> 14) & ((1<<11)-1)
        parent_mask = ((species_idx, primary_idx))
        def is_child_or_self(pid):
            sp = (pid >> (11+14)) & ((1<<7)-1)
            pr = (pid >> 14) & ((1<<11)-1)
            return (sp, pr) == parent_mask
        streaks_to_plot = []
        for species in (self.current_streaks or []):
            for energy_bin in species:
                for streak in energy_bin:
                    positions, pid, *_ = streak
                    if is_child_or_self(pid):
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
        self.traj_ax.set_title(f"Trajectories for {selection}")
        self.traj_canvas.draw()

    def create_analysis_tab(self):
        frame = self.tab_analysis
        control_row = ttk.Frame(frame)
        control_row.pack(side='top', fill='x', padx=5, pady=3)

        # Dropdown for primaries
        ttk.Label(control_row, text="Primary PID:").pack(side='left')
        self.selected_analysis_primary = tk.StringVar()
        self.analysis_primary_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_analysis_primary, state='readonly'
        )
        self.analysis_primary_combobox.pack(side='left', padx=(0, 10))
        self.analysis_primary_combobox.state(['disabled'])
        self.analysis_primary_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_analysis_primary())

        # Dropdown for delta rays
        ttk.Label(control_row, text="Delta Ray PID:").pack(side='left')
        self.selected_analysis_delta = tk.StringVar()
        self.analysis_delta_combobox = ttk.Combobox(
            control_row, textvariable=self.selected_analysis_delta, state='readonly'
        )
        self.analysis_delta_combobox.pack(side='left')
        self.analysis_delta_combobox.state(['disabled'])
        self.analysis_delta_combobox.bind('<<ComboboxSelected>>', lambda e: self._update_analysis_delta())

        # Info display
        self.analysis_info = tk.Text(frame, height=15, width=80, wrap='word', font=('Consolas', 10))
        self.analysis_info.pack(fill='both', expand=True, padx=10, pady=8)
        self.analysis_info.config(state='disabled')


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

    def _clear_analysis_info(self):
        self.analysis_info.config(state='normal')
        self.analysis_info.delete('1.0','end')
        self.analysis_info.config(state='disabled')


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
