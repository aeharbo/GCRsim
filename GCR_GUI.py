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
        fig = Figure(figsize=(5,5))
        self.traj_ax = fig.add_subplot(111, projection='3d')
        self.traj_canvas = FigureCanvasTkAgg(fig, master=self.tab_3d)
        self.traj_canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_analysis_tab(self):
        ttk.Label(self.tab_analysis, text="Analysis tools coming soon...").pack(pady=10)

    def create_advanced_tab(self):
        ttk.Label(self.tab_advanced, text="Advanced configuration coming soon...").pack(pady=10)

    def create_log_tab(self):
        self.log_text = tk.Text(self.tab_log, wrap='none', height=10)
        self.log_text.pack(fill='both', expand=True)

    def new_sim(self):
        for attr in ('current_heatmap','current_streaks','current_count'):
            setattr(self, attr, None)
        self.heatmap_ax.clear(); self.heatmap_canvas.draw()
        self.traj_ax.clear();   self.traj_canvas.draw()
        self.log_text.delete('1.0','end')

    def load_sim(self):
        f = filedialog.askopenfilename(filetypes=[('HDF5 files','*.h5')])
        if not f: return
        hm, st, ct = CosmicRaySimulation.load_sim(f)
        self.current_heatmap, self.current_streaks, self.current_count = hm,st,ct
        self._update_heatmap(hm)

    def save_sim(self):
        if self.current_heatmap is None:
            messagebox.showwarning('No Data','Run or load a simulation first.')
            return
        f = filedialog.asksaveasfilename(defaultextension='.h5',filetypes=[('HDF5 files','*.h5')])
        if not f: return
        self.sim.save_sim(self.current_heatmap,self.current_streaks,self.current_count,f)
        messagebox.showinfo('Saved',f'Simulation saved to {f}')

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
