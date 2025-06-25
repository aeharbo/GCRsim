import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from GCRsim_v02f import CosmicRaySimulation
import threading

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cosmic Ray Simulator")
        self.geometry("1000x700")

        # style for completed progress bar
        self.style = ttk.Style(self)
        # ensure a known theme
        try:
            self.style.theme_use('default')
        except Exception:
            pass
        self.style.configure("green.Horizontal.TProgressbar", background='green')

        # Simulator and results placeholders
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

        # Tabs
        self.tab_control = ttk.Frame(self.notebook)
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_3d = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_advanced = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_control, text="Simulation")
        self.notebook.add(self.tab_heatmap, text="Heatmap")
        self.notebook.add(self.tab_3d, text="3D Trajectory")
        self.notebook.add(self.tab_analysis, text="Analysis")
        self.notebook.add(self.tab_advanced, text="Advanced Config")
        self.notebook.add(self.tab_log, text="Log")

        self.create_control_tab()
        self.create_heatmap_tab()
        self.create_3d_tab()
        self.create_analysis_tab()
        self.create_advanced_tab()
        self.create_log_tab()

    def create_control_tab(self):
        frame = self.tab_control
        params_frame = ttk.LabelFrame(frame, text="Simulation Parameters")
        params_frame.pack(side='left', fill='y', padx=5, pady=5)

        # Grid size
        ttk.Label(params_frame, text="Grid Size:").grid(row=0, column=0, sticky='e')
        self.grid_size_var = tk.IntVar(value=1024)
        ttk.Entry(params_frame, textvariable=self.grid_size_var, width=10).grid(row=0, column=1)

        # Exposure time dt
        ttk.Label(params_frame, text="Exposure Time (dt):").grid(row=1, column=0, sticky='e')
        self.dt_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.dt_var, width=10).grid(row=1, column=1)

        # Date
        ttk.Label(params_frame, text="Date (fractional year):").grid(row=2, column=0, sticky='e')
        self.date_var = tk.DoubleVar(value=2025.0)
        ttk.Entry(params_frame, textvariable=self.date_var, width=10).grid(row=2, column=1)

        # Max workers
        ttk.Label(params_frame, text="Max Workers:").grid(row=3, column=0, sticky='e')
        self.max_workers_var = tk.IntVar(value=4)
        ttk.Entry(params_frame, textvariable=self.max_workers_var, width=10).grid(row=3, column=1)

        # Full simulation checkbox
        self.full_sim_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Run all species", variable=self.full_sim_var).grid(row=4, columnspan=2, pady=5)

        # Control buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side='top', fill='x', padx=5, pady=5)
        self.run_button = ttk.Button(btn_frame, text="Run", command=self.run_sim)
        self.run_button.pack(side='left')
        ttk.Button(btn_frame, text="Pause", command=self.pause_sim).pack(side='left')
        ttk.Button(btn_frame, text="Stop", command=self.stop_sim).pack(side='left')
        self.progress = ttk.Progressbar(btn_frame, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(side='left', padx=5)

    def create_heatmap_tab(self):
        fig = Figure(figsize=(5,5))
        self.heatmap_ax = fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        self.heatmap_canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_3d_tab(self):
        fig = Figure(figsize=(5,5))
        self.traj_ax = fig.add_subplot(111, projection='3d')
        self.traj_canvas = FigureCanvasTkAgg(fig, master=self.tab_3d)
        self.traj_canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_analysis_tab(self):
        ttk.Label(self.tab_analysis, text="Analysis tools coming soon...").pack(padx=10, pady=10)

    def create_advanced_tab(self):
        ttk.Label(self.tab_advanced, text="Advanced configuration coming soon...").pack(padx=10, pady=10)

    def create_log_tab(self):
        self.log_text = tk.Text(self.tab_log, wrap='none', height=10)
        self.log_text.pack(fill='both', expand=True)

    def new_sim(self):
        # Clear previous results
        self.current_heatmap = None
        self.current_streaks = None
        self.current_count = None
        self.heatmap_ax.clear()
        self.heatmap_canvas.draw()
        self.traj_ax.clear()
        self.traj_canvas.draw()
        self.log_text.delete('1.0', tk.END)

    def load_sim(self):
        fname = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5"), ("All files", "*")])
        if not fname:
            return
        try:
            heatmap, streaks, count = CosmicRaySimulation.load_sim(fname)
            self.current_heatmap = heatmap
            self.current_streaks = streaks
            self.current_count = count
            # display heatmap
            self.heatmap_ax.clear()
            self.heatmap_ax.imshow(heatmap, cmap='inferno', origin='lower')
            self.heatmap_canvas.draw()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def save_sim(self):
        if self.current_heatmap is None:
            messagebox.showwarning("No Data", "Run or load a simulation first.")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".h5",
                                             filetypes=[("HDF5 files", "*.h5"), ("All files", "*")])
        if not fname:
            return
        try:
            sim = self.sim if not self.full_sim_var.get() else None
            sim.save_sim(self.current_heatmap, self.current_streaks, self.current_count, fname)
            messagebox.showinfo("Saved", f"Simulation saved to {fname}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def run_sim(self):
        # Disable button and clear heatmap
        self.run_button.config(state='disabled')
        self.heatmap_ax.clear()
        self.heatmap_canvas.draw()
        self.traj_ax.clear()
        self.traj_canvas.draw()

        # Read parameters
        grid = self.grid_size_var.get()
        dt = self.dt_var.get()
        date = self.date_var.get()
        maxw = self.max_workers_var.get()
        full = self.full_sim_var.get()

        # Configure progressbar
        if full:
            total = len(CosmicRaySimulation.Z_list)
            self.progress.config(mode='determinate', maximum=total, value=0, style='Horizontal.TProgressbar')
        else:
            self.progress.config(mode='indeterminate', style='Horizontal.TProgressbar')
            self.progress.start(10)

        # Run in background thread
        threading.Thread(target=self._run_sim_thread,
                         args=(grid, dt, date, maxw, full),
                         daemon=True).start()

    def _run_sim_thread(self, grid, dt, date, maxw, full):
        try:
            if full:
                heatmap_list = []
                streaks_list = []
                counts = []
                combined = None
                for idx in range(len(CosmicRaySimulation.Z_list)):
                    sim = CosmicRaySimulation(species_index=idx,
                                              grid_size=grid, dt=dt,
                                              date=date,
                                              progress_bar=True,
                                              max_workers=maxw)
                    hm, st, cnt = sim.run_sim()
                    heatmap_list.append(hm)
                    streaks_list.append(st)
                    counts.append((sim.species_index, cnt))
                    # update progress bar
                    self.progress.after(0, lambda: self.progress.step(1))
                    # incremental heatmap
                    combined = hm if combined is None else combined + hm
                    self.progress.after(0, lambda data=combined: self._update_heatmap(data))
                self.current_heatmap = combined
                self.current_streaks = streaks_list
                self.current_count = sum(c for _,c in counts)
            else:
                sim = CosmicRaySimulation(grid_size=grid, dt=dt, date=date,
                                          progress_bar=True, max_workers=maxw)
                hm, streaks, cnt = sim.run_sim()
                self.current_heatmap = hm
                self.current_streaks = streaks
                self.current_count = cnt
                # stop indeterminate
                self.progress.after(0, self.progress.stop)
                self.progress.after(0, lambda: self.progress.config(mode='determinate', value=self.progress['maximum']))
                self.progress.after(0, lambda data=hm: self._update_heatmap(data))

            # after complete, draw final and 3D trajectory
            self.after(0, self._display_results)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Run Error", str(e)))
        finally:
            # re-enable button and mark complete
            self.after(0, lambda: self.run_button.config(state='normal'))
            self.after(0, lambda: self.progress.config(style="green.Horizontal.TProgressbar", value=self.progress['maximum']))

    def _update_heatmap(self, data):
        self.heatmap_ax.clear()
        self.heatmap_ax.imshow(data, cmap='inferno', origin='lower')
        self.heatmap_canvas.draw()

    def _display_results(self):
        # ensure heatmap is drawn
        self.heatmap_canvas.draw()
        # 3D trajectory of first primary
        try:
            full = self.full_sim_var.get()
            first = (self.current_streaks[0][0] if full else self.current_streaks[0])
            xs, ys, zs = zip(*first[0])
            self.traj_ax.clear()
            self.traj_ax.plot(xs, ys, zs, '-o', markersize=2)
            self.traj_canvas.draw()
        except Exception:
            pass

    def pause_sim(self):
        # TODO: implement pause/resume logic
        pass

    def stop_sim(self):
        # TODO: implement stop logic
        pass

if __name__ == "__main__":
    app = Application()
    app.mainloop()
