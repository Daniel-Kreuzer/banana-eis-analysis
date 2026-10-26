"""Desktop interface for batch fitting, result tables, and longitudinal plots."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

from .configuration import BatchConfig
from .database import Database
from .discovery import discover
from .engine import ELECTRODE_MODELS, FRUIT_MODELS, ShieldModel
from .exporting import export_results
from .pipeline import automatic_workers, run_jobs


ALL = "All"
GRAPH_MODES = ("Individual fit", "Date sweep", "Banana comparison", "Parameter trends")


def _complex_values(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.asarray(json.loads(row["frequency_json"]), dtype=float)
    raw = np.asarray(json.loads(row["real_json"]), dtype=float) + 1j * np.asarray(
        json.loads(row["imag_json"]), dtype=float
    )
    indices = np.asarray(json.loads(row["selected_indices_json"]), dtype=int)
    predicted = np.asarray(json.loads(row["predicted_real_json"]), dtype=float) + 1j * np.asarray(
        json.loads(row["predicted_imag_json"]), dtype=float
    )
    return frequency, raw, indices, predicted


def _display_values(row: dict, deembedded: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency, raw, indices, predicted = _complex_values(row)
    selected_frequency = frequency[indices]
    selected_raw = raw[indices]
    config = json.loads(row["config_json"])
    if config["correct_fixed_path"]:
        fixed = ShieldModel(
            config["fixed_resistance_ohm"], config["fixed_capacitance_f"]
        ).impedance(selected_frequency)
        if deembedded:
            return selected_frequency, selected_raw - fixed, predicted
        return selected_frequency, selected_raw, predicted + fixed
    return selected_frequency, selected_raw, predicted


def _rms_nrmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    """RMS complex error normalized by the RMS measured magnitude."""
    residual_rms = float(np.sqrt(np.mean(np.abs(predicted - measured) ** 2)))
    measured_rms = max(float(np.sqrt(np.mean(np.abs(measured) ** 2))), 1e-15)
    return residual_rms / measured_rms


def _domain_nrmse(row: dict) -> tuple[float | None, float | None]:
    """Return (de-embedded, raw/full-signal) NRMSE for a stored fit."""
    if row.get("nrmse") is None or not row.get("predicted_real_json"):
        return None, None
    deembedded_nrmse = float(row["nrmse"])
    _, raw, complete_model = _display_values(row, False)
    return deembedded_nrmse, _rms_nrmse(raw, complete_model)


def _between(value: str, start: str, end: str) -> bool:
    return (not start or value >= start) and (not end or value <= end)


def _trend_scale(values: list[float]) -> tuple[str, float | None]:
    """Choose a readable scale for positive parameter trends with a large range."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or np.any(finite <= 0):
        return "linear", None
    ratio = float(finite.max() / finite.min())
    return ("log", ratio) if ratio >= 100.0 else ("linear", ratio)


class BatchApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Fruit EIS batch fitting")
        root.geometry("1580x960")
        root.minsize(1200, 780)
        self.events: queue.Queue = queue.Queue()
        self.rows: list[dict] = []
        self.row_lookup: dict[int, dict] = {}
        self.config_lookup: dict[str, int] = {}
        self.sort_column = "date"
        self.sort_reverse = False

        self.folder = StringVar()
        self.status = StringVar(value="Choose the top-level banana capture folder.")
        self.population = IntVar(value=30)
        self.iterations = IntVar(value=800)
        self.restarts = IntVar(value=5)
        self.frequency_points = StringVar(value="all")
        self.refit = BooleanVar(value=False)
        self.correct_path = BooleanVar(value=True)

        self.table_banana = StringVar(value=ALL)
        self.table_start = StringVar()
        self.table_end = StringVar()
        self.table_config = StringVar(value=ALL)
        self.table_fruit = StringVar(value=ALL)
        self.table_electrode = StringVar(value=ALL)
        self.table_status = StringVar(value=ALL)

        self.graph_mode = StringVar(value=GRAPH_MODES[0])
        self.graph_banana = StringVar()
        self.graph_date = StringVar()
        self.graph_start = StringVar()
        self.graph_end = StringVar()
        self.graph_config = StringVar()
        self.graph_fruit = StringVar(value=FRUIT_MODELS[0])
        self.graph_electrode = StringVar(value=ELECTRODE_MODELS[0])
        self.graph_parameter = StringVar()
        self.graph_deembedded = BooleanVar(value=False)
        self.graph_show_models = BooleanVar(value=True)
        self._build()

    @property
    def database_path(self) -> Path:
        return Path(self.folder.get()) / "EIS_Batch_Results" / "circuit_fits.sqlite"

    def _build(self) -> None:
        self._build_header()
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.tables_tab = ttk.Frame(self.tabs, padding=8)
        self.graphs_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.tables_tab, text="Results tables")
        self.tabs.add(self.graphs_tab, text="Graphs")
        self._build_tables_tab()
        self._build_graphs_tab()

    def _build_header(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Banana capture folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse…", command=self._browse).grid(row=0, column=2)
        self.scan_button = ttk.Button(top, text="Scan captures", command=self._scan)
        self.scan_button.grid(row=0, column=3, padx=(8, 0))
        self.fit_button = ttk.Button(top, text="Fit models", command=self._fit)
        self.fit_button.grid(row=0, column=4, padx=(8, 0))
        self.export_button = ttk.Button(top, text="Export CSV", command=self._export)
        self.export_button.grid(row=0, column=5, padx=(8, 0))
        top.columnconfigure(1, weight=1)
        settings = ttk.LabelFrame(self.root, text="Versioned fitting configuration", padding=8)
        settings.pack(fill="x", padx=10)
        for column, (label, variable, width) in enumerate((
            ("Population", self.population, 8), ("Iterations", self.iterations, 8),
            ("Restarts", self.restarts, 8), ("Frequency points", self.frequency_points, 9),
        )):
            ttk.Label(settings, text=label).grid(row=0, column=column * 2, padx=(0, 4))
            ttk.Entry(settings, textvariable=variable, width=width).grid(
                row=0, column=column * 2 + 1, padx=(0, 14)
            )
        ttk.Checkbutton(settings, text="Remove fixed series path", variable=self.correct_path).grid(
            row=0, column=8, padx=8
        )
        ttk.Checkbutton(settings, text="Refit completed results", variable=self.refit).grid(
            row=0, column=9, padx=8
        )
        ttk.Label(settings, text=f"Automatic workers: {automatic_workers()}").grid(
            row=0, column=10, padx=(15, 0)
        )
        progress_frame = ttk.Frame(self.root, padding=(10, 8))
        progress_frame.pack(fill="x")
        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        ttk.Label(progress_frame, textvariable=self.status).pack(side="left", padx=(10, 0))

    def _build_tables_tab(self) -> None:
        filters = ttk.LabelFrame(self.tables_tab, text="Filters", padding=7)
        filters.pack(fill="x", pady=(0, 8))
        controls = (
            ("Banana", self.table_banana, 12), ("Start date", self.table_start, 12),
            ("End date", self.table_end, 12), ("Configuration", self.table_config, 27),
            ("Fruit", self.table_fruit, 11), ("Electrode", self.table_electrode, 14),
            ("Status", self.table_status, 10),
        )
        self.table_filter_boxes = []
        for column, (label, variable, width) in enumerate(controls):
            ttk.Label(filters, text=label).grid(row=0, column=column, sticky="w", padx=3)
            box = ttk.Combobox(filters, textvariable=variable, width=width, state="readonly")
            box.grid(row=1, column=column, padx=3)
            box.bind("<<ComboboxSelected>>", lambda _event: self._apply_table_filters())
            self.table_filter_boxes.append(box)
        ttk.Button(filters, text="Clear filters", command=self._clear_table_filters).grid(
            row=1, column=len(controls), padx=(10, 3)
        )
        body = ttk.Panedwindow(self.tables_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        table_frame = ttk.Frame(body)
        detail_frame = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(table_frame, weight=4)
        body.add(detail_frame, weight=1)
        columns = ("banana", "date", "config", "fruit", "electrode",
                   "nrmse_deembedded", "nrmse_raw", "aicc", "status")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        widths = (75, 95, 185, 80, 95, 115, 95, 90, 70)
        headings = {
            "nrmse_deembedded": "De-embedded NRMSE",
            "nrmse_raw": "Raw NRMSE",
        }
        for column, width in zip(columns, widths, strict=True):
            self.results_table.heading(
                column, text=headings.get(column, column.replace("_", " ").title()),
                command=lambda name=column: self._sort_table(name),
            )
            self.results_table.column(
                column, width=width,
                anchor="e" if column in ("nrmse_deembedded", "nrmse_raw", "aicc") else "w",
            )
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scroll.set)
        self.results_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.results_table.bind("<<TreeviewSelect>>", self._table_selection_changed)
        ttk.Label(detail_frame, text="Selected fit", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.fit_detail_widget = ScrolledText(detail_frame, width=42, wrap="word", font=("Segoe UI", 9))
        self.fit_detail_widget.pack(fill="both", expand=True, pady=(8, 8))
        self._set_text(self.fit_detail_widget, "Select a result to view its complete parameters and configuration.")
        ttk.Button(detail_frame, text="View this fit in Graphs", command=self._view_selected_fit).pack(fill="x")

    def _build_graphs_tab(self) -> None:
        controls = ttk.LabelFrame(self.graphs_tab, text="Graph options", padding=7)
        controls.pack(fill="x", pady=(0, 8))
        definitions = (
            ("View", self.graph_mode, GRAPH_MODES, 18), ("Banana", self.graph_banana, (), 12),
            ("Date", self.graph_date, (), 12), ("Start", self.graph_start, (), 12),
            ("End", self.graph_end, (), 12), ("Configuration", self.graph_config, (), 27),
            ("Fruit", self.graph_fruit, FRUIT_MODELS, 10),
            ("Electrode", self.graph_electrode, ELECTRODE_MODELS, 13),
            ("Parameter", self.graph_parameter, (), 11),
        )
        self.graph_boxes = {}
        for column, (label, variable, values, width) in enumerate(definitions):
            ttk.Label(controls, text=label).grid(row=0, column=column, sticky="w", padx=2)
            box = ttk.Combobox(controls, textvariable=variable, values=values, width=width, state="readonly")
            box.grid(row=1, column=column, padx=2)
            self.graph_boxes[label] = box
            box.bind("<<ComboboxSelected>>", lambda _event: self._graph_option_changed())
        ttk.Checkbutton(
            controls, text="De-embedded", variable=self.graph_deembedded, command=self._draw_graphs
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            controls, text="Show fitted models", variable=self.graph_show_models, command=self._draw_graphs
        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Button(controls, text="Refresh", command=self._draw_graphs).grid(row=2, column=8, sticky="e", pady=(7, 0))
        body = ttk.Panedwindow(self.graphs_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        plot_frame = ttk.Frame(body)
        detail_frame = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(plot_frame, weight=5)
        body.add(detail_frame, weight=1)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(11, 7), constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(detail_frame, text="Graph details", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.graph_detail_widget = ScrolledText(detail_frame, width=36, wrap="word", font=("Segoe UI", 9))
        self.graph_detail_widget.pack(fill="both", expand=True, pady=(8, 0))
        self._set_text(self.graph_detail_widget, "Select graph options and refresh.")

    @staticmethod
    def _set_text(widget: ScrolledText, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Choose top-level banana capture folder")
        if folder:
            self.folder.set(folder)
            if self.database_path.exists():
                self._refresh()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in (self.scan_button, self.fit_button, self.export_button):
            button.configure(state=state)

    def _scan(self) -> None:
        folder = Path(self.folder.get())
        if not folder.is_dir():
            messagebox.showerror("Folder not found", "Choose the top-level banana capture folder first.")
            return
        self._set_busy(True)
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.status.set("Scanning and validating captures…")
        threading.Thread(target=self._scan_worker, args=(folder,), daemon=True).start()
        self.root.after(100, self._poll)

    def _scan_worker(self, folder: Path) -> None:
        try:
            captures, failures = discover(folder)
            database = Database(self.database_path)
            for capture in captures:
                database.add_capture(capture)
            database.close()
            self.events.put(("scan_done", len(captures), failures))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _current_config(self) -> BatchConfig:
        text = self.frequency_points.get().strip().lower()
        points = None if text in ("", "all") else int(text)
        config = BatchConfig(
            population=self.population.get(), iterations=self.iterations.get(),
            restarts=self.restarts.get(), frequency_points=points,
            correct_fixed_path=self.correct_path.get(),
        )
        config.validate()
        return config

    def _fit(self) -> None:
        if not self.database_path.exists():
            messagebox.showerror("No catalogue", "Scan the capture folder before fitting.")
            return
        try:
            config = self._current_config()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        self._set_busy(True)
        self.progress.configure(mode="determinate", value=0, maximum=1)
        self.status.set(f"Preparing {config.short_name()}…")
        threading.Thread(target=self._fit_worker, args=(config, self.refit.get()), daemon=True).start()
        self.root.after(100, self._poll)

    def _fit_worker(self, config: BatchConfig, refit: bool) -> None:
        database = Database(self.database_path)
        try:
            def progress(done: int, total: int, message: str) -> None:
                self.events.put(("fit_progress", done, total, message))
            succeeded, failed = run_jobs(database, config, refit, progress)
            self.events.put(("fit_done", succeeded, failed, config.short_name()))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            database.close()

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "scan_done":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self._set_busy(False)
                    self.status.set(f"Found {event[1]} valid captures; {len(event[2])} files skipped.")
                    self._refresh()
                    if event[2]:
                        preview = "\n".join(f"{path.name}: {message}" for path, message in event[2][:8])
                        messagebox.showwarning("Files skipped", preview)
                    return
                if event[0] == "fit_progress":
                    self.progress.configure(maximum=event[2], value=event[1])
                    self.status.set(f"{event[1]}/{event[2]} — {event[3]}")
                elif event[0] == "fit_done":
                    self._set_busy(False)
                    self.status.set(f"Complete: {event[1]} fitted, {event[2]} failed — {event[3]}")
                    self._refresh()
                    return
                elif event[0] == "error":
                    self.progress.stop()
                    self._set_busy(False)
                    self.status.set("Operation failed.")
                    messagebox.showerror("Operation failed", event[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _refresh(self) -> None:
        if not self.database_path.exists():
            return
        database = Database(self.database_path)
        self.rows = [dict(row) for row in database.fit_rows()]
        captures = [dict(row) for row in database.captures()]
        configs = [dict(row) for row in database.configs()]
        database.close()
        self.row_lookup = {row["id"]: row for row in self.rows}
        bananas = sorted({row["banana_id"] for row in captures})
        dates = sorted({row["measurement_date"] for row in captures})
        names = [row["display_name"] for row in configs]
        self.config_lookup = {row["display_name"]: row["id"] for row in configs}
        table_values = (
            [ALL, *bananas], ["", *dates], ["", *dates], [ALL, *names],
            [ALL, *FRUIT_MODELS], [ALL, *ELECTRODE_MODELS], [ALL, "complete", "failed", "running"],
        )
        for box, values in zip(self.table_filter_boxes, table_values, strict=True):
            box.configure(values=values)
        graph_values = {"Banana": bananas, "Date": dates, "Start": ["", *dates],
                        "End": ["", *dates], "Configuration": names}
        for label, values in graph_values.items():
            self.graph_boxes[label].configure(values=values)
        if bananas and self.graph_banana.get() not in bananas:
            self.graph_banana.set(bananas[0])
        if dates and self.graph_date.get() not in dates:
            self.graph_date.set(dates[0])
        if names and self.graph_config.get() not in names:
            self.graph_config.set(names[0])
        self._update_parameter_options()
        self._apply_table_filters()
        self._draw_graphs()

    def _clear_table_filters(self) -> None:
        for variable, value in ((self.table_banana, ALL), (self.table_start, ""), (self.table_end, ""),
                                (self.table_config, ALL), (self.table_fruit, ALL),
                                (self.table_electrode, ALL), (self.table_status, ALL)):
            variable.set(value)
        self._apply_table_filters()

    def _filtered_rows(self) -> list[dict]:
        rows = []
        for row in self.rows:
            if self.table_banana.get() != ALL and row["banana_id"] != self.table_banana.get():
                continue
            if not _between(row["measurement_date"], self.table_start.get(), self.table_end.get()):
                continue
            if self.table_config.get() != ALL and row["display_name"] != self.table_config.get():
                continue
            if self.table_fruit.get() != ALL and row["fruit_model"] != self.table_fruit.get():
                continue
            if self.table_electrode.get() != ALL and row["electrode_model"] != self.table_electrode.get():
                continue
            if self.table_status.get() != ALL and row["status"] != self.table_status.get():
                continue
            rows.append(row)
        return rows

    def _sort_value(self, row: dict, column: str):
        key = {"banana": "banana_id", "date": "measurement_date", "config": "display_name",
               "fruit": "fruit_model", "electrode": "electrode_model"}.get(column, column)
        if column in ("nrmse_deembedded", "nrmse_raw"):
            deembedded, raw = _domain_nrmse(row)
            value = deembedded if column == "nrmse_deembedded" else raw
            return float("inf") if value is None else value
        value = row.get(key)
        if value is None:
            return float("inf") if column == "aicc" else ""
        return value

    def _apply_table_filters(self) -> None:
        rows = sorted(self._filtered_rows(), key=lambda row: self._sort_value(row, self.sort_column),
                      reverse=self.sort_reverse)
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        for row in rows:
            deembedded_nrmse, raw_nrmse = _domain_nrmse(row)
            deembedded_text = "—" if deembedded_nrmse is None else f"{deembedded_nrmse:.3%}"
            raw_text = "—" if raw_nrmse is None else f"{raw_nrmse:.3%}"
            aicc = "—" if row["aicc"] is None else f"{row['aicc']:.2f}"
            self.results_table.insert("", "end", iid=str(row["id"]), values=(
                row["banana_id"], row["measurement_date"], row["display_name"],
                row["fruit_model"], row["electrode_model"], deembedded_text, raw_text,
                aicc, row["status"],
            ))
        message = f"Showing {len(rows)} of {len(self.rows)} fit results."
        visible_configs = {row["config_id"] for row in rows}
        if len(visible_configs) > 1:
            message += " Filter to one configuration before comparing AICc values."
        self.status.set(message)

    def _sort_table(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column, self.sort_reverse = column, False
        self._apply_table_filters()

    def _selected_table_row(self) -> dict | None:
        selected = self.results_table.selection()
        return self.row_lookup.get(int(selected[0])) if selected else None

    def _table_selection_changed(self, _event=None) -> None:
        row = self._selected_table_row()
        if row is not None:
            self._set_text(self.fit_detail_widget, self._detail_text(row, include_config=True))

    def _detail_text(self, row: dict, include_config: bool = False) -> str:
        lines = [f"{row['banana_id']} — {row['measurement_date']}",
                 f"{row['fruit_model']} + {row['electrode_model']}", f"Status: {row['status']}"]
        if row["nrmse"] is not None:
            deembedded_nrmse, raw_nrmse = _domain_nrmse(row)
            lines.extend((f"De-embedded NRMSE: {deembedded_nrmse:.4%}",
                          f"Raw-domain NRMSE: {raw_nrmse:.4%}", f"AICc: {row['aicc']:.4f}",
                          f"AIC: {row['aic']:.4f}", f"BIC: {row['bic']:.4f}",
                          f"Objective: {row['objective']:.6g}", f"Best seed: {row['best_seed']}",
                          f"Evaluations: {row['evaluations']:,}"))
        if row["parameters_json"]:
            lines.append("\nParameters")
            for name, value in json.loads(row["parameters_json"]).items():
                lines.append(f"{name}: {value:.7g}")
        if include_config:
            config = json.loads(row["config_json"])
            lines.append("\nConfiguration")
            for name in ("population", "iterations", "restarts", "frequency_points", "objective",
                         "correct_fixed_path", "seed", "abandonment_probability", "levy_beta", "step_scale"):
                lines.append(f"{name}: {config.get(name)}")
        if row["error_message"]:
            lines.append(f"\nError: {row['error_message']}")
        return "\n".join(lines)

    def _view_selected_fit(self) -> None:
        row = self._selected_table_row()
        if row is None:
            messagebox.showinfo("No result selected", "Select a result row first.")
            return
        self.graph_mode.set("Individual fit")
        self.graph_banana.set(row["banana_id"])
        self.graph_date.set(row["measurement_date"])
        self.graph_config.set(row["display_name"])
        self.graph_fruit.set(row["fruit_model"])
        self.graph_electrode.set(row["electrode_model"])
        self._update_parameter_options()
        self.tabs.select(self.graphs_tab)
        self._draw_graphs()

    def _graph_option_changed(self) -> None:
        self._update_parameter_options()
        self._draw_graphs()

    def _graph_rows(self) -> list[dict]:
        return [row for row in self.rows if row["display_name"] == self.graph_config.get()
                and row["fruit_model"] == self.graph_fruit.get()
                and row["electrode_model"] == self.graph_electrode.get()
                and row["status"] == "complete"]

    def _update_parameter_options(self) -> None:
        parameters = sorted({name for row in self._graph_rows() if row["parameters_json"]
                             for name in json.loads(row["parameters_json"])})
        self.graph_boxes["Parameter"].configure(values=parameters)
        if parameters and self.graph_parameter.get() not in parameters:
            self.graph_parameter.set(parameters[0])

    def _draw_graphs(self) -> None:
        for axis in list(self.figure.axes)[4:]:
            axis.remove()
        for axis in self.axes.flat:
            axis.clear()
        mode, rows = self.graph_mode.get(), self._graph_rows()
        if not rows:
            self.figure.suptitle("No completed fits match the selected options")
            self.canvas.draw_idle()
            return
        if mode == "Individual fit":
            self._draw_individual(rows)
        elif mode == "Date sweep":
            selected = [row for row in rows if row["banana_id"] == self.graph_banana.get()
                        and _between(row["measurement_date"], self.graph_start.get(), self.graph_end.get())]
            self._draw_comparison(selected, "measurement_date", f"{self.graph_banana.get()} date sweep")
        elif mode == "Banana comparison":
            selected = [row for row in rows if row["measurement_date"] == self.graph_date.get()]
            self._draw_comparison(selected, "banana_id", f"Banana comparison — {self.graph_date.get()}")
        else:
            selected = [row for row in rows if row["banana_id"] == self.graph_banana.get()
                        and _between(row["measurement_date"], self.graph_start.get(), self.graph_end.get())]
            self._draw_parameter_trends(selected)
        for axis in self.axes.flat:
            axis.grid(True, which="both", alpha=0.25)
        self.canvas.draw_idle()

    def _draw_individual(self, rows: list[dict]) -> None:
        matches = [row for row in rows if row["banana_id"] == self.graph_banana.get()
                   and row["measurement_date"] == self.graph_date.get()]
        if not matches:
            self.figure.suptitle("No completed fit for this banana and date")
            return
        row = matches[-1]
        frequency, measured, predicted = _display_values(row, self.graph_deembedded.get())
        magnitude, phase, nyquist, residual = self.axes.flat
        measured_label = "De-embedded measurement" if self.graph_deembedded.get() else "Raw measurement"
        model_label = "Fruit + electrode model" if self.graph_deembedded.get() else "Complete model"
        magnitude.loglog(frequency, np.abs(measured), "o", ms=4, label=measured_label)
        magnitude.loglog(frequency, np.abs(predicted), "-", lw=2, label=model_label)
        phase.semilogx(frequency, np.angle(measured, deg=True), "o", ms=4, label=measured_label)
        phase.semilogx(frequency, np.angle(predicted, deg=True), "-", lw=2, label=model_label)
        nyquist.plot(measured.real, -measured.imag, "o", ms=4, label=measured_label)
        nyquist.plot(predicted.real, -predicted.imag, "-", lw=2, label=model_label)
        relative = np.abs(predicted - measured) / np.maximum(np.abs(measured), 1e-15) * 100
        residual.semilogx(frequency, relative, "o-")
        residual.axhline(5, color="tab:orange", ls="--", lw=1)
        magnitude.set(title="Magnitude", xlabel="Frequency (Hz)", ylabel="|Z| (Ω)")
        phase.set(title="Phase", xlabel="Frequency (Hz)", ylabel="Phase (°)")
        nyquist.set(title="Nyquist", xlabel="Re(Z) (Ω)", ylabel="−Im(Z) (Ω)")
        domain = "De-embedded" if self.graph_deembedded.get() else "Raw domain"
        residual.set(title=f"{domain} relative residual", xlabel="Frequency (Hz)", ylabel="Error (%)")
        magnitude.legend(fontsize=8)
        self.figure.suptitle(
            f"{row['banana_id']} — {row['measurement_date']} — {row['fruit_model']} + {row['electrode_model']}"
        )
        self._set_text(self.graph_detail_widget, self._detail_text(row, include_config=True))

    def _draw_comparison(self, rows: list[dict], color_key: str, title: str) -> None:
        from matplotlib import colormaps
        rows.sort(key=lambda row: row[color_key])
        colors = colormaps["viridis"](np.linspace(0.05, 0.95, max(len(rows), 1)))
        magnitude, phase, nrmse_axis, aicc_axis = self.axes.flat
        labels, nrmse, aicc = [], [], []
        for color, row in zip(colors, rows, strict=True):
            frequency, measured, predicted = _display_values(row, self.graph_deembedded.get())
            label = row[color_key]
            labels.append(label)
            magnitude.loglog(frequency, np.abs(measured), "o", ms=2.5, color=color, label=label)
            phase.semilogx(frequency, np.angle(measured, deg=True), "o", ms=2.5, color=color, label=label)
            if self.graph_show_models.get():
                magnitude.loglog(frequency, np.abs(predicted), "-", lw=1.4, color=color)
                phase.semilogx(frequency, np.angle(predicted, deg=True), "-", lw=1.4, color=color)
            nrmse.append(row["nrmse"] * 100)
            aicc.append(row["aicc"])
        magnitude.set(title="Magnitude", xlabel="Frequency (Hz)", ylabel="|Z| (Ω)")
        phase.set(title="Phase", xlabel="Frequency (Hz)", ylabel="Phase (°)")
        nrmse_axis.plot(labels, nrmse, "o-")
        nrmse_axis.set(title="De-embedded NRMSE", xlabel=color_key.replace("_", " ").title(),
                       ylabel="NRMSE (%)")
        aicc_axis.plot(labels, aicc, "s-", color="tab:orange")
        aicc_axis.set(title="AICc", xlabel=color_key.replace("_", " ").title(), ylabel="AICc")
        magnitude.legend(fontsize=8, ncol=max(1, min(3, len(rows) // 7 + 1)))
        nrmse_axis.tick_params(axis="x", rotation=30)
        aicc_axis.tick_params(axis="x", rotation=30)
        self.figure.suptitle(title)
        model_note = "Markers are measurements; lines are fitted models." if self.graph_show_models.get() else "Markers show measurements; model curves are hidden."
        self._set_text(self.graph_detail_widget,
            f"{len(rows)} fits shown\n{self.graph_fruit.get()} + {self.graph_electrode.get()}\n"
            f"Configuration: {self.graph_config.get()}\n\n{model_note}"
        )

    def _draw_parameter_trends(self, rows: list[dict]) -> None:
        rows.sort(key=lambda row: row["measurement_date"])
        dates = [row["measurement_date"] for row in rows]
        parameter = self.graph_parameter.get()
        values = [json.loads(row["parameters_json"]).get(parameter, np.nan) for row in rows]
        parameter_axis, nrmse_axis, aicc_axis, objective_axis = self.axes.flat
        parameter_axis.plot(dates, values, "o-")
        scale, dynamic_range = _trend_scale(values)
        parameter_axis.set_yscale(scale)
        nrmse_axis.plot(dates, [row["nrmse"] * 100 for row in rows], "o-")
        aicc_axis.plot(dates, [row["aicc"] for row in rows], "s-", color="tab:orange")
        objective_axis.semilogy(dates, [row["objective"] for row in rows], "o-", color="tab:green")
        scale_note = " (log scale)" if scale == "log" else ""
        parameter_axis.set(title=f"{parameter or 'Parameter'}{scale_note}", xlabel="Date",
                           ylabel=parameter or "Value")
        nrmse_axis.set(title="De-embedded NRMSE", xlabel="Date", ylabel="NRMSE (%)")
        aicc_axis.set(title="AICc", xlabel="Date", ylabel="AICc")
        objective_axis.set(title="Optimization objective", xlabel="Date", ylabel="Relative complex error")
        for axis in self.axes.flat:
            axis.tick_params(axis="x", rotation=30)
        self.figure.suptitle(f"{self.graph_banana.get()} parameter and fit-quality trends")
        value_lines = [
            f"{date}: {value:.8g}" if np.isfinite(value) else f"{date}: unavailable"
            for date, value in zip(dates, values, strict=True)
        ]
        range_note = ""
        if scale == "log" and dynamic_range is not None:
            range_note = (
                f"\nY-axis: logarithmic ({dynamic_range:.1f}× range). "
                "A linear axis would make the smaller values appear to be zero."
            )
        self._set_text(
            self.graph_detail_widget,
            f"{len(rows)} dates shown\n{self.graph_fruit.get()} + {self.graph_electrode.get()}\n"
            f"Parameter: {parameter}\nConfiguration: {self.graph_config.get()}{range_note}\n\n"
            "Values\n" + "\n".join(value_lines),
        )

    def _export(self) -> None:
        if not self.database_path.exists():
            messagebox.showerror("No database", "Scan and fit captures first.")
            return
        database = Database(self.database_path)
        paths = export_results(database, self.database_path.parent)
        database.close()
        self.status.set(f"Exported {paths[0].name} and {paths[1].name}")
        if os.name == "nt":
            os.startfile(paths[0].parent)  # type: ignore[attr-defined]


def launch() -> None:
    root = Tk()
    BatchApp(root)
    root.mainloop()
