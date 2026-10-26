"""Interactive model-fitting viewer for fruit impedance spectra."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk

import numpy as np

from .cli import _write_outputs
from .cuckoo import CuckooSearchConfig
from .fit import FitConfig, FitResult, fit_spectrum
from .io import Spectrum, load_spectrum
from .models import ELECTRODE_MODELS, FRUIT_MODELS


FRUIT_LABELS = {"cole": "Cole", "paper": "R2 || (R1 + C1)"}
ELECTRODE_LABELS = {
    "none": "No electrode model",
    "capacitor": "Series capacitor",
    "warburg": "Warburg",
    "randles_cpe": "Randles/CPE",
}


def _model_name(result: FitResult) -> str:
    return f"{FRUIT_LABELS[result.fruit_model]} + {ELECTRODE_LABELS[result.electrode_model]}"


def _format_parameter(name: str, value: float) -> str:
    units = {
        "r_inf": "Ω", "r0": "Ω", "delta_r": "Ω", "r1": "Ω", "r2": "Ω",
        "r_e": "Ω", "tau_s": "s", "c1_f": "F", "c_e_f": "F",
        "sigma": "Ω·s⁻½", "q_e": "S·sⁿ",
    }
    return f"{name} = {value:.5g} {units.get(name, '')}".rstrip()


def fit_insights(result: FitResult, all_results: list[FitResult]) -> list[str]:
    """Generate cautious, human-readable diagnostics for one fitted model."""
    nrmse = result.metrics["nrmse_complex"]
    if nrmse < 0.02:
        quality = "Very close numerical fit (NRMSE below 2%)."
    elif nrmse < 0.05:
        quality = "Good numerical fit (NRMSE between 2% and 5%)."
    elif nrmse < 0.10:
        quality = "Moderate fit (NRMSE between 5% and 10%)."
    else:
        quality = "Weak fit (NRMSE above 10%); inspect the residual pattern."

    insights = [quality]
    ranked = sorted(all_results, key=lambda item: item.metrics["aicc"])
    rank = ranked.index(result) + 1
    insights.append(f"AICc rank: {rank} of {len(ranked)} (lower is better and penalizes extra parameters).")

    objectives = np.asarray(result.run_objectives, dtype=float)
    if len(objectives) > 1:
        spread = (np.max(objectives) - np.min(objectives)) / max(np.min(objectives), 1e-15)
        if spread < 0.02:
            insights.append("Independent restarts converged consistently.")
        elif spread < 0.20:
            insights.append("Some restart variation remains; fitted parameters deserve caution.")
        else:
            insights.append("Large restart variation suggests multiple minima or weak parameter identifiability.")

    bounds = result.configuration.get("parameter_bounds", [])
    near_bounds = []
    for bound in bounds:
        value = result.parameters.get(bound["name"])
        if value is None:
            continue
        if bound.get("scale") == "log":
            position = (np.log10(value) - np.log10(bound["lower"])) / (
                np.log10(bound["upper"]) - np.log10(bound["lower"])
            )
        else:
            position = (value - bound["lower"]) / (bound["upper"] - bound["lower"])
        if position < 0.015 or position > 0.985:
            near_bounds.append(bound["name"])
    if near_bounds:
        insights.append("Parameters near search bounds: " + ", ".join(near_bounds) + ".")

    residual = result.predicted_impedance_ohm - result.corrected_impedance_ohm
    relative = np.abs(residual) / np.maximum(np.abs(result.corrected_impedance_ohm), 1e-15)
    low = float(np.mean(relative[: max(2, len(relative) // 5)]))
    high = float(np.mean(relative[-max(2, len(relative) // 5) :]))
    if max(low, high) > 1.8 * max(min(low, high), 1e-15):
        region = "low" if low > high else "high"
        insights.append(f"Residuals are concentrated at {region} frequencies, indicating systematic mismatch there.")
    insights.append("Fit quality alone does not prove that fruit and electrode parameters are uniquely identifiable.")
    return insights


class FitViewer:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Fruit EIS model fitting")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 720)
        self.file_var = StringVar()
        self.status_var = StringVar(value="Choose a SensorPal XLSX or impedance CSV file.")
        self.shield_var = BooleanVar(value=True)
        self.corrected_view_var = BooleanVar(value=False)
        self.events: queue.Queue = queue.Queue()
        self.spectrum: Spectrum | None = None
        self.results: list[FitResult] = []
        self.output_dir: Path | None = None
        self._build()

    def _build(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")
        ttk.Label(controls, text="Fruit capture file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.file_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse…", command=self._browse).grid(row=0, column=2)
        self.run_button = ttk.Button(controls, text="Fit all 8 models", command=self._start)
        self.run_button.grid(row=0, column=3, padx=(8, 0))
        self.open_button = ttk.Button(controls, text="Open output folder", command=self._open_output, state="disabled")
        self.open_button.grid(row=0, column=4, padx=(8, 0))
        ttk.Checkbutton(controls, text="Remove known shield path", variable=self.shield_var).grid(
            row=1, column=1, sticky="w", pady=(8, 0)
        )
        self.corrected_view = ttk.Checkbutton(
            controls,
            text="Show de-embedded fitting-domain plots",
            variable=self.corrected_view_var,
            command=self._redraw_selected,
            state="disabled",
        )
        self.corrected_view.grid(row=1, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        status = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        status.pack(fill="x")
        self.progress = ttk.Progressbar(status, maximum=8, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=(10, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=3)

        columns = ("rank", "model", "nrmse", "aicc")
        self.table = ttk.Treeview(left, columns=columns, show="headings", height=12)
        for key, title, width in (
            ("rank", "Rank", 45), ("model", "Model", 220),
            ("nrmse", "NRMSE", 75), ("aicc", "AICc", 90),
        ):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="w" if key == "model" else "e")
        self.table.pack(fill="x")
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)

        ttk.Label(left, text="Interpretation", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 4))
        self.insight = StringVar(value="Results and cautions will appear here.")
        ttk.Label(left, textvariable=self.insight, wraplength=330, justify="left").pack(fill="x", anchor="w")
        ttk.Label(left, text="Fitted parameters", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 4))
        self.parameters = StringVar(value="—")
        ttk.Label(left, textvariable=self.parameters, wraplength=330, justify="left").pack(fill="x", anchor="w")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.draw()
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose fruit impedance data",
            filetypes=(("Impedance data", "*.xlsx *.csv"), ("Excel workbook", "*.xlsx"), ("CSV", "*.csv")),
        )
        if selected:
            self.file_var.set(selected)

    def _start(self) -> None:
        path = Path(self.file_var.get().strip())
        if not path.is_file():
            messagebox.showerror("File not found", "Choose an existing XLSX or CSV file first.")
            return
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.corrected_view.configure(state="disabled")
        self.corrected_view_var.set(False)
        self.progress.configure(value=0)
        self.status_var.set("Loading data…")
        self.results = []
        for item in self.table.get_children():
            self.table.delete(item)
        threading.Thread(target=self._worker, args=(path, self.shield_var.get()), daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self, path: Path, correct_shield: bool) -> None:
        try:
            spectrum = load_spectrum(path)
            config = FitConfig(correct_shield=correct_shield, cuckoo=CuckooSearchConfig())
            results = []
            combinations = [(fruit, electrode) for fruit in FRUIT_MODELS for electrode in ELECTRODE_MODELS]
            for index, (fruit, electrode) in enumerate(combinations, start=1):
                self.events.put(("progress", index - 1, f"Fitting {FRUIT_LABELS[fruit]} + {ELECTRODE_LABELS[electrode]}…"))
                results.append(fit_spectrum(spectrum.frequency_hz, spectrum.impedance_ohm, fruit, electrode, config))
            output_dir = path.parent / f"{path.stem}_fit_results"
            _write_outputs(output_dir, spectrum, results)
            self.events.put(("done", spectrum, results, output_dir))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    self.progress.configure(value=event[1])
                    self.status_var.set(event[2])
                elif event[0] == "done":
                    self._complete(event[1], event[2], event[3])
                    return
                elif event[0] == "error":
                    self.run_button.configure(state="normal")
                    self.status_var.set("Fit failed.")
                    messagebox.showerror("Fit failed", event[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _complete(self, spectrum: Spectrum, results: list[FitResult], output_dir: Path) -> None:
        self.spectrum, self.results, self.output_dir = spectrum, results, output_dir
        ranked = sorted(results, key=lambda item: item.metrics["aicc"])
        for rank, result in enumerate(ranked, start=1):
            self.table.insert("", "end", iid=str(results.index(result)), values=(
                rank, _model_name(result), f"{result.metrics['nrmse_complex']:.3%}", f"{result.metrics['aicc']:.2f}",
            ))
        self.progress.configure(value=8)
        self.status_var.set(f"Complete — results saved to {output_dir.name}")
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.corrected_view.configure(state="normal")
        first = self.table.get_children()[0]
        self.table.selection_set(first)
        self.table.focus(first)
        self._show_result(self.results[int(first)])

    def _selection_changed(self, _event=None) -> None:
        selected = self.table.selection()
        if selected:
            self._show_result(self.results[int(selected[0])])

    def _redraw_selected(self) -> None:
        self._selection_changed()

    def _show_result(self, result: FitResult) -> None:
        if self.spectrum is None:
            return
        frequency = self.spectrum.frequency_hz
        if self.corrected_view_var.get():
            measured = result.corrected_impedance_ohm
            predicted = result.predicted_impedance_ohm
            measured_label = "De-embedded measurement"
            model_label = "Fruit + electrode model"
            view_label = "de-embedded fitting domain"
        else:
            measured = self.spectrum.impedance_ohm
            if result.configuration["fit"]["correct_shield"]:
                shield = result.configuration["shield"]
                from .models import ShieldModel

                predicted = result.predicted_impedance_ohm + ShieldModel(
                    shield["resistance_ohm"], shield["capacitance_f"]
                ).impedance(frequency)
                model_label = "Complete model including fixed path"
            else:
                predicted = result.predicted_impedance_ohm
                model_label = "Model"
            measured_label = "Raw Excel measurement"
            view_label = "raw measurement domain"
        relative = np.abs(predicted - measured) / np.maximum(np.abs(measured), 1e-15) * 100.0
        for axis in self.axes.flat:
            axis.clear()

        mag, phase, nyquist, residual = self.axes.flat
        mag.loglog(frequency, np.abs(measured), "o", ms=4, label=measured_label)
        mag.loglog(frequency, np.abs(predicted), "-", lw=2, label=model_label)
        mag.set(title="Magnitude", xlabel="Frequency (Hz)", ylabel="|Z| (Ω)")
        mag.legend()
        phase.semilogx(frequency, np.angle(measured, deg=True), "o", ms=4, label=measured_label)
        phase.semilogx(frequency, np.angle(predicted, deg=True), "-", lw=2, label=model_label)
        phase.set(title="Phase", xlabel="Frequency (Hz)", ylabel="Phase (°)")
        nyquist.plot(measured.real, -measured.imag, "o", ms=4, label=measured_label)
        nyquist.plot(predicted.real, -predicted.imag, "-", lw=2, label=model_label)
        nyquist.set(title="Nyquist", xlabel="Re(Z) (Ω)", ylabel="−Im(Z) (Ω)")
        residual.semilogx(frequency, relative, "o-")
        residual.axhline(5.0, color="tab:orange", ls="--", lw=1)
        residual.set(title="Relative residual", xlabel="Frequency (Hz)", ylabel="Error (%)")
        for axis in self.axes.flat:
            axis.grid(True, which="both", alpha=0.25)
        self.figure.suptitle(f"{_model_name(result)} — {view_label}", fontsize=14)
        self.canvas.draw_idle()
        self.insight.set("\n\n".join(f"• {line}" for line in fit_insights(result, self.results)))
        self.parameters.set("\n".join(_format_parameter(k, v) for k, v in result.parameters.items()))

    def _open_output(self) -> None:
        if self.output_dir is None:
            return
        if sys.platform == "win32":
            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(self.output_dir)])


def launch() -> None:
    root = Tk()
    FitViewer(root)
    root.mainloop()
