"""Small front end for the reproducible MLP trainer."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import TrainingConfig
from .data import dataset_names
from .trainer import run_training


class MLTrainingApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Fruit EIS MLP study")
        root.geometry("1050x720")
        self.dataset_directory = StringVar()
        self.output_directory = StringVar()
        self.outer_folds = IntVar(value=5)
        self.inner_folds = IntVar(value=4)
        self.candidates = IntVar(value=24)
        self.max_iterations = IntVar(value=5000)
        self.resume = BooleanVar(value=True)
        self.status = StringVar(value="Choose the prepared ML_Datasets folder.")
        self.selection_status = StringVar(value="No datasets loaded.")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Prepared datasets").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.dataset_directory).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._choose_datasets).grid(row=0, column=2)
        ttk.Label(frame, text="Results folder").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.output_directory).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._choose_output).grid(row=1, column=2)

        options = ttk.LabelFrame(frame, text="Nested grouped cross-validation", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 8))
        for index, (label, variable) in enumerate((
            ("Outer folds", self.outer_folds), ("Inner folds", self.inner_folds),
            ("Random candidates", self.candidates), ("Max iterations", self.max_iterations),
        )):
            ttk.Label(options, text=label).grid(row=0, column=index * 2, padx=(0, 4))
            ttk.Entry(options, textvariable=variable, width=7).grid(row=0, column=index * 2 + 1, padx=(0, 14))
        ttk.Checkbutton(options, text="Resume matching run", variable=self.resume).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        ttk.Label(options, text="Preprocessing: standard, robust, and circuit log+standard (inside each fold)").grid(
            row=1, column=3, columnspan=4, sticky="w", pady=(8, 0)
        )
        ttk.Label(options, text="Variants: MLP-ReLU and MLP-tanh; selection metric: macro F1").grid(
            row=2, column=0, columnspan=7, sticky="w", pady=(5, 0)
        )

        datasets = ttk.LabelFrame(frame, text="Datasets", padding=8)
        datasets.grid(row=3, column=0, columnspan=3, sticky="nsew")
        datasets.rowconfigure(0, weight=1)
        datasets.columnconfigure(0, weight=1)
        # ``multiple`` keeps existing selections when one row is clicked.  With
        # ``extended``, a normal click silently collapsed "Select all" to one
        # dataset, which made an apparently full study run only that row.
        self.dataset_list = __import__("tkinter").Listbox(datasets, selectmode="multiple", height=11)
        self.dataset_list.grid(row=0, column=0, sticky="nsew")
        self.dataset_list.bind("<<ListboxSelect>>", self._update_selection_status)
        selection_controls = ttk.Frame(datasets)
        selection_controls.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(selection_controls, text="Select all", command=self._select_all).pack(side="left")
        ttk.Button(selection_controls, text="Clear", command=self._clear_selection).pack(side="left", padx=6)
        ttk.Label(selection_controls, textvariable=self.selection_status).pack(side="left", padx=8)
        run_controls = ttk.Frame(frame)
        run_controls.grid(row=4, column=0, sticky="w", pady=8)
        self.start_all_button = ttk.Button(run_controls, text="Run all datasets", command=self._start_all)
        self.start_all_button.pack(side="left")
        self.start_selected_button = ttk.Button(
            run_controls, text="Run selected only", command=self._start_selected
        )
        self.start_selected_button.pack(side="left", padx=6)
        ttk.Label(frame, textvariable=self.status).grid(row=4, column=1, columnspan=2, sticky="w")
        self.log = ScrolledText(frame, height=14, wrap="word", state="disabled")
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(3, weight=1)
        frame.rowconfigure(5, weight=1)

    def _choose_datasets(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            names = dataset_names(path)
        except Exception as exc:
            messagebox.showerror("Invalid dataset folder", str(exc))
            return
        self.dataset_directory.set(path)
        self.output_directory.set(str(Path(path).parent / "ML_Results"))
        self.dataset_list.delete(0, "end")
        for name in names:
            self.dataset_list.insert("end", name)
        self._select_all()
        self.status.set(f"Loaded {len(names)} datasets.")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_directory.set(path)

    def _select_all(self) -> None:
        self.dataset_list.select_set(0, "end")
        self._update_selection_status()

    def _clear_selection(self) -> None:
        self.dataset_list.selection_clear(0, "end")
        self._update_selection_status()

    def _update_selection_status(self, _event: object | None = None) -> None:
        selected = len(self.dataset_list.curselection())
        total = self.dataset_list.size()
        self.selection_status.set(f"{selected} of {total} selected")

    def _start_all(self) -> None:
        self._start([self.dataset_list.get(index) for index in range(self.dataset_list.size())])

    def _start_selected(self) -> None:
        self._start([self.dataset_list.get(index) for index in self.dataset_list.curselection()])

    def _start(self, selected: list[str]) -> None:
        if self.running:
            return
        if not self.dataset_directory.get() or not self.output_directory.get() or not selected:
            messagebox.showerror("Missing selection", "Choose the dataset folder, output folder and datasets.")
            return
        config = TrainingConfig(
            outer_folds=self.outer_folds.get(), inner_folds=self.inner_folds.get(),
            random_candidates=self.candidates.get(), max_iterations=self.max_iterations.get(),
        )
        self.running = True
        self.start_all_button.configure(state="disabled")
        self.start_selected_button.configure(state="disabled")
        self.status.set(f"Training {len(selected)} dataset(s)…")
        thread = threading.Thread(
            target=self._worker,
            args=(config, selected, self.dataset_directory.get(), self.output_directory.get(), self.resume.get()),
            daemon=True,
        )
        thread.start()
        self.root.after(100, self._poll)

    def _worker(self, config: TrainingConfig, selected: list[str], dataset_directory: str,
                output_directory: str, resume: bool) -> None:
        try:
            result = run_training(
                dataset_directory, output_directory, config,
                selected, resume, lambda message: self.events.put(("log", message)),
            )
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._append(str(value))
                elif kind == "done":
                    self.running = False
                    self.start_all_button.configure(state="normal")
                    self.start_selected_button.configure(state="normal")
                    self.status.set(f"Complete: {value.run_directory}")
                    messagebox.showinfo("ML study complete", f"Summary: {value.summary_csv}")
                else:
                    self.running = False
                    self.start_all_button.configure(state="normal")
                    self.start_selected_button.configure(state="normal")
                    self.status.set("Training failed.")
                    messagebox.showerror("Training failed", str(value))
        except queue.Empty:
            pass
        if self.running:
            self.root.after(200, self._poll)

    def _append(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def launch() -> None:
    root = Tk()
    MLTrainingApp(root)
    root.mainloop()

