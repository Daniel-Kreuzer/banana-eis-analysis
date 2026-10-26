"""Standalone GUI for generating model-specific ML datasets."""

from __future__ import annotations

from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .builder import build_datasets


class MLPrepApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Fruit EIS ML dataset preparation")
        root.geometry("980x650")
        self.database = StringVar()
        self.output_directory = StringVar()
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Fitting database").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.database).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._choose_database).grid(row=0, column=2)
        ttk.Label(frame, text="Output folder").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.output_directory).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._choose_output).grid(row=1, column=2)
        ttk.Button(frame, text="Build ML datasets", command=self._build_datasets).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 10)
        )
        self.output = ScrolledText(frame, wrap="word", height=28)
        self.output.grid(row=3, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(3, weight=1)
        self._set_output(
            "This creates eight circuit-model datasets and one raw-spectrum dataset, each with "
            "full and practical physical-feature variants. Existing circuit fits are not modified."
        )

    def _choose_database(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("SQLite database", "*.sqlite"),))
        if path:
            self.database.set(path)
            self.output_directory.set(str(Path(path).parent / "ML_Datasets"))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_directory.set(path)

    def _build_datasets(self) -> None:
        if not self.database.get() or not self.output_directory.get():
            messagebox.showerror("Missing path", "Choose the fitting database and output folder.")
            return
        try:
            result = build_datasets(self.database.get(), self.output_directory.get())
        except Exception as exc:
            messagebox.showerror("Dataset build failed", str(exc))
            return
        lines = [
            "BUILD COMPLETE", "",
            f"Circuit-model rows in master table: {result.circuit_rows}",
            f"Raw-spectrum rows: {result.raw_rows}",
            f"Output database: {result.sqlite_path}",
            f"Manifest: {result.manifest_path}", "", "Boundary-hit rows:",
        ]
        lines.extend(f"  {name}: {count}" for name, count in result.boundary_counts.items())
        self._set_output("\n".join(lines))
        messagebox.showinfo("Build complete", "The ML datasets were created successfully.")

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def launch() -> None:
    root = Tk()
    MLPrepApp(root)
    root.mainloop()
