"""Small standalone GUI for importing physical and ripeness metadata."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .importer import ImportPreview, import_metadata, infer_study_start, preview_metadata


class MetadataApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Fruit EIS metadata importer")
        root.geometry("980x650")
        root.minsize(780, 520)
        self.database = StringVar()
        self.dimensions = StringVar()
        self.ripeness = StringVar()
        self.study_start = StringVar()
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        fields = (
            ("Fitting database", self.database, self._choose_database),
            ("Dimensions workbook", self.dimensions, self._choose_dimensions),
            ("Ripeness workbook", self.ripeness, self._choose_ripeness),
        )
        for row, (label, variable, command) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(frame, text="Browse…", command=command).grid(row=row, column=2)
        ttk.Label(frame, text="Study Day 1 date (YYYY-MM-DD)").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.study_start, width=16).grid(row=3, column=1, sticky="w", padx=8)
        ttk.Button(frame, text="Infer from captures", command=self._infer_date).grid(row=3, column=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 8))
        ttk.Button(buttons, text="Preview", command=self._preview).pack(side="left")
        ttk.Button(buttons, text="Import/update database", command=self._import).pack(side="left", padx=8)

        self.output = ScrolledText(frame, wrap="word", height=25)
        self.output.grid(row=5, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(5, weight=1)
        self._set_output(
            "Choose circuit_fits.sqlite, BananaDimensions.xlsx and Ripeness.xlsx.\n\n"
            "Preview validates the files without modifying the database. Ripeness observations "
            "without a matching EIS capture are allowed and retained."
        )

    def _choose_database(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("SQLite database", "*.sqlite"), ("All files", "*.*")))
        if path:
            self.database.set(path)
            self._infer_date()

    def _choose_dimensions(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("Excel workbook", "*.xlsx"),))
        if path:
            self.dimensions.set(path)

    def _choose_ripeness(self) -> None:
        path = filedialog.askopenfilename(filetypes=(("Excel workbook", "*.xlsx"),))
        if path:
            self.ripeness.set(path)

    def _infer_date(self) -> None:
        if not self.database.get():
            return
        try:
            self.study_start.set(infer_study_start(self.database.get()).isoformat())
        except Exception as exc:
            messagebox.showerror("Cannot infer date", str(exc))

    def _inputs(self) -> tuple[Path, Path, Path, date]:
        if not all((self.database.get(), self.dimensions.get(), self.ripeness.get(), self.study_start.get())):
            raise ValueError("Choose all three files and enter the Study Day 1 date.")
        return (
            Path(self.database.get()), Path(self.dimensions.get()), Path(self.ripeness.get()),
            date.fromisoformat(self.study_start.get().strip()),
        )

    def _preview(self) -> None:
        try:
            result = preview_metadata(*self._inputs())
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return
        self._show_preview(result, imported=False)

    def _import(self) -> None:
        try:
            result = import_metadata(*self._inputs())
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self._show_preview(result, imported=True)
        messagebox.showinfo("Import complete", "Physical properties and ripeness were updated successfully.")

    def _show_preview(self, result: ImportPreview, imported: bool) -> None:
        heading = "IMPORT COMPLETE" if imported else "PREVIEW — DATABASE NOT MODIFIED"
        lines = [
            heading,
            "",
            f"Banana property rows: {len(result.properties)}",
            f"Ripeness observations: {len(result.ripeness)}",
            f"Ripeness observations without an EIS capture: {len(result.unmatched_ripeness)}",
            f"Capture dates: {', '.join(result.capture_dates)}",
        ]
        if result.unmatched_ripeness:
            lines.extend(("", "Metadata-only ripeness observations:"))
            lines.extend(
                f"  {item.banana_id}, {item.measurement_date} (day {item.study_day}): {item.ripeness_class}"
                for item in result.unmatched_ripeness
            )
        if result.warnings:
            lines.extend(("", "Warnings:"))
            lines.extend(f"  {warning}" for warning in result.warnings)
        self._set_output("\n".join(lines))

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def launch() -> None:
    root = Tk()
    MetadataApp(root)
    root.mainloop()
