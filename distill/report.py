"""The end-of-run spreadsheet.

`metrics.json` is the machine-readable record; this is the one you open.  It holds the
same numbers the summary logs, laid out as a table per sheet:

  accuracy         one row per task -- accuracy over every class seen so far, and the
                   triangular breakdown over each individual task
  summary          final accuracy, average incremental accuracy, forgetting
  bias correction  what stage 2 fitted at each task, and validation either side of it

openpyxl is optional: without it the same three tables are written as CSV files
alongside, so a run never fails for want of a spreadsheet library.
"""

import csv
import logging
import os

log = logging.getLogger(__name__)


def _accuracy_rows(results, tasks, name):
    """The triangular table: after task N, how each task seen so far is doing."""
    model = results[name]
    per_task = model["per_task_accuracy"]
    seen = model["seen_accuracy"]
    width = max((len(row) for row in per_task), default=0)

    header = (["after task", "classes added", "classes seen", "all seen (%)"]
              + [f"T{i + 1} (%)" for i in range(width)])
    rows = []
    for t, (acc, row) in enumerate(zip(seen, per_task)):
        classes = tasks[t] if t < len(tasks) else []
        seen_count = sum(len(tasks[i]) for i in range(min(t + 1, len(tasks))))
        # Blanks, not zeros, for tasks that had not arrived yet -- a zero here would
        # read as "got them all wrong" rather than "was not asked".
        padded = list(row) + [None] * (width - len(row))
        rows.append([t + 1, " ".join(str(c) for c in classes), seen_count, acc] + padded)
    return header, rows


def _summary_rows(results, name):
    model = results[name]
    return ["metric", "value"], [
        ["dataset", results.get("dataset")],
        ["architecture", results.get("arch")],
        ["stage-1 loss", results.get("loss")],
        ["bias-correction mode", results.get("bic_mode")],
        ["tasks", len(results.get("tasks", []))],
        ["final accuracy (%)", model["final"]],
        ["average incremental accuracy (%)", model["average_incremental"]],
        ["average forgetting (points)", model["forgetting"]],
    ]


def _bias_rows(results):
    """One row per task that ran stage 2; the first task has no old classes and is absent."""
    header = ["task", "mode", "alpha", "beta", "b: new-class mean", "b: old-class mean",
              "val before (%)", "before new (%)", "before old (%)",
              "val after (%)", "after new (%)", "after old (%)"]
    rows = []
    for rec in results.get("bias_correction", []):
        bias, num_old = rec.get("bias"), rec.get("num_old", 0)
        new_mean = old_mean = None
        if bias:
            new_mean = sum(bias[num_old:]) / max(len(bias) - num_old, 1)
            old_mean = sum(bias[:num_old]) / num_old if num_old else None
        before, after = rec.get("val_before", {}), rec.get("val_after", {})
        rows.append([
            rec.get("task"), rec.get("mode"), rec.get("alpha"), rec.get("beta"),
            new_mean, old_mean,
            before.get("acc"), before.get("new_acc"), before.get("old_acc"),
            after.get("acc"), after.get("new_acc"), after.get("old_acc"),
        ])
    return header, rows


def build_tables(results, name="model"):
    """(sheet name, header, rows) for every table, in the order they should appear."""
    tasks = results.get("tasks", [])
    tables = [("accuracy", *_accuracy_rows(results, tasks, name)),
              ("summary", *_summary_rows(results, name))]
    bias_header, bias_rows = _bias_rows(results)
    if bias_rows:
        tables.append(("bias correction", bias_header, bias_rows))
    return tables


def _write_csv(tables, save_dir, stem):
    paths = []
    for sheet, header, rows in tables:
        path = os.path.join(save_dir, f"{stem}_{sheet.replace(' ', '_')}.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        paths.append(path)
    return paths


def write_report(results, save_dir, name="model", stem="accuracy"):
    """Write the spreadsheet; no-op without a save dir.  Returns the path written."""
    if not save_dir:
        return None
    if name not in results:
        log.debug("no metrics recorded for %r, skipping the spreadsheet", name)
        return None

    os.makedirs(save_dir, exist_ok=True)
    tables = build_tables(results, name)

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        paths = _write_csv(tables, save_dir, stem)
        log.info("openpyxl is not installed; wrote %d CSV files instead of a workbook",
                 len(paths))
        return paths[0] if paths else None

    book = Workbook()
    book.remove(book.active)  # drop the default empty sheet
    for sheet_name, header, rows in tables:
        sheet = book.create_sheet(sheet_name)
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append(row)
        sheet.freeze_panes = "A2"
        for i, title in enumerate(header, start=1):
            sheet.column_dimensions[sheet.cell(row=1, column=i).column_letter].width = \
                max(len(str(title)) + 2, 12)

    path = os.path.join(save_dir, f"{stem}.xlsx")
    book.save(path)
    log.info("wrote %s", path)
    return path
