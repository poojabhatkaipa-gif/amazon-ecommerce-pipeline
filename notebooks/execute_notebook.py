"""Execute a notebook in-process (no Jupyter kernel / sockets) and store outputs in it.

    python notebooks/execute_notebook.py notebooks/eda.ipynb
Normal alternative:  jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
"""
import ast
import os
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path = Path(sys.argv[1]).resolve()
os.chdir(path.parent)
nb = nbformat.read(path, 4)
shell = InteractiveShell.instance()
count = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    outs = []
    tree = ast.parse(cell.source)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.unparse(tree.body[-1]); tree.body = tree.body[:-1]
    body = ast.unparse(tree)
    with capture_output() as io:
        r = shell.run_cell(body, store_history=False)
        if r.success and last:
            r2 = shell.run_cell(last, store_history=False)
            if r2.success and r2.result is not None:
                data, meta = shell.display_formatter.format(r2.result)
                outs.append(nbformat.v4.new_output("execute_result", data=data, metadata=meta, execution_count=count))
    if not r.success:
        raise SystemExit(f"cell {count} failed: {r.error_in_exec or r.error_before_exec}")
    pre = []
    if io.stdout:
        pre.append(nbformat.v4.new_output("stream", name="stdout", text=io.stdout))
    for o in io.outputs:
        pre.append(nbformat.v4.new_output("display_data", data=o.data, metadata=o.metadata))
    cell.outputs = pre + outs
    cell.execution_count = count
nbformat.write(nb, path)
print(f"executed {count} code cells -> {path.name}")
