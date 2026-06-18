# Jupyter post-save hook - auto-push notebook outputs to S3 on save.
# Append this to ~/.jupyter/jupyter_notebook_config.py on EC2 after
# confirming cursor_setup.py can find the shared project_utility_scripts script.

import subprocess
from notebook.services.contents.filemanager import FileContentsManager

_original_save = FileContentsManager.save


def _patched_save(self, model, path=""):
    result = _original_save(self, model, path)
    if path.endswith(".ipynb"):
        nb_abs_path = self.root_dir + "/" + path
        subprocess.Popen([
            "python",
            "c:/Projects/pgx-analysis/cursor_setup.py",
            "push-outputs",
            nb_abs_path,
        ])
    return result


FileContentsManager.save = _patched_save
