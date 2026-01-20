from IPython.display import HTML, display
import os
import subprocess


def get_notebook_path():
    """Get the current notebook's path relative to the workspace root."""
    try:
        from IPython import get_ipython
        kernel = get_ipython()
        if kernel is not None:
            return kernel.config['IPKernelApp']['connection_file']
    except:
        return None
    return None


def create_nav_bar(current_notebook=None):
    """
    Create a navigation bar with links to all analysis notebooks.
    
    Args:
        current_notebook (str): Name of the current notebook to highlight
    """
    notebooks = {
        'Main Pipeline': 'pgx_cohort_pipeline.ipynb',
        'bupaR Analysis': 'bupaR_analysis/bupaR_pipeline.ipynb',
        'CatBoost R Analysis': 'catboost_analysis/catboost_r.ipynb'
    }
    
    nav_items = []
    for name, path in notebooks.items():
        if name == current_notebook:
            nav_items.append(f'<strong>{name}</strong>')
        else:
            nav_items.append(f'<a href="{path}">{name}</a>')
    
    nav_html = f"""
    <div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: bold;">Navigation:</div>
            <div style="display: flex; gap: 20px;">
                {' | '.join(nav_items)}
            </div>
        </div>
    </div>
    """
    return HTML(nav_html)


def create_section_links(sections):
    """
    Create links to specific sections within the current notebook.
    
    Args:
        sections (dict): Dictionary of section names and their anchor IDs
    """
    section_items = []
    for name, anchor in sections.items():
        section_items.append(f'<a href="#{anchor}">{name}</a>')
    
    section_html = f"""
    <div style="background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Sections:</div>
        <div style="display: flex; gap: 15px;">
            {' | '.join(section_items)}
        </div>
    </div>
    """
    return HTML(section_html)

def add_navigation(current_notebook=None, sections=None):
    """
    Add both navigation bar and section links to the notebook.
    
    Args:
        current_notebook (str): Name of the current notebook
        sections (dict): Dictionary of section names and their anchor IDs
    """
    display(create_nav_bar(current_notebook))
    if sections:
        display(create_section_links(sections))


def sync_to_s3(bucket_path="s3://pgx-repository/pgx-analysis/"):
    """
    Sync local files to S3 bucket, excluding unnecessary files and directories.
    
    Args:
        bucket_path (str): S3 bucket path to sync to
    """
    cmd = [
        "aws", "s3", "sync", ".",
        bucket_path,
        "--exact-timestamps",
        "--exclude", "*",
        "--include", "*.ipynb",
        "--include", "*.qmd",
        "--include", "*.py",
        "--include", "*[^.].R",
        "--include", "*.png",
        "--exclude", "*.Rproj.user/*",
        "--exclude", "*.Rproj.user",
        "--exclude", "*.trunk/*",
        "--exclude", "*.trunk",
        "--exclude", "*.venv/*",
        "--exclude", "*.venv",
        "--exclude", "*.venv311/*",
        "--exclude", "*.venv311",
        "--exclude", "*.venv310/*",
        "--exclude", "*.venv310",
        "--exclude", "*.ipynb_checkpoints/*",
        "--exclude", "*.ipynb_checkpoints"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Successfully synced files to S3")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing to S3: {e}")


def sync_from_s3(bucket_path="s3://pgx-repository/pgx-analysis/"):
    """
    Sync files from S3 bucket to local directory, excluding unnecessary files and directories.
    
    Args:
        bucket_path (str): S3 bucket path to sync from
    """
    cmd = [
        "aws", "s3", "sync",
        bucket_path, ".",
        "--exact-timestamps",
        "--exclude", "*",
        "--include", "*.ipynb",
        "--include", "*.qmd",
        "--include", "*.py",
        "--include", "*[^.].R",
        "--include", "*.png",
        "--exclude", "*.Rproj.user/*",
        "--exclude", "*.Rproj.user",
        "--exclude", "*.trunk/*",
        "--exclude", "*.trunk",
        "--exclude", "*.venv/*",
        "--exclude", "*.venv",
        "--exclude", "*.venv311/*",
        "--exclude", "*.venv311",
        "--exclude", "*.venv310/*",
        "--exclude", "*.venv310",
        "--exclude", "*.ipynb_checkpoints/*",
        "--exclude", "*.ipynb_checkpoints"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Successfully synced files from S3")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing from S3: {e}")

