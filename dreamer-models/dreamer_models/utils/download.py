import os
import shutil
import subprocess
import sys

python = sys.executable


def run(command, folder_path=None, try_until_success=True, **kwargs):
    run_kwargs = {
        'args': command,
        'shell': True,
        'env': os.environ,
        'encoding': 'utf8',
        'errors': 'ignore',
        'capture_output': True,
    }
    run_kwargs.update(kwargs)
    while True:
        print(command)
        result = subprocess.run(**run_kwargs)
        if result.returncode == 0:
            break
        else:
            if try_until_success:
                print(result.stderr)
                if folder_path is not None and os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
            else:
                raise ValueError(result.stderr)


def get_repo_dir():
    root_dir = os.path.abspath(__file__).split('giga_models')[0]
    repo_dir = os.path.join(root_dir, '3rdparty')
    os.makedirs(repo_dir, exist_ok=True)
    return os.environ.get('GIGA_MODELS_REPO_DIR', repo_dir)


def git_clone(url_path, repo_name=None, prefix='https://mirror.ghproxy.com/', force=False, **kwargs):
    if repo_name is None:
        repo_name = url_path.split('/')[-1]
        assert repo_name.endswith('.git')
        repo_name = repo_name[:-4]
    repo_path = os.path.join(get_repo_dir(), repo_name)
    if os.path.exists(repo_path):
        if force:
            shutil.rmtree(repo_path)
        else:
            return repo_path
    if prefix is not None:  # ref: https://ghproxy.com/
        url_path = prefix + url_path
    run(f'git clone "{url_path}" "{repo_path}"', folder_path=repo_path, try_until_success=True, **kwargs)
    return repo_path


def run_pip(command, **kwargs):
    run(f'{python} -m pip {command}', try_until_success=False, **kwargs)


def run_python(command, **kwargs):
    run(f'{python} {command}', try_until_success=False, **kwargs)
