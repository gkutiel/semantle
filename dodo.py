from doit.tools import run_once

tokens_tar = 'wiki.tokenize.tar'


def task_download_tokens():
    return {
        'targets': [tokens_tar],
        'uptodate': [run_once],
        'actions': [
            f'''wget \
                "https://drive.google.com/file/d/1ysGHa2XsqEzAcs-guFS-05b66bsdSwyU/view?usp=drive_link&export=download&confirm=t" \
                -O {tokens_tar}''']}


def task_extract_tokens():
    return {
        'file_dep': [tokens_tar],
        'actions': [f'dtrx -rf {tokens_tar}']}
