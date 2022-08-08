from doit.tools import run_once

tokens_tar = 'wiki.tokenize.tar'


def task_download_tokens():
    return {
        'targets': [tokens_tar],
        'uptodate': [run_once],
        'actions': [
            f'''wget \
                "https://drive.google.com/u/0/uc?id=1sJi6qElF9ex-zCHagLh4CNgZLohzZEz5&export=download&confirm=t&uuid=1e70a5c4-0fd6-439c-a9ed-bd593815602d" \
                -O {tokens_tar}''']}


def task_extract_tokens():
    return {
        'file_dep': [tokens_tar],
        'actions': [f'dtrx -rf {tokens_tar}']}
