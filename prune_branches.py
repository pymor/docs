#!/usr/bin/env python3

import contextlib
import shutil
import sys
import re
import os
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

@contextlib.contextmanager
def remember_cwd(dirname):
    curdir = os.getcwd()
    try:
        os.chdir(dirname)
        yield curdir
    finally:
        os.chdir(curdir)


def _get_pymor_branches():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tp:
        os.chdir(tp)
        subprocess.check_call(['git', 'clone', 'https://github.com/pymor/pymor.git', 'pymor'])
        os.chdir('pymor')
        branches = [b.replace('origin/', '') for b in
                    subprocess.check_output(['git', 'branch', '-r'], universal_newlines=True).split()]
        tags = subprocess.check_output(['git', 'tag', '-l'], universal_newlines=True).split()
        return branches, tags


def _update_refs():
    subprocess.check_call(['git', 'update-ref', '-d', 'refs/original/refs/heads/master'])


def _prune_branch(branches):
    os.chdir(ROOT)
    branches = ' '.join(branches)
    env = os.environ.copy()
    env['FILTER_BRANCH_SQUELCH_WARNING=1'] = "1"
    cmd = ['git', 'filter-branch', '-f', '--tree-filter', rf'rm -rf {branches}', '--prune-empty', 'HEAD']
    subprocess.check_call(cmd, universal_newlines=True, env=env)
    _update_refs()


def _get_to_prune_branches():
    branches, tags = _get_pymor_branches()
    subs = [d.name for d in os.scandir(ROOT) if d.is_dir()]
    return [s for s in subs if s not in branches and s not in tags]


dels = _get_to_prune_branches()
_prune_branch(dels)
subprocess.check_call(['echo', 'git', 'gc', '--aggressive'])
subprocess.check_call(['echo', 'git', 'push', 'origin','-f'])
