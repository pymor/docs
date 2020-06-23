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
        return [b.replace('origin/', '') for b in subprocess.check_output(['git', 'branch', '-r'], universal_newlines=True).split()]


def _update_refs():
    subprocess.check_call(['git', 'update-ref', '-d', 'refs/original/refs/heads/master'])


def _prune_branch(branch):
    os.chdir(ROOT)
    cmd = ['git', 'filter-branch', '-f', '--tree-filter', r'rm -rf branches/{}'.format(branch), '--prune-empty', 'HEAD']
    subprocess.check_call(cmd, universal_newlines=True)
    _update_refs()


def _get_to_prune_branches():
    pymor = _get_pymor_branches()
    subs = [d.name for d in os.scandir(ROOT / 'branches') if d.is_dir()]
    return [s for s in subs if s not in pymor]


dels = _get_to_prune_branches()
for b in dels:
    _prune_branch(b)
# subprocess.check_call(['git', 'gc', '--aggressive'])
# subprocess.check_call(['echo', 'git', 'push', 'origin','-f'])
