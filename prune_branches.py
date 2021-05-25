#!/usr/bin/env python3

import contextlib
import shutil
import sys
import re
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from pprint import pformat

ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
BLOCKLIST = [".git-rewrite", ".git", "latest", ".binder", ".github"]
KEEP_BRANCHES = re.compile("20\d\d(-|\.)\d(-|\.).")
LOCAL_ONLY_BRANCHES = re.compile("^github-push-.*")

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
        branches = [b for b in branches if "HEAD" not in b]
        tags = subprocess.check_output(['git', 'tag', '-l'], universal_newlines=True).split()
        return branches, tags


def _update_refs():
    subprocess.check_call(['git', 'update-ref', '-d', 'refs/original/refs/heads/main'])


def _del_remote_branches(branches):
    for branch in [b for b in branches if not LOCAL_ONLY_BRANCHES.match(b)]:
        try:
            subprocess.check_call(['git', 'push', 'origin', '--delete', branch])
        except:
            print(f"Failed to delete {branch} in origin")


def _prune_branches(branches):
    os.chdir(ROOT)
    str_branches = ' '.join(branches)
    print(f'pruning: {branches}')
    env = os.environ.copy()
    env['FILTER_BRANCH_SQUELCH_WARNING'] = "1"
    cmd = ['git', 'filter-branch', '-f', '--tree-filter', rf'rm -rf {str_branches}', '--prune-empty', 'HEAD']
    subprocess.check_call(cmd, universal_newlines=True, env=env)
    _del_remote_branches(branches)
    _update_refs()


def _get_to_prune_branches():
    branches, tags = _get_pymor_branches()
    print(f'Active branches: {" ".join(branches)}\nTags: {" ".join(tags)}')
    tags.extend([t.replace(".", "-") for t in tags])
    subs = [d.name for d in os.scandir(ROOT) if d.is_dir()]
    def _ok(br):
        if KEEP_BRANCHES.match(br):
            return False
        return br not in branches and br not in tags and br not in BLOCKLIST
    return [s for s in subs if _ok(s)]


dels = _get_to_prune_branches()
_prune_branches(dels)
print('''Now execute
git gc --aggressive
git push origin --force
''')
