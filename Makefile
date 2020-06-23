.PHONY: list git_squash prune 

list:
	./makeindex.py

git_squash:
	GIT_EDITOR="sed -i -e 's;Updated\ docs\ for.*;squash\!\ squashed\ docs\ for;g' -e 's;pick;reword;g'" \
		git rebase -i --root -X theirs \
	&& GIT_EDITOR=true git rebase -i --autosquash --root -X theirs

cleanup: git_squash prune index

prune:
	./prune_branches.py
