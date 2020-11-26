SHELL := /bin/bash

set-dev-env:
	# create local development python virtual env
	# activate local virtual env
	# install python packages requirements for development
	python3 -m venv dev_env && \
	source dev_env/bin/activate && \
	pip install -r requirements-dev.txt

activate-dev:
	# activate local development virtual env
	source dev_env/bin/activate

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install universal-ctags
	# install tags file in .git to avoid having it tracked by git
	ctags -R -f .git/tags deslib/
