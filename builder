#!/usr/bin/env nix-shell 
#! nix-shell -i bash -p python37 python37Packages.twine

python setup.py sdist bdist

twine upload dist/*



