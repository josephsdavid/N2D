#!/usr/bin/env nix-shell 
#! nix-shell -i bash -p python37 python37Packages.twine python37Packages.wheel
#
SOURCE_DATE_EPOCH=$(date +%s)

python setup.py sdist bdist_wheel

twine upload dist/*



