let
  pkgs = import <nixpkgs>{};

  tfpkgs = import (
    builtins.fetchGit {
      name = "nixos-tensorflow-2";
      url = https://github.com/nixos/nixpkgs;
      ref = "d59b4d07045418bae85a9bdbfdb86d60bc1640bc";}) {};

  transforms3d = pkgs.callPackage ./nix/transforms3d.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  umap = pkgs.callPackage ./nix/umap.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  datareader = pkgs.callPackage ./nix/datareader.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  n2d = pkgs.callPackage ./nix/n2d.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    pythonSource = pkgs.python37Packages;
    umapVar = umap;
  };

in
  pkgs.mkShell {
    name = "simpleEnv";
    buildInputs = [
      pkgs.python37
      pkgs.python37Packages.numpy
      pkgs.python37Packages.scikitlearn
      pkgs.python37Packages.numba
      pkgs.zip
      pkgs.python37Packages.scipy
      pkgs.python37Packages.pip
      pkgs.python37Packages.pandas
      pkgs.python37Packages.hdbscan
      pkgs.python37Packages.seaborn
      pkgs.python37Packages.h5py
      tfpkgs.python37Packages.tensorflowWithCuda
      tfpkgs.python37Packages.tensorflow-tensorboard
      tfpkgs.python37Packages.tensorflow-probability
      umap
      pkgs.python37Packages.pillow
      pkgs.python37Packages.matplotlib
      pkgs.python37Packages.virtualenv
      pkgs.python37Packages.twine
      pkgs.python37Packages.wheel
      n2d
      pkgs.python37Packages.sphinx
      pkgs.python37Packages.recommonmark
      pkgs.python37Packages.sphinx_rtd_theme
      pkgs.python37Packages.ipython
      datareader
      transforms3d
      pkgs.python37Packages.statsmodels
    ];
    shellHook = ''
      export SOURCE_DATE_EPOCH=$(date +%s) 
      '';

  }
