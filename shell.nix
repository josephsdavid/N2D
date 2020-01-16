let
  pkgs = import <nixpkgs>{};

  tfpkgs = import (
    builtins.fetchGit {
      name = "nixos-tensorflow-2";
      url = https://github.com/nixos/nixpkgs;
      ref = "d59b4d07045418bae85a9bdbfdb86d60bc1640bc";}) {};



      umap = with pkgs; callPackage ./nix/umap.nix {
        buildPythonPackage = python37.pkgs.buildPythonPackage;
        fetchPypi = python37.pkgs.fetchPypi;
        pythonSource = python37Packages;
      };

  tf_packages = with tfpkgs; {
    tf = python37Packages.tensorflowWithCuda;
    tf-tb = python37Packages.tensorflow-tensorboard;
  };


  local_packages = {
    umap-learn = umap;
    n2d = pkgs.callPackage ./nix/n2d.nix {
      buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
      pythonSource = pkgs.python37Packages;
      packs = pkgs;
      umapVar = umap;
    };
  };

in
  pkgs.mkShell {
    name = "simpleEnv";
    buildInputs = with pkgs // local_packages // tf_packages; [
      python37
      python37Packages.numpy
      python37Packages.scikitlearn
      python37Packages.numba
      zip
      python37Packages.scipy
      python37Packages.pip
      python37Packages.pandas
      python37Packages.hdbscan
      python37Packages.seaborn
      python37Packages.h5py
      tf
      tf-tb
      umap-learn
      python37Packages.matplotlib
      python37Packages.twine
      python37Packages.wheel
      n2d
      python37Packages.sphinx
      python37Packages.recommonmark
      python37Packages.sphinx_rtd_theme
      python37Packages.ipython
      python37Packages.statsmodels
    ];
    shellHook = ''
      export SOURCE_DATE_EPOCH=$(date +%s) 
      '';

  }
