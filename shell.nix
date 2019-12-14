let
  pkgs = import <nixpkgs> {};

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
    buildInputs = with pkgs; [
      python37
      python37Packages.numpy
      python37Packages.scikitlearn
      python37Packages.numba
      zip
      python37Packages.scipy
      python37Packages.pip
      python37Packages.pandas
      python37Packages.seaborn
      python37Packages.h5py
      python37Packages.tensorflowWithCuda
      python37Packages.tensorflow-tensorboard
      python37Packages.tensorflow-probability
      umap
      python37Packages.pillow
      python37Packages.matplotlib
      python37Packages.Keras
      python37Packages.virtualenv
      python37Packages.twine
      python37Packages.wheel
      n2d
      #python37Packages.hdbscan
      python37Packages.sphinx
      python37Packages.recommonmark
      python37Packages.sphinx_rtd_theme
      python37Packages.ipython
      datareader
      transforms3d
      python37Packages.statsmodels
    ];
    shellHook = ''
      export SOURCE_DATE_EPOCH=$(date +%s) 
      '';

  }
