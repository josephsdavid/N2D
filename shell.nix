let
  pkgs = import <stable> {};

  umap = pkgs.callPackage ./umap.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  n2d = pkgs.callPackage ./n2d.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    pythonSource = pkgs.python37Packages;
    umapVar = umap;
  };
#  # bring in yellowbrick from pypi
#    umap = pkgs.python37.pkgs.buildPythonPackage rec {
#      pname = "umap-learn";
#      version = "0.3.10" ;
#
#      src = pkgs.python37.pkgs.fetchPypi {
#        inherit pname version;
#        sha256 = "02ada2yy6km6zgk2836kg1c97yrcpalvan34p8c57446finnpki1";
#      };
#      #doCheck = false;
#      checkInputs = with pkgs.python37Packages; [nose];
#      buildInputs = with pkgs.python37Packages; [numpy scipy scikitlearn numba] ;
#      propogatedBuildInputs =  with pkgs.python37Packages;[numba];
#    };
#
#    n2d = pkgs.python37.pkgs.buildPythonPackage rec {
#      pname = "n2d";
#      version = "0.0.2";
#      src = ./.;
#      BuildInputs = [
#        pkgs.python37Packages.h5py
#        pkgs.python37Packages.Keras
#        umap
#        pkgs.python37Packages.numpy
#        pkgs.python37Packages.scikitlearn
#        pkgs.python37Packages.numba
#        pkgs.python37Packages.scipy
#        pkgs.python37Packages.pandas
#        pkgs.python37Packages.seaborn
#        pkgs.python37Packages.tensorflowWithCuda
#        pkgs.python37Packages.matplotlib
#      ];
#      PropogatedBuildInputs = [pkgs.python37Packages.h5py pkgs.python37Packages.Keras];
#    };

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
#      python37Packages.pytorch
#      python37Packages.torchvision
      python37Packages.tensorflowWithCuda
      python37Packages.tensorflow-tensorboard
      umap
      python37Packages.pillow
      python37Packages.matplotlib
      python37Packages.Keras
      python37Packages.virtualenv
      python37Packages.twine
      python37Packages.wheel
      n2d
    ];
    shellHook = ''
      export SOURCE_DATE_EPOCH=$(date +%s) # 1980
      '';

  }
