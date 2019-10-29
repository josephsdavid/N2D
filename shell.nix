let
  pkgs = import <stable> {};
  # bring in yellowbrick from pypi
    umap = pkgs.python37.pkgs.buildPythonPackage rec {
      pname = "umap-learn";
      version = "0.3.10" ;

      src = pkgs.python37.pkgs.fetchPypi {
        inherit pname version;
        sha256 = "02ada2yy6km6zgk2836kg1c97yrcpalvan34p8c57446finnpki1";
      };
      #doCheck = false;
      checkInputs = with pkgs.python37Packages; [nose];
      buildInputs = with pkgs.python37Packages; [numpy scipy scikitlearn numba] ;
      propogatedBuildInputs =  with pkgs.python37Packages;[numba];
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
#      python37Packages.pytorch
#      python37Packages.torchvision
      python37Packages.tensorflowWithCuda
      python37Packages.tensorflow-tensorboard
      umap
      python37Packages.pillow
      python37Packages.matplotlib
      python37Packages.Keras
    ];
   shellHook = ''
      '';

  }
