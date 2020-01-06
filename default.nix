let
  pkgs = import <nixpkgs> {};

  umap = pkgs.callPackage ./nix/umap.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  n2d = pkgs.callPackage ./nix/n2d.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    pythonSource = pkgs.python37Packages;
    umapVar = umap;
  };

in with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "n2d-env";
  env = buildEnv {name = name; paths = buildInputs;};
  builder = builtins.toFile "builder.sh" ''
    source $stdenv/setup; ln -s $env $out
  '';

  buildInputs = [
    python37
    python37Packages.tensorflowWithCuda
    python37Packages.scikitlearn
    python37Packages.seaborn
    python37Packages.matplotlib
    python37Packages.Keras
    python37Packages.h5py
    python37Packages.scipy
    python37Packages.numba
    umap
    n2d
  ];
}
