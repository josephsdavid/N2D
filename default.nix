let
  pkgs = import <nixpkgs> {};
  tfpkgs = import (
    builtins.fetchGit {
      name = "nixos-tensorflow-2";
      url = https://github.com/nixos/nixpkgs;
      ref = "d59b4d07045418bae85a9bdbfdb86d60bc1640bc";}) {};

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

in 
pkgs.stdenv.mkDerivation rec {
  name = "n2d-env";
  env = pkgs.buildEnv {name = name; paths = buildInputs;};
  builder = builtins.toFile "builder.sh" ''
    source $stdenv/setup; ln -s $env $out
  '';

  buildInputs = [
   pkgs.python37
   tfpkgs.python37Packages.tensorflowWithCuda
   tfpkgs.python37Packages.tensorflow-tensorboard
   pkgs.python37Packages.scikitlearn
   pkgs.python37Packages.seaborn
   pkgs.python37Packages.matplotlib
   pkgs.python37Packages.Keras
   pkgs.python37Packages.h5py
   pkgs.python37Packages.scipy
   pkgs.python37Packages.numba
   umap
   n2d
  ];
}
