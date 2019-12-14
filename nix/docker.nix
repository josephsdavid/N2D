let
  pkgs = import <stable> {};
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

  deps = pkgs.dockerTools.buildImage {
    name = deps;
    contents = [
      pkgs.python37
      pkgs.python37Packages.numba
    ];
  };


in pkgs.dockerTools.buildImage {
  name = "n2d";
  contents = [n2d];
}


