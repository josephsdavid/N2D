{lib, buildPythonPackage, pythonSource, packs, umapVar}:

let
  additionalIgnores = "/../examples
/../docs";
in
    buildPythonPackage rec {
      pname = "n2d";
      version = "0.3.1";

    
      src = packs.nix-gitignore.gitignoreSource additionalIgnores ./..;

      buildInputs = [
        pythonSource.h5py
        pythonSource.tensorflow
        pythonSource.scipy
        pythonSource.numpy
        pythonSource.pandas
        pythonSource.seaborn
        pythonSource.matplotlib
        pythonSource.scikitlearn
        pythonSource.numba
        umapVar
      ];
      propogatedBuildInputs = [
        pythonSource.h5py
        pythonSource.tensorflow
        pythonSource.scipy
        pythonSource.numpy
        pythonSource.pandas
        pythonSource.seaborn
        pythonSource.matplotlib
        pythonSource.scikitlearn
        pythonSource.numba
        umapVar
      ];

    }
