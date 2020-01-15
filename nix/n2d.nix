{lib, buildPythonPackage, pythonSource, umapVar}:

    buildPythonPackage rec {
      pname = "n2d";
      version = "0.2.5";
    

      src = ./..;

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
