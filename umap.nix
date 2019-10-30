{lib, buildPythonPackage, fetchPypi, pythonSource}:


buildPythonPackage rec {
  pname = "umap-learn";
  version = "0.3.10" ;

  src = fetchPypi {
    inherit pname version;
    sha256 = "02ada2yy6km6zgk2836kg1c97yrcpalvan34p8c57446finnpki1";
  };

  checkInputs = with pythonSource; [nose];
  buildInputs = with pythonSource; [numpy scipy scikitlearn numba] ;
  propogatedBuildInputs =  with pythonSource;[numba];
}

