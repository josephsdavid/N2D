{lib, buildPythonPackage, fetchPypi, pythonSource}:


buildPythonPackage rec {
  pname = "transforms3d";
  version = "0.3.1" ;

  src = fetchPypi {
    inherit pname version;
    sha256 ="0y4dm1xrd9vlrnz5dzym8brww5smzh0ij223h35n394aqybpfk20";
  };

  #doCheck = false;
  checkInputs = with pythonSource; [nose];
  buildInputs = [
    pythonSource.numpy
    pythonSource.sympy
      ] ;
  propogatedBuildInputs = with pythonSource; [numpy] ;
}

