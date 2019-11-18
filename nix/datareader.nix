{lib, buildPythonPackage, fetchPypi, pythonSource}:


buildPythonPackage rec {
  pname = "pandas-datareader";
  version = "0.8.1" ;

  src = fetchPypi {
    inherit pname version;
    sha256 ="14kskislpv1psk8c2xz4qik4n228iiz6fmxcw41y3dc4dhb6nxmq";
  };

  doCheck = false;
  buildInputs = with pythonSource; [pandas lxml requests] ;
  propogatedBuildInputs =  with pythonSource;[pandas lxml requests];
}

