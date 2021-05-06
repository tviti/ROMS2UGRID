{ pkgs ? import <nixpkgs> { } }:

let
  myPy3Pkgs = pypkgs: with pypkgs; [
    ipython
    netcdf4
    xarray
    python-language-server
  ];

  myPy3 = pkgs.python3.withPackages myPy3Pkgs;
in pkgs.mkShell rec {
  buildInputs = [ myPy3 ];
}
