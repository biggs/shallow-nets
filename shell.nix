{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/292e674bb11a7962a1d597420be914c48f3f8e10.tar.gz") {} }:
# Import the first version flax is included in. Bump this to get newer versions (maybe).

with pkgs;

mkShell {
  buildInputs = with python3Packages; [
    tensorflow
    # flax
    matplotlib
    scikit-learn
    numpy
    jax
    (jaxlib.override { cudaSupport = true; })
  ];
}
