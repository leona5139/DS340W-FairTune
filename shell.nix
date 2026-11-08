# `nix-shell` here drops you into the FHS env from fhs.nix (see that file).
{ pkgs ? import <nixpkgs> { } }:
(import ./fhs.nix { inherit pkgs; }).env
