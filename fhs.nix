# FHS environment for running pip's prebuilt ROCm PyTorch wheels on NixOS without
# enabling nix-ld system-wide. The wheels bundle their own ROCm runtime; this only
# provides the standard system libraries they link against, at FHS paths.
#
#   Interactive:  nix-shell                      (enters bash in the FHS env; activates .venv if present)
#   Scripted:     nix-build fhs.nix -o .fhs-env && ./.fhs-env/bin/fairtune-fhs -c 'source .venv/bin/activate && ...'
{ pkgs ? import <nixpkgs> { } }:
pkgs.buildFHSEnv {
  name = "fairtune-fhs";
  targetPkgs = p: with p; [
    python3
    bash coreutils findutils gnused gawk gnugrep git which procps
    stdenv.cc.cc.lib zlib zstd xz bzip2 libffi openssl
    libdrm numactl elfutils ncurses libGL glib expat
  ];
  profile = ''
    if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
  '';
  # The libdrm bundled in the ROCm torch wheel looks for its GPU-name table here.
  extraBuildCommands = ''
    mkdir -p $out/opt/amdgpu/share/libdrm
    ln -s /usr/share/libdrm/amdgpu.ids $out/opt/amdgpu/share/libdrm/amdgpu.ids
  '';
  runScript = "bash";
}
