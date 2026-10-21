#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared helper for the scripts that compile CUDA extensions inside the
# isolated pixi environment.
#
# conda-forge splits the CUDA toolkit across several packages and keeps the
# real headers in <prefix>/targets/x86_64-linux/include, while nvcc (a real
# binary in <prefix>/bin) looks for them in <prefix>/include. Torch's
# cpp_extension additionally expects <prefix>/lib64.
#
# merge_cuda_layout merges the two views with symlinks so that a plain
# CUDA_HOME=<prefix> works. It only touches the (git-ignored) pixi env.
# ---------------------------------------------------------------------------

#: The GPU architectures the extensions are compiled for.
#:   8.6 - NVIDIA A40 (Ampere)
#:   8.9 - RTX 4060 Ti (Ada Lovelace)
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-8.6;8.9}"

merge_cuda_layout() {
  local prefix="${1:?prefix required}"
  local targets="$prefix/targets/x86_64-linux"

  if [ ! -d "$targets" ]; then
    echo "    (no conda-forge CUDA layout at $targets - using the system CUDA)"
    return 0
  fi

  echo "    merging the conda-forge CUDA layout into a single toolkit root"
  for f in "$targets"/include/*; do
    b="$(basename "$f")"
    [ -e "$prefix/include/$b" ] || ln -sfn "$f" "$prefix/include/$b"
  done
  for f in "$targets"/lib/*; do
    b="$(basename "$f")"
    [ -e "$prefix/lib/$b" ] || ln -sfn "$f" "$prefix/lib/$b"
  done
  [ -e "$prefix/lib64" ] || ln -sfn "$prefix/lib" "$prefix/lib64"
  [ -e "$prefix/nvvm" ] || ln -sfn "$targets/nvvm" "$prefix/nvvm"
}

#: site-packages directory of the active interpreter
site_packages_dir() {
  python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
}
