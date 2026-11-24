{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "victory-linux";

  packages = with pkgs; [
    rustup
    cargo-xwin
    vulkan-loader
    vulkan-tools
    pkg-config
    clang
    llvm
    lld
    python3
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # Prefer a GE-Proton compatibility tool already installed by Steam.
    if [ -z "''${GE_PROTON:-}" ]; then
      for candidate in \
        "$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton-nix" \
        "$HOME/.local/share/Steam/compatibilitytools.d"/GE-Proton* \
        "$HOME/.steam/root/compatibilitytools.d"/GE-Proton*; do
        if [ -x "$candidate/proton" ]; then
          export GE_PROTON="$candidate"
          break
        fi
      done
    fi
    # Prefer an installed GE-Proton compatibility tool.
    export WINEDLLOVERRIDES="''${WINEDLLOVERRIDES:-d3d12=n,b;d3d12core=n,b;dxgi=n,b}"
    echo "VIctory Linux shell: cargo xwin build --release -p victory --target x86_64-pc-windows-msvc"
    if [ -n "''${GE_PROTON:-}" ]; then
      echo "GE-Proton: $GE_PROTON (launch with ./scripts/run-ge-proton.sh)"
    else
      echo "GE-Proton not found; set GE_PROTON to its installation directory."
    fi
    echo "Rust uses rust-toolchain.toml (rustup may download it on first use)."
  '';
}
