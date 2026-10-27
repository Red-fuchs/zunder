#!/bin/sh
# Metal shaders (mlx) need Xcode's `metal`; CommandLineTools alone fails with "unable to spawn process 'metal'".
cd "$(dirname "$0")"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
swift build -c release "$@" && echo "ok: .build/release/zunder (keep mlx-swift_Cmlx.bundle next to the binary)"
