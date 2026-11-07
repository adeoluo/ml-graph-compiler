# Records docs/demo/mlir.gif from a real run: `vhs docs/demo/mlir.tape` from the repo root.
Output docs/demo/mlir.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 640
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' M=../.bench-data/ml-graph-compiler/models && uv sync -q --inexact && source .venv/bin/activate && clear` Enter
Sleep 8s
Show
Type `mgc mlir tiny-gpt --elide-constants | head -n 14 | cut -c 1-140` Enter
Sleep 3s
Type `mgc mlir tiny-gpt --verify -o tiny-gpt.mlir` Enter
Sleep 5s
Hide
Type `rm -f tiny-gpt.mlir` Enter
Sleep 500ms
