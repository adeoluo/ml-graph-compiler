# Records docs/demo/optimize.gif from a real run: `vhs docs/demo/optimize.tape` from the repo root.
Output docs/demo/optimize.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 900
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' M=../.bench-data/ml-graph-compiler/models && uv sync -q --inexact && source .venv/bin/activate && clear` Enter
Sleep 8s
Show
Type `mgc optimize $M/mobilenetv2.onnx -O O3` Enter
Sleep 10s
