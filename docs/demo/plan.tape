# Records docs/demo/plan.gif from a real run: `vhs docs/demo/plan.tape` from the repo root.
Output docs/demo/plan.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' M=../.bench-data/ml-graph-compiler/models && uv sync -q --inexact && source .venv/bin/activate && clear` Enter
Sleep 15s
Show
Type `mgc plan $M/resnet50.onnx` Enter
Sleep 15s
