# Records docs/demo/compile.gif from a real run: `vhs docs/demo/compile.tape` from the repo root.
Output docs/demo/compile.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 720
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' M=../.bench-data/ml-graph-compiler/models && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 8s
Show
Type `mgc compile $M/mobilenetv2.onnx --check --bench 50 --threads 4` Enter
Sleep 25s
