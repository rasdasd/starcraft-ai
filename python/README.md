# bwbot

Python side of the Brood War bot environment. See the repository root `README.md` for setup.

```
python -m bwbot.run examples.basic_terran --frame-skip 2 --speed 0
```

Optional ML extras (none required by the core):

```
pip install -e ".[torch]"      # PyTorch
pip install -e ".[onnx]"       # onnxruntime + onnx
pip install -e ".[tensorrt]"   # TensorRT python bindings (needs the NVIDIA SDK)
```
