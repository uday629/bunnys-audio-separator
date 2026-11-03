# Third-party notices

Bunnys Audio Separator is built on these open-source projects:

| Component | Use | License |
|---|---|---|
| [Demucs](https://github.com/facebookresearch/demucs) (htdemucs_6s model, Meta AI Research) | AI source separation model, converted to ONNX | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (incl. DirectML / CoreML providers) | Model inference on CPU/GPU | MIT |
| [PyAV](https://github.com/PyAV-Org/PyAV) and the FFmpeg libraries it bundles | Audio decoding | BSD-3-Clause (PyAV), LGPL/GPL (FFmpeg) |
| [NumPy](https://numpy.org) | Signal processing | BSD-3-Clause |
| [Python](https://www.python.org) / [PyInstaller](https://pyinstaller.org) bootloader | Runtime / packaging | PSF / GPL with bootloader exception |

Demucs — MIT License — Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
