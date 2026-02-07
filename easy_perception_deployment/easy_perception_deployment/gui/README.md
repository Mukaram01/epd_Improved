## What Is This?
This document contains instructions on how to run the Graphical User Interface of **easy_perception_deployment**.

## Dependencies (Ubuntu 22.04 / ROS 2 Humble)
Ubuntu 22.04 ships with **Python 3.10**, which is the supported version for ROS 2 Humble. Use the system Python whenever possible to avoid Qt/ROS mismatches. Anaconda is **optional** and generally **not recommended** for the GUI on Humble, because it can introduce incompatible Qt bindings. If you choose to use Anaconda, ensure it provides Python 3.10 and PyQt5/Qt5 and does not override ROS environment packages.

Required dependencies:
1. **Python 3.10** (system default on Ubuntu 22.04)
2. **Qt binding: PyQt5 (Qt5)**
3. **matplotlib**
4. **torch / torchvision**

### Qt binding (PyQt5)
Use the Ubuntu package for PyQt5 to stay aligned with ROS 2 Humble’s Qt5 setup:

```bash
sudo apt update
sudo apt install -y python3-pyqt5
```

If you must use pip, install PyQt5 into the same Python 3.10 environment you use for ROS:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install PyQt5
```

### Other non-ROS Python dependencies
Install these into the same Python 3.10 environment (system or virtualenv):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install matplotlib
```

For PyTorch, prefer a build that supports Python 3.10. A typical CPU-only install is:

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you need CUDA, follow the official PyTorch selector to match your CUDA version and Python 3.10.

## Setup
Follow the instructions below to create the virtual environment.
