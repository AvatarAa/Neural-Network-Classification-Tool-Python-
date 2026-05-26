"""Launcher for the Iris neural network experiment.

The implementation lives in src/IrisNeuralNetwork.py so the repository can keep
source code organized while preserving the original run command:

    python IrisNeuralNetwork.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "src" / "IrisNeuralNetwork.py"
    runpy.run_path(str(script_path), run_name="__main__")
