import os

# Must be set before the first torch import. Allows TFT ops that MPS has not
# implemented to fall back to CPU instead of crashing the training run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
