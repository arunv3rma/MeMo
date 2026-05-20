import os
if os.environ.get("CARTRIDGES_DIR") is None:
    raise ValueError("CARTRIDGES_DIR is not set. Please set it to the path to the cartridges directory.")

if os.environ.get("CARTRIDGES_OUTPUT_DIR") is None:
    raise ValueError("CARTRIDGES_OUTPUT_DIR is not set. Please set it to the path to the cartridges output directory.")
