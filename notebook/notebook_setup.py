from pathlib import Path
import sys
# Define the bsae direct
BASE_DIR = Path(__file__).resolve().parent.parent

# Add the base directory to the system path

if BASE_DIR not in sys.path:

    sys.path.append(str(BASE_DIR))


