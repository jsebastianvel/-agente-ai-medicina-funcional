import sys
from pathlib import Path

# Makes "import agente_ai" work for pytest without an editable install (which
# fails locally on this machine - see project memory on the OneDrive issue).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
