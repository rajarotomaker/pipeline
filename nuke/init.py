import os
import sys

nuke_root = os.path.dirname(__file__)

#Build full paths to scripts and gizmos folders
scripts_path = os.path.join(nuke_root, "scripts")

# Add to Python path so Nuke can import them
for p in [scripts_path]:
    if p not in sys.path:
        sys.path.append(p)