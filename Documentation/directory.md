pipeline/
│
├── nuke/
│   ├── init.py              # Adds scripts & gizmos to sys.path
│   ├── menu.py              # Adds tools to Nuke menu
│   │
│   ├── scripts/             # All Nuke python tools
│   │   ├── matte2Roto/
│   │   │   ├── __init__.py
│   │   │   ├── matte_to_shapes.py
│   │   │   ├── utils.py
│   │   │   └── version.py
│   │   └── rgb2Roto/
│   │       ├── __init__.py
│   │       └── rgb_to_shapes.py
│   │
│   ├── gizmos/              # Nuke gizmos (optional)
│   └── icons/               # Shelf/menu icons (optional)
│
├── python/                  # General Python utilities
│
├── tests/                   # Unit tests for logic
│   ├── test_align.py
│   └── test_resample.py
│
├── requirements.txt         # Python dependencies
└── README.md


# workflow
1. Nuke loads init.py
This adds the scripts folder to Python PATH.

2. Nuke loads menu.py
This adds menu buttons and hot-reload support

3. Each tool exposes a run function
e.g.
matte2Roto/matte_to_shapes.py -> run()