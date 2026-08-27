# Grid-plane pins of the example models

One JSON file per gallery script (`<dir>__<stem>.json`): the grid planes of
its first mesh with their provenance (`mesh.planes.as_dict()`), plus node
counts and cell sizes.  `tests/unit/test_gallery_planes.py` rebuilds the first
mesh of every script and compares.  A mesher change that adds, removes or
moves a plane fails there with a per-axis diff — inspect it, then regenerate:

    MAGNELIO_UPDATE_PLANE_PINS=1 python -m pytest tests/unit/test_gallery_planes.py
