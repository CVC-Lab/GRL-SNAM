# Attribution — Austin scene geometry (`volrover_austin_demo.py`)

The `volrover_austin_demo.py` example renders real-world Austin, TX geometry from
a `geometry_bundle` (e.g. the CVC-DBG `austin_south` bundle). That geometry is
derived from public sources and must be credited when you publish renders:

- **Buildings, roads, water** — © **OpenStreetMap** contributors, licensed under
  the **Open Database License (ODbL)**, https://openstreetmap.org/copyright.
  Attribution to OpenStreetMap is **required** for any published render or
  derived work.
- **Terrain elevation** — **SRTM** (NASA/USGS), **U.S. public domain**.

Do **not** ship the Esri `satellite.png` overlay that some bundles carry — it is
proprietary Esri World Imagery and is not redistributable. The loaders in
`grl_snam_lab.scenes` never load it; the demo colors the ground itself.

The bundle geometry itself is not formally licensed in-repo; the above is read
from its generation pipeline (OSM via osmnx/Overpass, SRTM via Open Topo Data).
When in doubt, treat it as ODbL + attribution-required.
