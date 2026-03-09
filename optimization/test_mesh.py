import meshio
import numpy as np

m = meshio.read("InputData/wingbox.msh")

points = m.points[:, :3]

# gom tất cả triangle blocks + physical tags
tri_blocks = [c.data for c in m.cells if c.type == "triangle"]

phys_data = None
for key in ["gmsh:physical", "cell_tags"]:
    if key in m.cell_data:
        phys_data = m.cell_data[key]
        break

tag_blocks = [d for d, c in zip(phys_data, m.cells) if c.type == "triangle"]

def nodes_of_tag(tag):
    out = []
    for cells, tags in zip(tri_blocks, tag_blocks):
        mask = (tags == tag)
        if np.any(mask):
            out.append(cells[mask].ravel())
    if not out:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(out))

TAG_UPPER, TAG_LOWER = 14, 15
TAG_RIB0, TAG_RIB750, TAG_RIB1500, TAG_RIB2250, TAG_RIB3000 = 38, 39, 40, 41, 42
TAG_MAINSPAR, TAG_TAILSPAR = 43, 44

skin_nodes = np.unique(np.concatenate([
    nodes_of_tag(TAG_UPPER), nodes_of_tag(TAG_LOWER)
]))

for name, tag in [
    ("RIB0", TAG_RIB0),
    ("RIB750", TAG_RIB750),
    ("RIB1500", TAG_RIB1500),
    ("RIB2250", TAG_RIB2250),
    ("RIB3000", TAG_RIB3000),
    ("MAINSPAR", TAG_MAINSPAR),
    ("TAILSPAR", TAG_TAILSPAR),
]:
    n = nodes_of_tag(tag)
    shared = np.intersect1d(skin_nodes, n)
    print(f"{name:10s}: total nodes = {len(n):6d}, shared with skin = {len(shared):6d}")