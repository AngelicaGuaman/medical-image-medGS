import pyvista as pv
from pathlib import Path

colors = {
    
    "class_1.ply": "lightpink",

    #imagen derecha tiene la clase 2 y 3
    "class_2.ply": "gold",
    "class_3.ply": "cyan",

    #imagen derecha tiene la clase 4 y 5
    "class_4.ply": "lightgreen",
    "class_5.ply": "lightblue",
}

plotter = pv.Plotter()

for path in Path("meshes").glob("*.ply"):
    plotter.add_mesh(
        pv.read(path),
        color=colors.get(path.name, "white"),
        smooth_shading=True,
        opacity=0.85,
    )

plotter.add_axes()
plotter.show_grid()
plotter.show()