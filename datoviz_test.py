import numpy as np
import datoviz as dvz


positions = np.asarray(
    [
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.0, 0.5, 0.0],
    ],
    dtype=np.float32,
    order="C",
)

colors = np.asarray(
    [
        [255, 50, 50, 255],
        [50, 255, 50, 255],
        [50, 50, 255, 255],
    ],
    dtype=np.uint8,
    order="C",
)

diameters = np.asarray(
    [30.0, 30.0, 30.0],
    dtype=np.float32,
    order="C",
)

scene = dvz.dvz_scene()
figure = dvz.dvz_figure(scene, 800, 600, 0)
panel = dvz.dvz_panel_full(figure)

visual = dvz.dvz_point(scene, 0)

dvz.dvz_visual_set_data(
    visual,
    "position",
    positions,
)

dvz.dvz_visual_set_data(
    visual,
    "color",
    colors,
)

dvz.dvz_visual_set_data(
    visual,
    "diameter_px",
    diameters,
)

dvz.dvz_panel_add_visual(
    panel,
    visual,
    None,
)

dvz.run(
    scene,
    figure,
    title="Datoviz Test",
)

dvz.dvz_scene_destroy(scene)