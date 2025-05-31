def load_scene_categories_sun(scene_cat_file: str) -> dict:
    """
    Reads sceneCategories.txt lines like:
       ADE_train_00000001 airport_terminal
       ADE_train_00000002 airport_terminal
    Returns a dict: { "ADE_train_00000001": "airport_terminal", ... }
    """
    scene_map = {}
    with open(scene_cat_file, 'r') as f:
        for line in f:
            base, scene_str = line.strip().split()
            scene_map[base] = scene_str
    return scene_map