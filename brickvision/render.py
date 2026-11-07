"""Isometric renderer and scene helpers shared by the BrickVision generators.

A scene is a list of axis-aligned bricks on a 30x30 stud plate plus an occupancy map `occ[(x, y, z)] = id`.
`render_scene` -> (RGB image, id mask 'I' holding brick id + 1 and 0 = background, {id: visibility}, projector).
"""

import io
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "red": (196, 40, 28),
    "blue": (13, 105, 172),
    "yellow": (245, 205, 48),
    "green": (75, 151, 75),
    "orange": (218, 133, 65),
    "white": (242, 243, 243),
    "gray": (155, 161, 168),
    "purple": (123, 46, 146),
}
PLATE = 30  # baseplate studs per side
UX, UY, HZ = 16, 8, 15  # iso projection: px per stud (x-y), (x+y), per brick height
MIN_VIS = 0.55  # visible share of a brick's expected area
MIN_PIXELS = 80  # visible pixels per brick
THEME_ROLES = ("top", "side_x", "side_y", "outline", "stud")
PLATE_THEMES = {  # insertion order matters (Crop ID indexes list(PLATE_THEMES))
    name: dict(zip(THEME_ROLES, rgb))
    for name, rgb in {
        "sand": ((219, 205, 171), (187, 169, 132), (164, 146, 112), (142, 125, 94), (187, 171, 137)),
        "sky": ((181, 211, 226), (147, 180, 199), (124, 157, 177), (104, 137, 156), (151, 184, 202)),
        "lavender": ((209, 195, 224), (177, 158, 197), (153, 135, 174), (130, 111, 151), (181, 163, 201)),
        "mint": ((188, 219, 205), (153, 186, 171), (130, 162, 148), (108, 140, 126), (158, 191, 176)),
        "peach": ((231, 202, 187), (199, 169, 153), (174, 145, 131), (150, 122, 108), (203, 173, 157)),
        "slate": ((177, 190, 203), (143, 157, 171), (121, 135, 149), (98, 112, 126), (148, 162, 176)),
        "rose": ((226, 190, 201), (194, 157, 170), (169, 134, 147), (145, 112, 124), (198, 162, 175)),
        "lemon": ((229, 221, 171), (197, 188, 137), (172, 163, 116), (148, 139, 96), (201, 192, 142)),
    }.items()
}
THEME_ORDER = ("sand", "sky", "lavender", "mint", "rose", "lemon", "peach", "slate")
# canonical baseplate colors painted by render_scene -> theme role
BASE_PIXELS = dict(
    zip(((196, 206, 196), (168, 178, 168), (148, 158, 148), (140, 150, 140), (170, 180, 170)), THEME_ROLES)
)


def shade(c, f):
    return tuple(min(255, max(0, int(v * f))) for v in c)


def poly_area(poly):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def make_projector(maxz):
    """Scene height in bricks -> (P(x, y, z) -> pixel, canvas W, canvas H)."""
    W = (PLATE + 3) * 2 * UX
    Y0 = 24 + (maxz + 1) * HZ
    H = Y0 + 2 * PLATE * UY + 36
    X0 = W // 2

    def P(x, y, z):
        return (X0 + (x - y) * UX, Y0 + (x + y) * UY - z * HZ)

    return P, W, H


def draw_brick(d, b, P, id_fill=None):
    """Draw the 3 visible iso faces of brick b; return the face polygons."""
    x0, y0, z0 = b["x"], b["y"], b["z"]
    x1, y1, z1 = x0 + b["lx"], y0 + b["ly"], z0 + 1
    top = [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]
    fx = [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)]
    fy = [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)]
    if id_fill is not None:
        for poly in (top, fx, fy):
            d.polygon(poly, fill=id_fill)
    else:
        c = COLORS[b["color"]]
        d.polygon(top, fill=shade(c, 1.12), outline=(35, 35, 40))
        d.polygon(fx, fill=shade(c, 0.80), outline=(35, 35, 40))
        d.polygon(fy, fill=shade(c, 0.58), outline=(35, 35, 40))
    return top, fx, fy


def draw_studs(d, b, P, occ, id_fill=None):
    z1 = b["z"] + 1
    c = COLORS[b["color"]]
    for i in range(b["lx"]):
        for j in range(b["ly"]):
            cx, cy = b["x"] + i, b["y"] + j
            if (cx, cy, z1) in occ:
                continue
            X, Y = P(cx + 0.5, cy + 0.5, z1)
            rx, ry, sh = UX * 0.30, UY * 0.34, 3
            if id_fill is not None:
                d.ellipse([X - rx, Y - ry - sh, X + rx, Y + ry], fill=id_fill)
            else:
                d.ellipse([X - rx, Y - ry, X + rx, Y + ry], fill=shade(c, 0.85))
                d.ellipse([X - rx, Y - ry - sh, X + rx, Y + ry - sh], fill=shade(c, 1.22), outline=shade(c, 0.7))


def render_scene(scene):
    """Scene with `bricks` + `occ` -> (RGB image, id mask 'I', {id: visibility}, projector)."""
    bricks, occ = scene["bricks"], scene["occ"]
    P, W, H = make_projector(max(b["z"] + 1 for b in bricks))
    img = Image.new("RGB", (W, H), (247, 247, 250))
    idm = Image.new("I", (W, H), 0)
    d, di = ImageDraw.Draw(img), ImageDraw.Draw(idm)
    top = [P(0, 0, 0), P(PLATE, 0, 0), P(PLATE, PLATE, 0), P(0, PLATE, 0)]
    thick = 7
    d.polygon([top[1], top[2], (top[2][0], top[2][1] + thick), (top[1][0], top[1][1] + thick)], fill=(168, 178, 168))
    d.polygon([top[3], top[2], (top[2][0], top[2][1] + thick), (top[3][0], top[3][1] + thick)], fill=(148, 158, 148))
    d.polygon(top, fill=(196, 206, 196), outline=(140, 150, 140))
    for cx in range(PLATE):
        for cy in range(PLATE):
            if (cx, cy, 0) not in occ:
                X, Y = P(cx + 0.5, cy + 0.5, 0)
                d.ellipse([X - 3, Y - 1.6, X + 3, Y + 1.6], outline=(170, 180, 170))
    areas = {}
    for b in sorted(bricks, key=lambda b: (b["x"] + b["y"], b["z"])):
        top, fx, fy = draw_brick(d, b, P)
        draw_brick(di, b, P, id_fill=b["id"] + 1)
        draw_studs(d, b, P, occ)
        draw_studs(di, b, P, occ, id_fill=b["id"] + 1)
        # expected visible area: front faces + only the UNCOVERED share of the top face
        cells = [(b["x"] + i, b["y"] + j) for i in range(b["lx"]) for j in range(b["ly"])]
        exposed = sum(1 for (cx, cy) in cells if (cx, cy, b["z"] + 1) not in occ) / len(cells)
        areas[b["id"]] = poly_area(fx) + poly_area(fy) + poly_area(top) * exposed
    counts = Counter(idm.getdata())
    vis = {bid: counts.get(bid + 1, 0) / max(a, 1.0) for bid, a in areas.items()}
    return img, idm, vis, P


def recolor_plate(image, theme_name):
    """Copy of a canonical render with only the exact baseplate colors replaced by the theme's."""
    theme = PLATE_THEMES[theme_name]
    output = image.copy()
    pixels = output.load()
    for yy in range(output.height):
        for xx in range(output.width):
            role = BASE_PIXELS.get(pixels[xx, yy])
            if role is not None:
                pixels[xx, yy] = theme[role]
    return output


def add_brick(scene, x, y, z, lx, ly, color, role, pod_index, query_class):
    """Append one brick to scene (asserting no stud-volume overlap); return its id."""
    brick_id = len(scene["bricks"])
    brick = {"id": brick_id, "x": x, "y": y, "z": z, "lx": lx, "ly": ly, "color": color}
    brick.update(construction_role=role, pod_index=pod_index, query_class=bool(query_class))
    for dx in range(lx):
        for dy in range(ly):
            cell = (x + dx, y + dy, z)
            assert cell not in scene["occ"], (cell, brick)
            scene["occ"][cell] = brick_id
    scene["bricks"].append(brick)
    return brick_id


def blank_scene(seed, scene_index, family, theme, query_color):
    scene = {"bricks": [], "occ": {}, "plate": PLATE, "seed": seed, "scene_index": scene_index, "family": family}
    return {**scene, "plate_theme": theme, "plate_theme_rgb": PLATE_THEMES[theme], "query_color": query_color}


def origins(rng, count, width, height):
    """`count` non-overlapping width x height footprint origins (one-stud gap), sorted by screen x."""
    for _ in range(1000):
        out = []
        for _slot in range(count):
            for _try in range(2000):
                x, y = rng.randint(0, PLATE - width), rng.randint(0, PLATE - height)
                if all(
                    x + width + 1 <= ox or ox + width + 1 <= x or y + height + 1 <= oy or oy + height + 1 <= y
                    for ox, oy in out
                ):
                    out.append((x, y))
                    break
            else:
                break
        if len(out) == count:
            return sorted(out, key=lambda xy: ((xy[0] + width / 2.0) - (xy[1] + height / 2.0)) * UX)
    raise RuntimeError("could not place footprints")


def visibility_gate(scene, id_mask, visibility):
    """Assert every brick is readable; return (min visibility, min visible pixels)."""
    counts = Counter(id_mask.getdata())
    min_vis = min(visibility[b["id"]] for b in scene["bricks"])
    min_pixels = min(counts.get(b["id"] + 1, 0) for b in scene["bricks"])
    assert min_vis >= MIN_VIS, ("visibility", min_vis)
    assert min_pixels >= MIN_PIXELS, ("pixels", min_pixels)
    return round(min_vis, 4), min_pixels


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def dilate(mask, iterations):
    """4-neighborhood dilation of a bool (H, W) mask."""
    out = mask.copy()
    for _ in range(iterations):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def interior_anchor(mask):
    """Interior pixel (x, y) of a bool (H, W) mask via iterative erosion."""
    current = mask
    while True:
        eroded = current.copy()
        eroded[1:, :] &= current[:-1, :]
        eroded[:-1, :] &= current[1:, :]
        eroded[:, 1:] &= current[:, :-1]
        eroded[:, :-1] &= current[:, 1:]
        eroded[0, :] = eroded[-1, :] = eroded[:, 0] = eroded[:, -1] = False
        if not eroded.any():
            ys, xs = np.nonzero(current)
            return int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
        current = eroded


def rerender(scene):
    """Re-render a stored scene -> (RGB image before plate recolor, id mask 'I')."""
    occ = {
        (b["x"] + dx, b["y"] + dy, b["z"]): b["id"]
        for b in scene["bricks"]
        for dx in range(b["lx"])
        for dy in range(b["ly"])
    }
    image, id_mask, _, _ = render_scene(dict(scene, occ=occ))
    return image, id_mask


def render_id_mask(scene):
    """Stored scene -> (H, W) int32 brick-id mask (id + 1, 0 = background)."""
    return np.asarray(rerender(scene)[1])
