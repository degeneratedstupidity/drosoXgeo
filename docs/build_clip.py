"""Blender scene for the post: a cartoon fly at a gaming rig, playing the real recorded run (illustration).

The monitor shows a real recorded attempt of the connectome-constrained fly model in the real game (brain.live
--record; frames in results/clip/screen_crop/, made from the npz), and the fly's front leg presses the space bar on
exactly the frames the model pressed (results/clip/timeline.json). Everything else (the room, the rig, the fly) is
an illustration.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P docs/build_clip.py -- --still 250      # one preview frame
    /Applications/Blender.app/Contents/MacOS/Blender -b -P docs/build_clip.py -- --render          # all frames
    ... -- --wide --render                                                                       # 16:9 version
    /Applications/Blender.app/Contents/MacOS/Blender -P docs/build_clip.py                          # open it in Blender
Output in results/clip/ (gitignored): scene .blend, preview PNGs, render/####.png.
"""

import json
import math
import os
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(ROOT, "results", "clip")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
STAGES = argv[argv.index("--stages") + 1].split(",") if "--stages" in argv else ["room", "rig", "deco", "fly"]
WIDE = "--wide" in argv                          # 16:9 (1920x1080) instead of the square 1080x1080
TAG = "_wide" if WIDE else ""

timeline = json.load(open(os.path.join(CLIP, "timeline.json")))
PRESSED = timeline["pressed"]
N = len(PRESSED)
FPS = timeline["fps"]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 1, N
scene.render.resolution_x, scene.render.resolution_y = (1920, 1080) if WIDE else (1080, 1080)
for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        pass
scene.eevee.taa_render_samples = 32
for attr, val in (("use_raytracing", True), ("use_shadows", True)):
    if hasattr(scene.eevee, attr):
        setattr(scene.eevee, attr, val)
scene.render.use_motion_blur = True
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast" if "AgX - Medium High Contrast" in [
    i.identifier for i in scene.view_settings.bl_rna.properties["look"].enum_items] else "None"

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0008, 0.0008, 0.0025, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0


# ---------------------------------------------------------------- helpers
def material(name, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, emit=None, strength=0.0, transmission=0.0,
             alpha=1.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    sets = {"Base Color": (*color, 1), "Roughness": rough, "Metallic": metal, "Alpha": alpha,
            "Transmission Weight": transmission, "Coat Weight": coat}
    if emit is not None:
        sets.update({"Emission Color": (*emit, 1), "Emission Strength": strength})
    for k, v in sets.items():
        if k in b.inputs:
            b.inputs[k].default_value = v
    if alpha < 1 or transmission > 0:
        for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, val)
                except TypeError:
                    pass
    return m


def emission(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], out.inputs[0])
    return m


def rgb_cycle(m, phase=0.0, speed=0.04):
    """Animate an emission material's colour between purple, cyan and magenta (drivers on the frame)."""
    e = [n for n in m.node_tree.nodes if n.type == "EMISSION"][0]
    exprs = (f"0.45 + 0.45 * sin(frame * {speed} + {phase:.3f})",            # red: purple/magenta <-> cyan
             f"0.15 + 0.45 * max(0, -sin(frame * {speed} + {phase:.3f}))",   # green only on the cyan side
             "1.0")
    for i, ex in enumerate(exprs):
        e.inputs["Color"].driver_add("default_value", i).driver.expression = ex


def box(name, size, loc, mat, bevel=0.0, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments = bevel, 3
    o.data.materials.append(mat)
    return o


def cylinder(name, r, depth, loc, mat, rot=(0, 0, 0), verts=32):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot, vertices=verts)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    return o


def sphere(name, r, loc, mat, scale=(1, 1, 1), seg=32):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=seg, ring_count=seg // 2)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    return o


def segment(name, a, b, r, mat):
    """A cylinder from point a to point b."""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    o = cylinder(name, r, length, ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2), mat, verts=12)
    o.rotation_mode = "QUATERNION"
    from mathutils import Vector
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector((dx, dy, dz)))
    bpy.ops.object.shade_smooth()
    return o


def area_light(name, loc, rot, size, power, color):
    bpy.ops.object.light_add(type="AREA", location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.data.size = size
    o.data.energy = power
    o.data.color = color
    return o


def point_light(name, loc, power, color, radius=0.05):
    bpy.ops.object.light_add(type="POINT", location=loc)
    o = bpy.context.object
    o.name = name
    o.data.energy = power
    o.data.color = color
    o.data.shadow_soft_size = radius
    return o


DESK = 0.75                                  # desk top height (m)

# ---------------------------------------------------------------- room, desk, monitor, keyboard
if "room" in STAGES:
    floor = material("floor", (0.012, 0.011, 0.013), rough=0.35)
    wall = material("wall", (0.007, 0.008, 0.018), rough=0.85)
    box("floor", (8, 8, 0.02), (0, 0, -0.01), floor)
    box("wall_back", (8, 0.05, 3.2), (0, 0.75, 1.6), wall)
    box("wall_left", (0.05, 8, 3.2), (-1.35, 0, 1.6), wall)
    strip_c = emission("led_cyan", (0.1, 0.8, 1.0), 6)
    strip_p = emission("led_purple", (0.65, 0.15, 1.0), 6)
    box("led_back_low", (3.0, 0.01, 0.012), (0, 0.72, 1.02), strip_c)
    box("led_back_high", (8, 0.01, 0.015), (0, 0.72, 2.45), strip_p)
    box("led_left", (0.01, 3.0, 0.015), (-1.32, 0, 2.45), strip_p)
    # an abstract poster: glowing "neurons" on a dark board
    poster = bpy.data.materials.new("poster")
    poster.use_nodes = True
    nt = poster.node_tree
    b = nt.nodes["Principled BSDF"]
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 7
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.9, 0.3, 1.0, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.04, (0.01, 0.01, 0.03, 1)
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 1.2
    b.inputs["Base Color"].default_value = (0.01, 0.01, 0.02, 1)
    frame_m = material("frame", (0.01, 0.01, 0.01), rough=0.3, metal=0.5)


    # desk
    desk = material("desk", (0.006, 0.006, 0.007), rough=0.25, coat=0.4)
    leg = material("desk_leg", (0.03, 0.03, 0.03), rough=0.3, metal=0.9)
    box("desk_top", (1.7, 0.8, 0.04), (0, 0.15, DESK - 0.02), desk, bevel=0.005)
    for x in (-0.8, 0.8):
        for y in (-0.2, 0.5):
            box(f"desk_leg_{x}_{y}", (0.05, 0.05, DESK - 0.04), (x, y, (DESK - 0.04) / 2), leg)
    under = emission("desk_under", (0.6, 0.1, 1.0), 4)
    rgb_cycle(under, phase=1.0, speed=0.02)
    box("desk_led", (1.6, 0.01, 0.008), (0, -0.245, DESK - 0.045), under)
    mat_pad = material("mousepad", (0.01, 0.01, 0.012), rough=0.9)
    box("deskmat", (1.0, 0.36, 0.004), (0.05, -0.08, DESK + 0.002), mat_pad, bevel=0.002)

    # monitor with the real recorded run
    plastic = material("monitor_plastic", (0.012, 0.012, 0.014), rough=0.35)
    W, H = 0.64, 0.64 * 247 / 440
    sx, sy, sz = 0.0, 0.33, DESK + 0.17 + H / 2
    box("monitor_bezel", (W + 0.02, 0.025, H + 0.02), (sx, sy + 0.013, sz), plastic, bevel=0.004)
    box("monitor_neck", (0.05, 0.03, 0.2), (sx, sy + 0.06, DESK + 0.1), plastic)
    box("monitor_foot", (0.26, 0.18, 0.012), (sx, sy + 0.04, DESK + 0.006), plastic, bevel=0.004)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(sx, sy - 0.0005, sz), rotation=(math.pi / 2, 0, 0))
    screen = bpy.context.object
    screen.name = "screen"
    screen.scale = (W, H, 1)
    scr = bpy.data.materials.new("screen")
    scr.use_nodes = True
    nt = scr.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(os.path.join(CLIP, "screen_crop", "0001.png"))
    img.source = "SEQUENCE"
    tex.image = img
    tex.image_user.frame_duration = N
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = 0
    tex.image_user.use_auto_refresh = True
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 2.2
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs[0])
    screen.data.materials.append(scr)
    # the screen's light on the desk and the fly
    area_light("screen_glow", (sx, sy - 0.03, sz), (math.radians(90), 0, 0), W, 11, (0.55, 0.35, 1.0))

    # keyboard, keys, space bar (pressed by the fly, see below)
    kb_body = material("kb_body", (0.02, 0.02, 0.022), rough=0.3, metal=0.6)
    keycap = material("keycap", (0.03, 0.03, 0.035), rough=0.45)
    kx, ky = -0.06, 0.1
    box("kb", (0.44, 0.15, 0.018), (kx, ky, DESK + 0.013), kb_body, bevel=0.004)
    glow = emission("kb_glow", (0.3, 0.6, 1.0), 8)
    rgb_cycle(glow, speed=0.05)
    box("kb_glow", (0.45, 0.16, 0.003), (kx, ky, DESK + 0.005), glow)
    for row in range(5):
        for col in range(15):
            x = kx - 0.2 + col * 0.0285
            y = ky + 0.055 - row * 0.0255
            if row == 4 and 4 <= col <= 9:
                continue                                  # the space bar's place
            box(f"key_{row}_{col}", (0.024, 0.021, 0.01), (x, y, DESK + 0.027), keycap, bevel=0.002)
    space = box("space", (0.15, 0.021, 0.01), (kx - 0.2 + 6.5 * 0.0285, ky + 0.055 - 4 * 0.0255, DESK + 0.027),
                keycap, bevel=0.002)
    mouse_m = material("mouse", (0.015, 0.015, 0.017), rough=0.3)
    sphere("mouse", 0.035, (0.34, 0.06, DESK + 0.012), mouse_m, scale=(0.9, 1.4, 0.5))

    # room lights
    area_light("key", (-0.9, -1.2, 2.3), (math.radians(55), 0, math.radians(-35)), 1.5, 10, (0.6, 0.65, 1.0))
    area_light("rim", (0.6, 1.0, 1.6), (math.radians(-60), 0, math.radians(160)), 0.8, 8, (0.8, 0.3, 1.0))

    # camera: over the fly's shoulder, a slow push-in
    bpy.ops.object.camera_add(location=(0.5, -0.8, 1.06))
    cam = bpy.context.object
    scene.camera = cam
    cam.data.lens = 32
    if WIDE:                                     # same height of view as the square, more room left and right
        cam.data.sensor_fit, cam.data.sensor_height = "VERTICAL", cam.data.sensor_width
    bpy.ops.object.empty_add(location=(-0.1, 0.17, 0.9))
    target = bpy.context.object
    target.name = "cam_target"
    c = cam.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    cam.keyframe_insert("location", frame=1)
    cam.location = (0.42, -0.68, 1.02)
    cam.keyframe_insert("location", frame=N)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.75            # about the fly's head; f/8 keeps the screen readable
    cam.data.dof.aperture_fstop = 8.0

# ---------------------------------------------------------------- the rig: glass tower, RGB fans, GPU, AIO, RAM
def fan(name, loc, axis, size=0.12, phase=0.0, speed=1.1, ring_phase=0.0):
    """A case fan facing along `axis` ('x' or 'y'): frame, RGB ring, 7 blades spinning (driver on the frame)."""
    rot = (math.pi / 2, 0, 0) if axis == "y" else (0, math.pi / 2, 0)
    dark = bpy.data.materials.get("fan_frame") or material("fan_frame", (0.01, 0.01, 0.012), rough=0.4)
    blade_m = bpy.data.materials.get("fan_blade") or material("fan_blade", (0.6, 0.6, 0.7), rough=0.3,
                                                                 transmission=0.6, alpha=0.6)
    ring = emission(f"{name}_ring", (0.6, 0.2, 1.0), 12)
    rgb_cycle(ring, phase=ring_phase, speed=0.06)
    bpy.ops.object.empty_add(location=loc, rotation=rot)
    hub_empty = bpy.context.object
    hub_empty.name = f"{name}_spin"
    # frame: a square with a round hole, approximated by four bars
    t = 0.008
    for i, (dx, dz, sx, sz) in enumerate(((0, size / 2 - t / 2, size, t), (0, -size / 2 + t / 2, size, t),
                                         (size / 2 - t / 2, 0, t, size), (-size / 2 + t / 2, 0, t, size))):
        if axis == "y":
            box(f"{name}_frame{i}", (sx, 0.025, sz), (loc[0] + dx, loc[1], loc[2] + dz), dark)
        else:
            box(f"{name}_frame{i}", (0.025, sx, sz), (loc[0], loc[1] + dx, loc[2] + dz), dark)
    bpy.ops.mesh.primitive_torus_add(major_radius=size * 0.46, minor_radius=0.0035, location=loc, rotation=rot)
    r = bpy.context.object
    r.name = f"{name}_ringobj"
    r.data.materials.append(ring)
    hub = cylinder(f"{name}_hub", size * 0.16, 0.022, (0, 0, 0), dark)
    hub.parent = hub_empty
    hub.location = (0, 0, 0)
    for k in range(7):
        a = 2 * math.pi * k / 7
        bl = box(f"{name}_blade{k}", (size * 0.3, size * 0.11, 0.003), (0, 0, 0), blade_m)
        bl.parent = hub_empty
        bl.location = (math.cos(a) * size * 0.25, math.sin(a) * size * 0.25, 0)
        bl.rotation_euler = (0.45, 0, a)
    d = hub_empty.driver_add("rotation_euler", 2).driver
    d.expression = f"frame * {speed} + {phase}"
    return hub_empty


if "rig" in STAGES:
    case_m = material("case", (0.012, 0.012, 0.014), rough=0.3, metal=0.7)
    glass = material("glass", (0.05, 0.05, 0.07), rough=0.02, transmission=1.0, alpha=0.18)
    x0, x1, y0, y1, z0, z1 = -0.82, -0.56, -0.08, 0.44, DESK, DESK + 0.52
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    box("case_left", (0.006, y1 - y0, z1 - z0), (x0, cy, cz), case_m)
    box("case_back", (x1 - x0, 0.006, z1 - z0), (cx, y1, cz), case_m)
    box("case_top", (x1 - x0, y1 - y0, 0.008), (cx, cy, z1), case_m)
    box("case_bottom", (x1 - x0, y1 - y0, 0.02), (cx, cy, z0 + 0.01), case_m)
    box("glass_side", (0.004, y1 - y0, z1 - z0), (x1, cy, cz), glass)
    box("glass_front", (x1 - x0, 0.004, z1 - z0), (cx, y0, cz), glass)
    for i in range(3):                                         # three front intake fans, just behind the glass
        fan(f"front_fan{i}", (cx, y0 + 0.03, z0 + 0.1 + i * 0.145), "y", phase=i * 0.7, ring_phase=i * 0.8)
    fan("rear_fan", (cx, y1 - 0.03, z1 - 0.1), "y", phase=0.3, ring_phase=2.0)
    # motherboard, RAM, AIO pump, GPU
    board = material("board", (0.02, 0.025, 0.03), rough=0.5)
    box("motherboard", (0.006, 0.32, 0.34), (x0 + 0.01, y1 - 0.19, z1 - 0.2), board)
    for i in range(4):
        ram_led = emission(f"ram{i}_led", (0.6, 0.2, 1.0), 10)
        rgb_cycle(ram_led, phase=i * 0.5, speed=0.08)
        box(f"ram{i}", (0.035, 0.008, 0.11), (x0 + 0.03, y1 - 0.12 + i * 0.013, z1 - 0.15), case_m)
        box(f"ram{i}_top", (0.035, 0.008, 0.012), (x0 + 0.03, y1 - 0.12 + i * 0.013, z1 - 0.088), ram_led)
    pump_ring = emission("pump_ring", (0.2, 0.8, 1.0), 14)
    rgb_cycle(pump_ring, phase=3.0, speed=0.07)
    cylinder("pump", 0.03, 0.03, (x0 + 0.03, y1 - 0.2, z1 - 0.17), case_m, rot=(0, math.pi / 2, 0))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.026, minor_radius=0.003, location=(x0 + 0.046, y1 - 0.2,
                                                                                       z1 - 0.17),
                                     rotation=(0, math.pi / 2, 0))
    bpy.context.object.data.materials.append(pump_ring)
    gpu_m = material("gpu", (0.03, 0.03, 0.035), rough=0.3, metal=0.8)
    box("gpu", (0.11, 0.3, 0.05), (x0 + 0.065, y1 - 0.2, z0 + 0.19), gpu_m, bevel=0.004)
    gpu_led = emission("gpu_led", (0.6, 0.2, 1.0), 10)
    rgb_cycle(gpu_led, phase=1.5, speed=0.05)
    box("gpu_strip", (0.004, 0.28, 0.006), (x0 + 0.121, y1 - 0.2, z0 + 0.205), gpu_led)
    # GPU fans face down: spin around z
    for i in range(2):
        bpy.ops.object.empty_add(location=(x0 + 0.065, y1 - 0.27 + i * 0.13, z0 + 0.163))
        e = bpy.context.object
        e.name = f"gpu_fan{i}_spin"
        for k in range(9):
            a = 2 * math.pi * k / 9
            bl = box(f"gpu_fan{i}_blade{k}", (0.03, 0.012, 0.002), (0, 0, 0),
                     bpy.data.materials.get("fan_blade"))
            bl.parent = e
            bl.location = (math.cos(a) * 0.022, math.sin(a) * 0.022, 0)
            bl.rotation_euler = (0.5, 0, a)
        e.driver_add("rotation_euler", 2).driver.expression = f"frame * 1.4 + {i}"
    # the RGB lights inside the case, and the glow they throw on the desk
    point_light("case_light_a", (cx, y0 + 0.1, z0 + 0.3), 10, (0.6, 0.2, 1.0))
    point_light("case_light_b", (cx, y1 - 0.1, z1 - 0.12), 8, (0.2, 0.8, 1.0))
    for i, (yy, zz, sy_, sz_) in enumerate(((y0 + 0.01, z1 - 0.012, y1 - y0 - 0.02, 0.006),
                                             (y0 + 0.01, z0 + 0.025, y1 - y0 - 0.02, 0.006))):
        strip = emission(f"case_strip{i}", (0.6, 0.2, 1.0), 14)
        rgb_cycle(strip, phase=i * 1.7, speed=0.05)
        box(f"case_strip{i}", (0.006, sy_, sz_), (x1 - 0.008, cy, zz), strip)
    for i, zz in enumerate((z0 + 0.03, z1 - 0.03)):
        strip = emission(f"case_front_strip{i}", (0.2, 0.8, 1.0), 14)
        rgb_cycle(strip, phase=2.5 + i, speed=0.05)
        box(f"case_front_strip{i}", (x1 - x0 - 0.02, 0.006, 0.006), (cx, y0 + 0.008, zz), strip)

# ---------------------------------------------------------------- RGB deco and figurines
def figurine_robot(loc, s=1.0):
    x, y, z = loc
    metal = material("robot_metal", (0.5, 0.52, 0.6), rough=0.25, metal=0.9)
    eye = emission("robot_eye", (0.1, 0.9, 1.0), 20)
    box("robot_body", (0.04 * s, 0.03 * s, 0.045 * s), (x, y, z + 0.035 * s), metal, bevel=0.004 * s)
    box("robot_head", (0.035 * s, 0.028 * s, 0.03 * s), (x, y, z + 0.075 * s), metal, bevel=0.004 * s)
    for dx in (-0.008, 0.008):
        sphere("robot_eye", 0.0045 * s, (x + dx * s, y - 0.0145 * s, z + 0.078 * s), eye, seg=12)
    segment("robot_antenna", (x, y, z + 0.09 * s), (x, y, z + 0.11 * s), 0.0012 * s, metal)
    tip = emission("robot_tip", (1.0, 0.1, 0.5), 25)
    sphere("robot_tip", 0.004 * s, (x, y, z + 0.112 * s), tip, seg=12)
    for dx in (-0.026, 0.026):
        segment("robot_arm", (x + dx * s, y, z + 0.05 * s), (x + dx * 1.2 * s, y - 0.01 * s, z + 0.02 * s),
                0.004 * s, metal)
    for dx in (-0.01, 0.01):
        segment("robot_leg", (x + dx * s, y, z + 0.013 * s), (x + dx * s, y, z), 0.005 * s, metal)


def figurine_astronaut(loc, s=1.0):
    x, y, z = loc
    suit = material("suit", (0.85, 0.85, 0.9), rough=0.6)
    visor = material("visor", (0.02, 0.02, 0.04), rough=0.05, metal=1.0, emit=(0.2, 0.6, 1.0), strength=0.6)
    sphere("astro_body", 0.022 * s, (x, y, z + 0.03 * s), suit, scale=(1, 0.85, 1.25))
    sphere("astro_helmet", 0.02 * s, (x, y, z + 0.07 * s), suit)
    sphere("astro_visor", 0.014 * s, (x, y - 0.01 * s, z + 0.071 * s), visor, scale=(1.1, 0.8, 0.8))
    box("astro_pack", (0.03 * s, 0.012 * s, 0.035 * s), (x, y + 0.02 * s, z + 0.035 * s), suit, bevel=0.003 * s)


def figurine_duck(loc, s=1.0):
    x, y, z = loc
    yellow = material("duck", (1.0, 0.75, 0.05), rough=0.3)
    orange = material("beak", (1.0, 0.35, 0.02), rough=0.3)
    sphere("duck_body", 0.025 * s, (x, y, z + 0.022 * s), yellow, scale=(1.0, 1.3, 0.85))
    sphere("duck_head", 0.016 * s, (x, y - 0.018 * s, z + 0.05 * s), yellow)
    bpy.ops.mesh.primitive_cone_add(radius1=0.006 * s, depth=0.014 * s, location=(x, y - 0.036 * s, z + 0.048 * s),
                                    rotation=(math.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(orange)


def figurine_brain(loc, s=1.0):
    x, y, z = loc
    m = bpy.data.materials.new("brain")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.95, 0.45, 0.6, 1)
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.6
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 6
    wave.inputs["Distortion"].default_value = 12
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.8
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    for dx in (-0.012, 0.012):
        sphere("brain_half", 0.02 * s, (x + dx * s, y, z + 0.03 * s), m, scale=(0.75, 1.25, 0.85))
    stand = material("stand", (0.02, 0.02, 0.02), rough=0.3, metal=0.7)
    cylinder("brain_stand", 0.022 * s, 0.01 * s, (x, y, z + 0.005 * s), stand)


def figurine_mushroom(loc, s=1.0):
    x, y, z = loc
    stem = material("stem", (0.9, 0.88, 0.8), rough=0.6)
    cap = material("cap", (0.85, 0.05, 0.08), rough=0.35)
    cylinder("mush_stem", 0.008 * s, 0.03 * s, (x, y, z + 0.015 * s), stem, verts=16)
    sphere("mush_cap", 0.022 * s, (x, y, z + 0.032 * s), cap, scale=(1, 1, 0.6))
    dot = material("dots", (0.95, 0.95, 0.95), rough=0.6)
    for a in range(5):
        aa = a * 2 * math.pi / 5
        sphere("mush_dot", 0.004 * s, (x + math.cos(aa) * 0.013 * s, y + math.sin(aa) * 0.013 * s,
                                        z + 0.04 * s), dot, seg=10)


if "deco" in STAGES:
    # hexagon light panels on the back wall, right of the monitor, each cycling its own colour
    hexes = [(0, 0), (1, 0), (0.5, 0.866), (1.5, 0.866), (1, 1.732), (2, 0), (1.5, -0.866)]
    for i, (hx, hy) in enumerate(hexes):
        m = emission(f"hex{i}", (0.6, 0.2, 1.0), 5)
        rgb_cycle(m, phase=i * 0.9, speed=0.035)
        cylinder(f"hex{i}", 0.052, 0.01, (0.3 + hx * 0.095, 0.715, 1.5 + hy * 0.095), m,
                 rot=(math.pi / 2, 0, math.pi / 6), verts=6)
    # a neon sign above the monitor
    bpy.ops.object.text_add(location=(-0.1, 0.71, 1.42), rotation=(math.pi / 2, 0, 0))
    t = bpy.context.object
    t.name = "neon"
    t.data.body = "JUMP!"
    t.data.size = 0.13
    t.data.extrude = 0.004
    t.data.align_x = "CENTER"
    t.location.x = -0.08
    t.data.materials.append(emission("neon", (1.0, 0.15, 0.6), 18))
    point_light("neon_glow", (0.0, 0.6, 1.48), 3, (1.0, 0.2, 0.6), radius=0.2)
    # monitor bias light: a glow on the wall behind the screen
    bias = emission("bias", (0.4, 0.2, 1.0), 10)
    rgb_cycle(bias, phase=0.5, speed=0.03)
    box("bias_strip", (0.6, 0.006, 0.006), (0.0, 0.37, DESK + 0.17 + 0.18), bias)
    area_light("bias_light", (0.0, 0.4, DESK + 0.36), (math.radians(-90), 0, 0), 0.6, 5, (0.5, 0.25, 1.0))
    # speakers with glowing rings, either side of the monitor
    spk = material("speaker", (0.015, 0.015, 0.017), rough=0.4)
    for sx_ in (-0.43, 0.43):
        box(f"speaker_{sx_}", (0.1, 0.11, 0.2), (sx_, 0.3, DESK + 0.1), spk, bevel=0.006)
        ring = emission(f"speaker_ring_{sx_}", (0.2, 0.8, 1.0), 12)
        rgb_cycle(ring, phase=1.0 if sx_ < 0 else 2.5, speed=0.06)
        for zz, rr in ((DESK + 0.06, 0.035), (DESK + 0.15, 0.02)):
            bpy.ops.mesh.primitive_torus_add(major_radius=rr, minor_radius=0.003, location=(sx_, 0.244, zz),
                                             rotation=(math.pi / 2, 0, 0))
            bpy.context.object.data.materials.append(ring)
    # a floating shelf above the PC with figurines, and an RGB underglow
    shelf = material("shelf", (0.02, 0.02, 0.022), rough=0.3)
    box("shelf", (0.56, 0.16, 0.02), (-0.62, 0.64, 1.45), shelf, bevel=0.003)
    under = emission("shelf_under", (0.6, 0.2, 1.0), 10)
    rgb_cycle(under, phase=3.5, speed=0.04)
    box("shelf_led", (0.54, 0.006, 0.006), (-0.62, 0.565, 1.438), under)
    area_light("shelf_light", (-0.62, 0.62, 1.43), (math.pi, 0, 0), 0.5, 6, (0.6, 0.3, 1.0))
    top = 1.46
    figurine_robot((-0.84, 0.63, top), 1.3)
    figurine_astronaut((-0.72, 0.63, top), 1.3)
    figurine_brain((-0.6, 0.63, top), 1.3)
    figurine_duck((-0.48, 0.63, top), 1.2)
    figurine_mushroom((-0.38, 0.63, top), 1.2)
    # two more on the desk
    figurine_robot((0.3, 0.13, DESK), 1.1)
    figurine_duck((0.22, 0.2, DESK), 1.0)
    # floor glow behind the desk
    floor_led = emission("floor_led", (0.2, 0.8, 1.0), 6)
    rgb_cycle(floor_led, phase=4.0, speed=0.025)
    box("floor_led", (3.0, 0.01, 0.01), (0, 0.7, 0.02), floor_led)

# ---------------------------------------------------------------- the fly
if "fly" in STAGES:
    S = 1.6                                         # cartoon scale: about 15 cm nose to tail
    space = bpy.data.objects["space"]
    sx_, sy_, sz_ = space.location
    key_top = sz_ + 0.005
    fx, fy, fz = sx_ - 0.1, sy_ - 0.15, DESK + 0.036 * S     # thorax centre: a low crouch, left of the space bar

    def fuzzy(name, color, scale=90, strength=0.3):
        m = material(name, color, rough=0.6)
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = strength
        nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        return m

    def banded(name, light, dark, scale, direction="Y", lo=0.42, hi=0.58):
        m = fuzzy(name, light)
        nt = m.node_tree
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = direction
        wave.inputs["Scale"].default_value = scale
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*dark, 1)
        ramp.color_ramp.elements[1].color = (*light, 1)
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = lo, hi
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
        nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        return m

    # Drosophila colours: tan-brown body, black bands on the abdomen, bright red eyes
    thorax_m = fuzzy("fly_thorax", (0.36, 0.22, 0.09))
    head_m = fuzzy("fly_head", (0.42, 0.26, 0.1))
    abd_m = banded("fly_abdomen", (0.42, 0.28, 0.11), (0.02, 0.015, 0.01), 5.5, direction="Y", lo=0.5, hi=0.6)
    face_m = material("fly_face", (0.75, 0.65, 0.45), rough=0.4, metal=0.3)         # the silvery-gold face
    eye_m = material("fly_eye", (0.75, 0.02, 0.01), rough=0.2, coat=1.0)
    nt = eye_m.node_tree
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 220
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    # wings: smoky membrane with darker veins (cells from a Voronoi pattern) and a thin-film sheen
    wing_m = material("fly_wing", (0.55, 0.6, 0.68), rough=0.1, alpha=0.5)
    nt = wing_m.node_tree
    wb = nt.nodes["Principled BSDF"]
    if "Thin Film Thickness" in wb.inputs:
        wb.inputs["Thin Film Thickness"].default_value = 600
    veins = nt.nodes.new("ShaderNodeTexWave")
    veins.wave_type = "BANDS"
    veins.bands_direction = "X"
    veins.wave_profile = "SAW"
    veins.inputs["Scale"].default_value = 2.2
    veins.inputs["Distortion"].default_value = 1.5
    tcw = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tcw.outputs["Object"], veins.inputs["Vector"])
    vein = nt.nodes.new("ShaderNodeMapRange")
    vein.inputs["From Min"].default_value, vein.inputs["From Max"].default_value = 0.0, 0.06
    vein.inputs["To Min"].default_value, vein.inputs["To Max"].default_value = 0.9, 0.32
    nt.links.new(veins.outputs["Fac"], vein.inputs["Value"])
    nt.links.new(vein.outputs["Result"], wb.inputs["Alpha"])
    leg_m = fuzzy("fly_leg", (0.3, 0.19, 0.08), scale=150, strength=0.2)
    hair_m = material("fly_hair", (0.01, 0.01, 0.01), rough=0.5)

    bpy.ops.object.empty_add(location=(fx, fy, fz))
    body = bpy.context.object
    body.name = "fly"

    def part(o, parent=None):
        parent = parent or body
        p = parent.matrix_world.translation if parent is not body else (fx, fy, fz)
        o.parent = parent
        o.location = (o.location[0] - p[0], o.location[1] - p[1], o.location[2] - p[2])
        return o

    def at(dx, dy, dz):
        return (fx + dx * S, fy + dy * S, fz + dz * S)

    part(sphere("fly_thorax", 0.03 * S, at(0, 0, 0), thorax_m, scale=(0.95, 1.12, 0.92)))
    part(sphere("fly_abdomen", 0.028 * S, at(0, -0.056, -0.004), abd_m, scale=(1.0, 1.4, 0.8)))
    part(sphere("fly_head", 0.02 * S, at(0, 0.04, 0.003), head_m, scale=(1.3, 0.75, 1.05)))
    part(sphere("fly_face", 0.012 * S, at(0, 0.052, -0.002), face_m, scale=(0.9, 0.5, 1.1)))
    for dx in (-1, 1):
        part(sphere("fly_eye", 0.021 * S, at(dx * 0.02, 0.041, 0.006), eye_m, scale=(0.75, 0.95, 1.12)))
        part(segment("fly_antenna", at(dx * 0.004, 0.056, 0.008), at(dx * 0.007, 0.064, 0.016), 0.0016 * S, leg_m))
        part(sphere("fly_antenna_tip", 0.0025 * S, at(dx * 0.007, 0.064, 0.017), leg_m, seg=10))
    part(segment("fly_proboscis", at(0, 0.054, -0.012), at(0, 0.058, -0.026), 0.003 * S, leg_m))
    part(sphere("fly_proboscis_pad", 0.0045 * S, at(0, 0.058, -0.027), leg_m, scale=(1.3, 1, 0.6), seg=12))
    # bristles on the thorax and the abdomen's end
    import random
    rnd = random.Random(3)
    for i in range(46):
        on_thorax = i < 30
        cx_, cy_, cz_, rx, ry, rz = ((0, 0, 0, 0.0285, 0.0336, 0.0276) if on_thorax else
                                     (0, -0.056, -0.004, 0.028, 0.039, 0.0224))
        a = rnd.uniform(-1.2, 1.2)
        b = rnd.uniform(-0.9, 0.9) if on_thorax else rnd.uniform(-1.4, -0.5)
        nx, ny, nz = math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)
        base = at(cx_ + nx * rx, cy_ + ny * ry, cz_ + nz * rz)
        tip = (base[0] + nx * 0.007 * S, base[1] + ny * 0.007 * S - 0.004 * S, base[2] + nz * 0.007 * S)
        part(segment("fly_bristle", base, tip, 0.0006 * S, hair_m))

    # wings: resting in a V over the abdomen; a gentle idle twitch, a buzz when the cube dies
    wings = []
    for dx in (-1, 1):
        bpy.ops.object.empty_add(location=at(dx * 0.012, -0.006, 0.024))
        hinge = part(bpy.context.object)
        hinge.name = f"wing_hinge_{dx}"
        bpy.ops.mesh.primitive_circle_add(vertices=48, radius=1, fill_type="NGON", location=(0, 0, 0))
        w = bpy.context.object
        w.name = f"fly_wing_{dx}"
        w.scale = (0.024 * S, 0.062 * S, 1)
        w.data.materials.append(wing_m)
        w.parent = hinge
        w.location = (dx * 0.018 * S, -0.052 * S, 0.006 * S)
        w.rotation_euler = (0.12, 0, dx * -0.42)
        wings.append((hinge, dx))

    # legs: three pairs, thick and slightly hairy; the front right one rests on the space bar
    def leg(name, hip, knee, foot, parent=None):
        out = [segment(name + "_femur", hip, knee, 0.0046 * S, leg_m),
               sphere(name + "_knee", 0.0046 * S, knee, leg_m, seg=12),
               segment(name + "_tibia", knee, foot, 0.0034 * S, leg_m),
               sphere(name + "_foot", 0.004 * S, foot, leg_m, scale=(1.3, 1.6, 0.6), seg=12)]
        for o in out:
            part(o, parent)
        return out

    for dx in (-1, 1):                               # knees rise above the body, feet spread wide
        leg(f"leg_mid_{dx}", at(dx * 0.022, -0.004, -0.012), at(dx * 0.048, 0.002, 0.02),
            (fx + dx * 0.064 * S, fy + 0.008 * S, DESK + 0.002))
        leg(f"leg_hind_{dx}", at(dx * 0.018, -0.022, -0.014), at(dx * 0.042, -0.046, 0.016),
            (fx + dx * 0.054 * S, fy - 0.066 * S, DESK + 0.002))
    leg("leg_front_-1", at(-0.016, 0.022, -0.012), at(-0.04, 0.045, 0.02), (fx - 0.05 * S, fy + 0.068 * S,
                                                                            DESK + 0.002))
    # the pressing leg hangs from a hip pivot, so a small rotation lifts the foot off the key or presses it
    hip = at(0.016, 0.02, -0.014)
    foot = (sx_ + 0.04, sy_ - 0.003, key_top + 0.002)
    knee = ((hip[0] + foot[0]) / 2 + 0.014 * S, (hip[1] + foot[1]) / 2 - 0.01 * S, max(hip[2], foot[2]) + 0.03 * S)
    bpy.ops.object.empty_add(location=hip)
    pivot = part(bpy.context.object)
    pivot.name = "press_pivot"
    leg("leg_front_press", hip, knee, foot, parent=pivot)

    # presses: the model's own decisions, frame by frame (results/clip/timeline.json)
    reach = math.dist(hip[1:], foot[1:])
    lift = math.atan2(0.014, reach)                 # foot ~14 mm above the key when not pressing
    down = -math.atan2(0.004, reach)                # and the key's 4 mm travel when pressing
    space_up = space.location.z
    # the space bar flashes cyan while pressed, so every press shows even when the leg is half hidden
    flash = material("space_flash", (0.03, 0.03, 0.035), rough=0.45, emit=(0.2, 0.85, 1.0), strength=0.0)
    space.data.materials[0] = flash
    fsock = flash.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    prev = None
    for f, p in enumerate(PRESSED, start=1):
        if p == prev:
            continue
        if prev is not None:                        # hold the old state until the frame before
            pivot.rotation_euler.x = down if prev else lift
            pivot.keyframe_insert("rotation_euler", index=0, frame=f - 1)
            space.location.z = space_up - (0.004 if prev else 0.0)
            space.keyframe_insert("location", index=2, frame=f - 1)
            fsock.default_value = 6.0 if prev else 0.0
            fsock.keyframe_insert("default_value", frame=f - 1)
        pivot.rotation_euler.x = down if p else lift
        pivot.keyframe_insert("rotation_euler", index=0, frame=f)
        space.location.z = space_up - (0.004 if p else 0.0)
        space.keyframe_insert("location", index=2, frame=f)
        fsock.default_value = 6.0 if p else 0.0
        fsock.keyframe_insert("default_value", frame=f)
        prev = p

    xs = timeline["shadow_x"]
    death = max(range(N), key=lambda i: xs[i] if xs[i] is not None else -1) + 1
    for hinge, dx in wings:
        hinge.driver_add("rotation_euler", 1).driver.expression = (
            f"{dx} * (0.04 * sin(frame * 0.21 + {dx}) + (0.55 * sin(frame * 2.9) if {death} <= frame <= {death + 18} "
            f"else 0))")
    body.driver_add("location", 2).driver.expression = f"{fz} + 0.0015 * sin(frame * 0.15)"
    area_light("fly_fill", (fx + 0.3, fy - 0.3, fz + 0.3), (math.radians(55), 0, math.radians(45)), 0.35, 3.5,
               (0.8, 0.75, 1.0))
    area_light("fly_rim", (fx - 0.2, fy + 0.2, fz + 0.22), (math.radians(-55), 0, math.radians(-135)), 0.25, 7,
               (0.7, 0.3, 1.0))

# ---------------------------------------------------------------- output
blend = os.path.join(CLIP, f"scene{TAG}.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
if "--still" in argv:
    f = int(argv[argv.index("--still") + 1])
    scene.frame_set(f)
    scene.render.filepath = os.path.join(CLIP, f"preview_{'_'.join(STAGES)}{TAG}_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
if "--render" in argv:
    os.makedirs(os.path.join(CLIP, f"render{TAG}"), exist_ok=True)
    scene.render.filepath = os.path.join(CLIP, f"render{TAG}", "")
    bpy.ops.render.render(animation=True)
