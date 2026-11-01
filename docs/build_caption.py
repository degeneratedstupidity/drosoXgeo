"""The caption laid over the clip (docs/build_clip.py): a transparent 1080x1080 PNG, text at the bottom.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P docs/build_caption.py     # -> results/clip/caption.png
"""

import os

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = ["Illustration. The screen is a real recorded run of a connectome-constrained",
         "fruit-fly model playing Geometry Dash; every key press is the model's own."]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 1080
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
bpy.ops.object.camera_add(location=(0, 0, 10))
cam = bpy.context.object
cam.data.type = "ORTHO"
cam.data.ortho_scale = 10
scene.camera = cam
font = bpy.data.fonts.load("/System/Library/Fonts/SFNS.ttf")
m = bpy.data.materials.new("caption")
m.use_nodes = True
nt = m.node_tree
nt.nodes.clear()
e = nt.nodes.new("ShaderNodeEmission")
e.inputs["Color"].default_value = (0.92, 0.92, 0.95, 1)
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(e.outputs[0], out.inputs[0])
for i, line in enumerate(LINES):
    bpy.ops.object.text_add(location=(-4.75, -4.35 - i * 0.27, 0))
    t = bpy.context.object
    t.data.body = line
    t.data.font = font
    t.data.size = 0.2
    t.data.materials.append(m)
# a dark band behind the text so it reads on any frame
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -4.55, -0.1))
band = bpy.context.object
band.scale = (10, 0.9, 1)
bm = bpy.data.materials.new("band")
bm.use_nodes = True
b = bm.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0, 0, 0, 1)
b.inputs["Alpha"].default_value = 0.55
bm.surface_render_method = "BLENDED"
band.data.materials.append(bm)
scene.render.filepath = os.path.join(ROOT, "results", "clip", "caption.png")
bpy.ops.render.render(write_still=True)
