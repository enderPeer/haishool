"""Render saved cell anatomy in an isolated Blender background process.

Example (the caller must use --background --factory-startup)::

    blender --background --factory-startup --python scripts/render_anatomy_blender.py -- \
        --anatomy runs/anatomy.json --out runs/anatomy-render --preview --samples 16

Final output is 3840 x 2160; --preview is 960 x 540. The surface is a voxel
union of the saved cell spheres, never a separately designed body envelope.
Materials, microscopic bump, camera, lighting and staging are display choices.
They do not reconstruct anatomy or pigmentation from the earlier world model.

The .blend retains the exact input JSON and an individually addressable hidden
sphere for every source cell. The optional mesh export contains the union only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


RENDERER_VERSION = "stored-cell-renderer-v1"
SURFACE_METHOD = "icosphere-cell-balls-voxel-union-v1"
PALETTE = {
    "structural": (0.39, 0.48, 0.32, 1.0),
    "neural": (0.25, 0.36, 0.48, 1.0),
    "sensory": (0.62, 0.43, 0.21, 1.0),
    "contractile": (0.48, 0.25, 0.19, 1.0),
    "digestive": (0.42, 0.42, 0.23, 1.0),
    "metabolic": (0.48, 0.39, 0.29, 1.0),
    "protective": (0.48, 0.50, 0.39, 1.0),
    "reproductive": (0.49, 0.32, 0.34, 1.0),
    "unspecified": (0.40, 0.43, 0.38, 1.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_anatomy(data: dict) -> list[dict]:
    """Validate geometric source data without making up or moving any cell."""
    if not isinstance(data, dict) or data.get("schema_version") != "cell-anatomy-v1":
        raise ValueError("expected a cell-anatomy-v1 document")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("anatomy must contain at least one saved cell")
    ids = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each cell must be an object")
        identifier = cell.get("id")
        if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier in ids:
            raise ValueError("cell IDs must be distinct integers")
        ids.add(identifier)
        position = cell.get("position")
        if not isinstance(position, list) or len(position) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in position
        ):
            raise ValueError(f"cell {identifier} requires three finite saved coordinates")
        radius = cell.get("radius")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"cell {identifier} requires a positive finite saved radius")
        if not isinstance(cell.get("tissue_role"), str) or not cell["tissue_role"]:
            raise ValueError(f"cell {identifier} requires a saved tissue_role")
        if not isinstance(cell.get("roles", []), list) or any(not isinstance(role, str) for role in cell.get("roles", [])):
            raise ValueError(f"cell {identifier} has invalid role tags")
    traits = data.get("source", {}).get("traits", {})
    genes = traits.get("genes") or {}
    sensors = traits.get("sensors") or {}
    eye_level = genes.get("eyes", (sensors.get("eyes") or {}).get("level"))
    if eye_level == 0 and any("eyes" in cell.get("roles", []) for cell in cells):
        raise ValueError("zero eyes gene conflicts with stored eye-role cells")
    return cells


def bounds_of(cells: list[dict]) -> tuple[list[float], list[float]]:
    return ([min(cell["position"][axis] - cell["radius"] for cell in cells) for axis in range(3)],
            [max(cell["position"][axis] + cell["radius"] for cell in cells) for axis in range(3)])


def voxel_size_for(cells: list[dict], preview: bool, requested: float | None = None) -> float:
    if requested is not None:
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("voxel size must be a positive finite number")
        return requested
    low, high = bounds_of(cells)
    longest = max(b - a for a, b in zip(low, high))
    return max(min(cell["radius"] for cell in cells) / (6.0 if preview else 10.0),
               longest / (256.0 if preview else 384.0))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anatomy", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True, help="output directory, or a particular .png path")
    result.add_argument("--samples", type=int, default=64)
    result.add_argument("--preview", action="store_true", help="960x540 QA image instead of 3840x2160")
    result.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "optix"), default="auto")
    result.add_argument("--mesh", choices=("glb", "ply", "none"), default="glb")
    result.add_argument("--voxel-size", type=float, help="surface approximation resolution, in source units")
    result.add_argument("--render-seed", type=int, default=85)
    result.add_argument("--label", help="optional presentation label, e.g. World 85; otherwise source world_seed")
    result.add_argument("--no-caption", action="store_true")
    result.add_argument("--cell-view", action="store_true", help="show exact saved sphere instances, instead of the derived union")
    return result


def _input(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def _material(bpy, name, color, cell_radius, *, finish=True):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = nodes.get("Principled BSDF")
    _input(principled, "Base Color", color)
    _input(principled, "Roughness", 0.33 if finish else 0.85)
    _input(principled, "Metallic", 0.0)
    _input(principled, "Specular IOR Level", 0.32 if finish else 0.15)
    _input(principled, "Subsurface Weight", 0.23 if finish else 0.0)
    _input(principled, "Subsurface Radius", (0.75, 0.44, 0.25))
    _input(principled, "Subsurface Scale", cell_radius * 0.24)
    _input(principled, "Coat Weight", 0.16 if finish else 0.0)
    _input(principled, "Coat Roughness", 0.23)
    if finish:
        coordinate = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "3D"
        _input(noise, "Scale", 26.0)
        _input(noise, "Detail", 3.0)
        _input(noise, "Roughness", 0.6)
        bump = nodes.new("ShaderNodeBump")
        _input(bump, "Strength", 0.20)
        _input(bump, "Distance", cell_radius * 0.018)
        links.new(coordinate.outputs["Object"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    material["provenance"] = "display palette and surface finish; not simulated pigmentation or tissue optics"
    return material


def _configure_render(bpy, args) -> dict:
    scene = bpy.context.scene
    width, height = (960, 540) if args.preview else (3840, 2160)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.engine = "CYCLES" if args.engine == "cycles" else "BLENDER_EEVEE"
    device_info = {"requested": args.device, "used": "CPU", "devices": [], "warnings": []}
    if args.engine == "cycles":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.seed = args.render_seed
        scene.cycles.use_animated_seed = False
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transmission_bounces = 6
        scene.cycles.transparent_max_bounces = 4
        scene.cycles.adaptive_threshold = 0.03 if args.preview else 0.012
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if args.device != "cpu":
            choices = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL") if args.device == "auto" else (args.device.upper(),)
            for backend in choices:
                try:
                    prefs.compute_device_type = backend
                    prefs.refresh_devices()
                    accelerated = [device for device in prefs.devices if device.type != "CPU"]
                    if accelerated:
                        for device in prefs.devices:
                            device.use = device.type != "CPU"
                        scene.cycles.device = "GPU"
                        device_info.update(used=backend, devices=[device.name for device in accelerated])
                        break
                except Exception as exc:
                    device_info["warnings"].append(f"{backend}: {exc}")
            else:
                if args.device != "auto":
                    raise RuntimeError(f"requested {args.device} accelerator unavailable; use --device cpu or auto")
                scene.cycles.device = "CPU"
                device_info["warnings"].append("No available accelerator; explicit CPU fallback.")
        else:
            scene.cycles.device = "CPU"
    else:
        device_info["used"] = "EEVEE graphics backend"
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = args.samples
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        pass
    scene.view_settings.exposure = 0.35
    scene.view_settings.gamma = 1.0
    scene.unit_settings.system = "NONE"
    return {"width": width, "height": height, "engine": scene.render.engine, "samples": args.samples,
            "render_seed": args.render_seed, "denoising": args.engine == "cycles", "device": device_info,
            "view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
            "exposure": scene.view_settings.exposure}


def _create_surface(bpy, cells, materials, role_indices, voxel_size, cell_view):
    import bmesh
    import numpy as np
    from mathutils import Matrix

    mesh = bpy.data.meshes.new("Saved cell balls - source geometry")
    bm = bmesh.new()
    for cell in cells:
        geometry = bmesh.ops.create_icosphere(bm, subdivisions=4, radius=cell["radius"],
                                             matrix=Matrix.Translation(cell["position"]))
        for vertex in geometry["verts"]:
            for face in vertex.link_faces:
                face.material_index = role_indices[cell["tissue_role"]]
    bm.to_mesh(mesh)
    bm.free()
    body = bpy.data.objects.new("Stored cell anatomy - sphere union", mesh)
    bpy.context.scene.collection.objects.link(body)
    for material in materials:
        mesh.materials.append(material)
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    before = {"vertices": len(mesh.vertices), "polygons": len(mesh.polygons)}
    if not cell_view:
        mesh.remesh_voxel_size = voxel_size
        mesh.remesh_voxel_adaptivity = 0.0
        mesh.use_remesh_preserve_volume = False
        if hasattr(mesh, "use_remesh_preserve_attributes"):
            mesh.use_remesh_preserve_attributes = True
        bpy.ops.object.voxel_remesh()
        # Attribute transfer in Blender is version-dependent. Assign every new
        # face from the nearest saved ball surface, never invent a tissue role.
        mesh = body.data
        centers = np.array([cell["position"] for cell in cells], dtype=np.float64)
        radii = np.array([cell["radius"] for cell in cells], dtype=np.float64)
        source_roles = np.array([role_indices[cell["tissue_role"]] for cell in cells], dtype=np.int32)
        face_centers = np.empty(len(mesh.polygons) * 3, dtype=np.float32)
        mesh.polygons.foreach_get("center", face_centers)
        face_centers = face_centers.reshape((-1, 3))
        assignment = np.empty(len(mesh.polygons), dtype=np.int32)
        for start in range(0, len(face_centers), 2048):
            block = face_centers[start:start + 2048]
            signed = np.linalg.norm(block[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
            assignment[start:start + len(block)] = source_roles[np.argmin(signed, axis=1)]
        mesh.polygons.foreach_set("material_index", assignment)
    for face in mesh.polygons:
        face.use_smooth = True
    mesh.update()
    body["source_cell_count"] = len(cells)
    body["surface_method"] = "saved icospheres, no union" if cell_view else SURFACE_METHOD
    body["voxel_size"] = 0.0 if cell_view else voxel_size
    body["units"] = "source units; no metres or real organism dimensions inferred"
    return body, {"source_sphere_subdivisions": 4, "before": before,
                  "after": {"vertices": len(mesh.vertices), "polygons": len(mesh.polygons)},
                  "voxel_size": None if cell_view else voxel_size,
                  "adaptivity": 0.0, "preserve_volume": False, "smoothing": "normals only; no surface displacement",
                  "tissue_assignment": "nearest saved cell sphere signed distance at face center"}


def _store_exact_cells(bpy, cells, materials, role_indices):
    import bmesh

    collection = bpy.data.collections.new("Exact stored cells - hidden source spheres")
    bpy.context.scene.collection.children.link(collection)
    unit_mesh = bpy.data.meshes.new("Source sphere radius one")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    bm.to_mesh(unit_mesh)
    bm.free()
    unit_mesh.materials.append(materials[0])
    for face in unit_mesh.polygons:
        face.use_smooth = True
    for cell in cells:
        obj = bpy.data.objects.new(f"cell_{cell['id']}", unit_mesh)
        collection.objects.link(obj)
        obj.location = cell["position"]
        obj.scale = (cell["radius"],) * 3
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials[role_indices[cell["tissue_role"]]]
        obj["cell_id"] = cell["id"]
        obj["parent_id"] = str(cell.get("parent_id"))
        obj["source_radius"] = cell["radius"]
        obj["tissue_role"] = cell["tissue_role"]
        obj["roles_json"] = json.dumps(cell.get("roles", []))
        obj.hide_render = True
    collection.hide_render = True
    collection.hide_viewport = True


def _volume_diagnostics(mesh, cells, cell_view):
    """Approximation diagnostics in source cubic units, not a conservation test.

    Summed sphere volumes are an upper bound on the exact union volume; the
    exact overlap-corrected union is not calculated here. Signed mesh volume
    depends on closed, consistently wound surfaces, so report those properties.
    """
    import bmesh

    sphere_sum = math.fsum((4.0 / 3.0) * math.pi * cell["radius"] ** 3 for cell in cells)
    measured = bmesh.new()
    try:
        measured.from_mesh(mesh)
        signed_volume = measured.calc_volume(signed=True)
        boundary_edges = sum(edge.is_boundary for edge in measured.edges)
        nonmanifold_edges = sum(not edge.is_manifold for edge in measured.edges)
        inconsistent_edges = sum(edge.is_manifold and not edge.is_contiguous for edge in measured.edges)
    finally:
        measured.free()
    return {
        "units": "source length units cubed",
        "source_sphere_volume_sum": sphere_sum,
        "source_sphere_sum_meaning": "sum before overlap removal; upper bound on exact ball-union volume",
        "signed_surface_volume": signed_volume,
        "signed_surface_volume_over_sphere_sum": signed_volume / sphere_sum,
        "signed_relative_difference_from_sphere_sum": signed_volume / sphere_sum - 1.0,
        "surface_measurement": "signed polyhedral mesh volume in unchanged source coordinates",
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "inconsistently_wound_edges": inconsistent_edges,
        "interpretation": (
            "Unmerged sphere mesh volume double-counts overlaps; this is not a union-volume measurement."
            if cell_view else
            "Voxel union is an approximation. A positive difference from the sphere-sum upper bound "
            "indicates geometric inflation. A negative difference includes real overlap removal as well "
            "as approximation error. This is not a mass- or volume-conservation claim."
        ),
    }


def _stage(bpy, cells, args, label):
    from mathutils import Vector

    low, high = bounds_of(cells)
    center = Vector([(a + b) / 2 for a, b in zip(low, high)])
    span = max(b - a for a, b in zip(low, high))
    radius = sum(cell["radius"] for cell in cells) / len(cells)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Neutral studio environment")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.045, 0.055, 0.050, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=span * 200, location=(center.x, center.y, low[2] - radius * 0.08))
    floor = bpy.context.object
    floor.name = "Studio backdrop - display only"
    floor.data.materials.append(_material(bpy, "Matte neutral studio", (0.055, 0.067, 0.058, 1.0), radius, finish=False))

    direction = Vector((1.25, -2.4, 1.15)).normalized()
    rotation = (-direction).to_track_quat("-Z", "Y")
    right, up = rotation @ Vector((1, 0, 0)), rotation @ Vector((0, 1, 0))
    def extent(axis):
        points = [(Vector(cell["position"]) - center).dot(axis) for cell in cells]
        return (min(point - cell["radius"] for point, cell in zip(points, cells)),
                max(point + cell["radius"] for point, cell in zip(points, cells)))
    horizontal, vertical, depth = extent(right), extent(up), extent(direction)
    projected_w, projected_h = horizontal[1] - horizontal[0], vertical[1] - vertical[0]
    lens, sensor_width, aspect = 70.0, 36.0, 16 / 9
    distance = max(projected_w / (sensor_width / lens * 0.76),
                   projected_h / (sensor_width / lens / aspect * 0.70)) + depth[1]
    camera_data = bpy.data.cameras.new("Anatomy studio camera")
    camera = bpy.data.objects.new("Anatomy studio camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + direction * distance
    camera.rotation_euler = rotation.to_euler()
    camera_data.type, camera_data.lens, camera_data.sensor_width = "PERSP", lens, sensor_width
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start, camera_data.clip_end = 0.01, distance + span * 500
    camera_data.dof.use_dof = False  # all saved cell geometry stays inspectable
    scene.camera = camera
    lighting = []
    for name, offset, energy, size, color in (
        ("Key softbox", (-0.8, -1.2, 1.8), 850, 0.85, (1.0, 0.88, 0.72)),
        ("Cool fill", (1.4, -0.4, 0.5), 300, 1.1, (0.72, 0.84, 1.0)),
        ("Rim softbox", (-0.3, 1.1, 1.5), 1100, 0.65, (0.90, 1.0, 0.92)),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy * (span / 5) ** 2
        light_data.shape, light_data.size, light_data.color = "DISK", span * size, color
        obj = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(offset) * span
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        lighting.append({"name": name, "position": list(obj.location), "energy": light_data.energy,
                         "size": light_data.size, "color": list(color)})
    caption = None
    if not args.no_caption:
        caption = _captions(bpy, camera, label)
    bpy.context.view_layer.update()
    return {"camera": {"position": list(camera.location), "rotation_euler": list(camera.rotation_euler),
                       "target": list(center), "lens_mm": lens, "sensor_width_mm": sensor_width,
                       "depth_of_field": False},
            "lights": lighting, "backdrop": "neutral matte floor; not a simulated habitat",
            "caption": caption}


def _captions(bpy, camera, label):
    from mathutils import Vector

    material = bpy.data.materials.new("Camera caption - display only")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.77, 0.82, 0.76, 1)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    distance = 0.8
    width = distance * camera.data.sensor_width / camera.data.lens
    height = width * 9 / 16
    rotation = camera.rotation_euler.to_quaternion()
    title = (label + " / " if label else "") + "stored cell anatomy v1"
    subtitle = "New cell-based model; not atom-by-atom reconstruction"
    finish = "Tissue colours and surface finish are display choices"
    for name, line, y, pixels in (("Caption title", title, .43, 44),
                                  ("Caption limitation", subtitle, -.425, 30),
                                  ("Caption finish", finish, -.449, 25)):
        curve = bpy.data.curves.new(name, "FONT")
        curve.body, curve.align_x, curve.align_y = line, "LEFT", "CENTER"
        curve.size = width * pixels / 3840
        obj = bpy.data.objects.new(name, curve)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = camera.location + rotation @ Vector((-width * .445, height * y, -distance))
        obj.rotation_euler = camera.rotation_euler
        curve.materials.append(material)
        for attribute in ("visible_shadow", "visible_diffuse", "visible_glossy", "visible_transmission"):
            if hasattr(obj, attribute):
                setattr(obj, attribute, False)
    return [title, subtitle, finish]


def main(argv=None):
    args = parser().parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples must be positive")
    import bpy
    if not bpy.app.background:
        raise RuntimeError("launch in a separate Blender --background --factory-startup process; never run in an open user scene")
    started = time.monotonic()
    source_path = args.anatomy.resolve()
    source_text = source_path.read_text(encoding="utf-8")
    data = json.loads(source_text)
    json.dumps(data, allow_nan=False)  # reject nonfinite metadata before expensive scene work
    cells = validate_anatomy(data)
    source_checksum = sha256(source_path)
    voxel_size = voxel_size_for(cells, args.preview, args.voxel_size)
    if args.out.suffix.lower() == ".png":
        image_path, out = args.out.resolve(), args.out.resolve().parent
    else:
        out = args.out.resolve()
        image_path = out / ("anatomy-preview.png" if args.preview else "anatomy-4k.png")
    out.mkdir(parents=True, exist_ok=True)
    # This process is required to be headless; clearing it cannot modify an open
    # Blender instance. No deletion or overwriting of source assets is performed.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    settings = _configure_render(bpy, args)
    roles = sorted({cell["tissue_role"] for cell in cells})
    role_indices = {role: index for index, role in enumerate(roles)}
    average_radius = sum(cell["radius"] for cell in cells) / len(cells)
    palette = {role: PALETTE.get(role, PALETTE["unspecified"]) for role in roles}
    materials = [_material(bpy, "Display tissue: " + role, palette[role], average_radius) for role in roles]
    body, geometry = _create_surface(bpy, cells, materials, role_indices, voxel_size, args.cell_view)
    geometry["volume_diagnostics"] = _volume_diagnostics(body.data, cells, args.cell_view)
    _store_exact_cells(bpy, cells, materials, role_indices)
    world_seed = data.get("source", {}).get("world_seed", data.get("world_seed"))
    label = args.label if args.label is not None else (f"World {world_seed}" if world_seed is not None else "")
    stage = _stage(bpy, cells, args, label)
    scene = bpy.context.scene
    scene["renderer_version"] = RENDERER_VERSION
    scene["anatomy_sha256"] = source_checksum
    scene["not_a_reconstructed_citizen"] = True
    source_block = bpy.data.texts.new("source-anatomy.json")
    source_block.write(source_text)
    receipt = {
        "renderer_version": RENDERER_VERSION, "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii", errors="replace"),
        "source": {"path": str(source_path), "sha256": source_checksum,
                   "schema_version": data["schema_version"], "algorithm_version": data.get("algorithm_version"),
                   "source_provenance": data.get("source"), "growth_seed": data.get("seed"),
                   "units": data.get("units"), "cell_count": len(cells)},
        "geometry": {"method": "unmerged saved cell spheres" if args.cell_view else SURFACE_METHOD, **geometry,
                     "added_anatomical_features": [], "source_coordinate_transform": "identity",
                     "role_counts": {role: sum(cell["tissue_role"] == role for cell in cells) for role in roles}},
        "render": settings, "staging": stage,
        "materials": {"meaning": "display encodings, not recorded pigmentation or measured tissue optics",
                      "role_rgba": palette, "roughness": .33, "subsurface_weight": .23,
                      "subsurface_scale": average_radius * .24, "coat_weight": .16,
                      "bump": {"source": "procedural 3D noise, display only", "scale": 26.0,
                               "detail": 3.0, "strength": .20, "distance": average_radius * .018}},
        "limitations": ["This is a new cell-based anatomical model, not an atom-by-atom reconstruction.",
                        "The union approximates saved cell balls at the declared voxel resolution.",
                        "Smooth normals and procedural bump change display shading, not cell positions or radii.",
                        "No eyes, limbs, ears, clothing or anatomy were added outside the stored cell geometry.",
                        "The specimen is not an identified citizen of the aggregate society simulation."],
        "outputs": {},
    }
    scene.render.filepath = str(image_path)
    settings_block = bpy.data.texts.new("render-provenance.json")
    settings_block.write(json.dumps(receipt, indent=2, allow_nan=False))
    blend_path = out / ("preview.blend" if args.preview else "saved.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    receipt["outputs"]["blend"] = {"path": str(blend_path), "sha256": sha256(blend_path)}
    if args.mesh != "none":
        bpy.ops.object.select_all(action="DESELECT")
        body.hide_set(False)
        body.select_set(True)
        bpy.context.view_layer.objects.active = body
        mesh_path = out / ("surface." + args.mesh)
        if args.mesh == "glb":
            bpy.ops.export_scene.gltf(filepath=str(mesh_path), export_format="GLB", use_selection=True,
                                      export_yup=False, export_materials="EXPORT", export_extras=True)
        else:
            bpy.ops.wm.ply_export(filepath=str(mesh_path), export_selected_objects=True,
                                  export_normals=True, export_colors="NONE")
        receipt["outputs"]["mesh"] = {"path": str(mesh_path), "sha256": sha256(mesh_path),
                                       "source": "selected derived surface only; staging excluded"}
    print(json.dumps({"status": "rendering", "source_cells": len(cells), "geometry": geometry,
                      "render": settings, "output": str(image_path)}), flush=True)
    bpy.ops.render.render(write_still=True)
    receipt["outputs"]["png"] = {"path": str(image_path), "sha256": sha256(image_path)}
    receipt["seconds"] = round(time.monotonic() - started, 3)
    receipt_path = out / ("preview-receipt.json" if args.preview else "render-receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "receipt": str(receipt_path), "seconds": receipt["seconds"]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
