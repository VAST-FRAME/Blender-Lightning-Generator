bl_info = {
    "name": "Lightning Generator",
    "blender": (4, 3, 0),
    "category": "Add Mesh",
    "version": (1, 0, 0),
    "description": "Procedural lightning bolts with time-mask vertex colors for HDRI rendering",
    "author": "Custom",
}

import bpy
import bmesh
from mathutils import Vector, kdtree
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty,
    StringProperty, PointerProperty
)
from bpy.types import Operator, Panel, PropertyGroup
import random
import math


# ─── Core Generation ────────────────────────────────────────────────────────────

def get_perpendicular_axes(direction):
    """Return two unit vectors orthogonal to direction."""
    d = direction.normalized()
    ref = Vector((1, 0, 0)) if abs(d.x) < 0.9 else Vector((0, 1, 0))
    p1 = d.cross(ref).normalized()
    p2 = d.cross(p1).normalized()
    return p1, p2


def midpoint_displace(points, t_values, iterations, disp_scale, rng):
    """
    Recursive midpoint displacement on a polyline.
    Displacement is perpendicular to each segment, scaled by segment length.
    t_values are linearly interpolated at each new midpoint.
    """
    for _ in range(iterations):
        new_pts = [points[0]]
        new_t = [t_values[0]]

        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            seg = b - a
            seg_len = seg.length

            if seg_len < 1e-7:
                new_pts.append(b.copy())
                new_t.append(t_values[i + 1])
                continue

            p1, p2 = get_perpendicular_axes(seg)
            disp = seg_len * disp_scale
            mid = (a + b) * 0.5
            mid = mid + p1 * rng.uniform(-disp, disp)
            mid = mid + p2 * rng.uniform(-disp * 0.5, disp * 0.5)

            t_mid = (t_values[i] + t_values[i + 1]) * 0.5

            new_pts.append(mid)
            new_t.append(t_mid)
            new_pts.append(b.copy())
            new_t.append(t_values[i + 1])

        points = new_pts
        t_values = new_t

    return points, t_values


def generate_chains(
    start, end,
    main_iters, main_disp,
    fork_prob, fork_max, fork_reach,
    fork_iters, fork_disp,
    current_depth, max_depth,
    t_start, t_end,
    rng
):
    """
    Recursively generate lightning chains.
    Returns list of (points_list, t_values_list).
    Index 0 is always the main bolt of this level.
    t_start / t_end define the time range this bolt occupies (0.0–1.0 for root).
    """
    start = Vector(start)
    end = Vector(end)
    total_len = (end - start).length
    main_dir = (end - start).normalized()

    # --- Main bolt ---
    pts = [start.copy(), end.copy()]
    ts = [t_start, t_end]
    pts, ts = midpoint_displace(pts, ts, main_iters, main_disp, rng)

    all_chains = [(pts, ts)]

    if current_depth >= max_depth:
        return all_chains

    # --- Forks from interior vertices ---
    interior = list(range(1, len(pts) - 1))
    rng.shuffle(interior)

    forks_made = 0
    for idx in interior:
        if forks_made >= fork_max:
            break
        if rng.random() > fork_prob:
            continue

        branch_pos = pts[idx].copy()
        branch_t = ts[idx]

        # Fork direction: biased along main bolt, spread laterally
        p1, p2 = get_perpendicular_axes(main_dir)
        fwd = main_dir * rng.uniform(0.2, 0.7)
        lat = p1 * rng.uniform(-0.8, 0.8) + p2 * rng.uniform(-0.3, 0.3)
        fork_dir = (fwd + lat).normalized()

        fork_len = total_len * fork_reach * rng.uniform(0.35, 1.0)
        fork_end = branch_pos + fork_dir * fork_len

        # Fork t_end: starts at branch t, extends partway toward global t_end
        fork_t_end = branch_t + (t_end - branch_t) * rng.uniform(0.3, 0.85)
        fork_t_end = min(fork_t_end, t_end * 0.99)

        fork_pts = [branch_pos.copy(), fork_end]
        fork_ts = [branch_t, fork_t_end]
        fork_pts, fork_ts = midpoint_displace(
            fork_pts, fork_ts,
            max(1, fork_iters - current_depth),
            fork_disp, rng
        )
        all_chains.append((fork_pts, fork_ts))
        forks_made += 1

        # Recurse for sub-forks
        if current_depth + 1 < max_depth and len(fork_pts) > 2:
            sub = generate_chains(
                fork_pts[0], fork_pts[-1],
                max(1, fork_iters - 1), fork_disp * 0.8,
                fork_prob * 0.55, max(1, fork_max // 2), fork_reach * 0.6,
                max(1, fork_iters - 1), fork_disp * 0.75,
                current_depth + 1, max_depth,
                branch_t, fork_t_end,
                rng
            )
            all_chains.extend(sub[1:])  # skip sub's main chain, we already have it

    return all_chains


# ─── Mesh Builder ───────────────────────────────────────────────────────────────

def build_mesh(chains, name, bolt_length, bolt_start, bolt_end, add_skin, skin_radius_pct):
    """
    Build a Blender mesh from chains.
    Each chain is a polyline. Junction vertices are welded so forks are
    topologically connected to their parent bolt.
    Vertex color attribute 'LightningTime' stores t_value as greyscale.
    """
    all_verts = []
    all_edges = []
    all_t = []
    edge_set = set()
    vert_map = {}

    axis = (Vector(bolt_end) - Vector(bolt_start))
    axis_len_sq = max(axis.length_squared, 1e-12)

    def linear_t_from_point(p):
        rel = Vector(p) - Vector(bolt_start)
        t = rel.dot(axis) / axis_len_sq
        return max(0.0, min(1.0, float(t)))

    def get_or_add_vert_idx(p):
        key = (p.x, p.y, p.z)
        idx = vert_map.get(key)
        if idx is None:
            idx = len(all_verts)
            vert_map[key] = idx
            all_verts.append((p.x, p.y, p.z))
            all_t.append(linear_t_from_point(p))
        return idx

    def add_edge(i0, i1):
        if i0 == i1:
            return
        e = (i0, i1) if i0 < i1 else (i1, i0)
        if e in edge_set:
            return
        edge_set.add(e)
        all_edges.append((i0, i1))

    def write_time_attribute(mesh_data, time_values):
        existing = mesh_data.color_attributes.get("LightningTime")
        if existing:
            mesh_data.color_attributes.remove(existing)

        color_attr = mesh_data.color_attributes.new(
            name="LightningTime",
            type='FLOAT_COLOR',
            domain='POINT'
        )

        count = min(len(color_attr.data), len(time_values))
        for i in range(count):
            t = max(0.0, min(1.0, float(time_values[i])))
            color_attr.data[i].color = (t, t, t, 1.0)

    def bake_time_to_final_mesh(mesh_data, source_verts, source_t):
        if not source_verts or not mesh_data.vertices:
            return

        kd = kdtree.KDTree(len(source_verts))
        for i, co in enumerate(source_verts):
            kd.insert(Vector(co), i)
        kd.balance()

        final_t = []
        for v in mesh_data.vertices:
            _, idx, _ = kd.find(v.co)
            final_t.append(source_t[idx])

        write_time_attribute(mesh_data, final_t)

    for pts, ts in chains:
        if not pts:
            continue

        prev_idx = get_or_add_vert_idx(pts[0])
        for i in range(1, len(pts)):
            cur_idx = get_or_add_vert_idx(pts[i])
            add_edge(prev_idx, cur_idx)
            prev_idx = cur_idx

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(all_verts, all_edges, [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # --- Skin Modifier ---
    if add_skin:
        # Set skin vertex radii via bmesh before adding modifier
        bm = bmesh.new()
        bm.from_mesh(mesh)
        skin_layer = bm.verts.layers.skin.verify()
        base_r = bolt_length * (skin_radius_pct / 100.0)
        for v in bm.verts:
            v[skin_layer].radius = (base_r, base_r)
        # Mark root at very first vertex (main bolt start)
        if bm.verts:
            bm.verts.ensure_lookup_table()
            bm.verts[0][skin_layer].use_root = True
        bm.to_mesh(mesh)
        bm.free()

        skin_mod = obj.modifiers.new("Skin", 'SKIN')
        skin_mod.use_smooth_shade = True

        sub_mod = obj.modifiers.new("Subdivision", 'SUBSURF')
        sub_mod.levels = 1
        sub_mod.render_levels = 2

        # Apply generated geometry so color data can be written on final verts.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(
            obj_eval,
            preserve_all_data_layers=False,
            depsgraph=depsgraph
        )
        old_mesh = obj.data
        obj.modifiers.clear()
        obj.data = baked_mesh
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

        bake_time_to_final_mesh(obj.data, all_verts, all_t)
    else:
        write_time_attribute(mesh, all_t)

    return obj


# ─── Operators ──────────────────────────────────────────────────────────────────

def run_generation(context, props):
    asset_obj = context.object
    if not asset_obj:
        return None, "Select a Lightning Asset object"

    start_obj = props.start_obj
    end_obj = props.end_obj

    if not start_obj or not end_obj:
        return None, "Set both Start and End objects in the panel"

    start = start_obj.matrix_world.translation.copy()
    end = end_obj.matrix_world.translation.copy()
    bolt_length = (end - start).length

    if bolt_length < 1e-4:
        return None, "Start and End objects are too close"

    rng = random.Random(props.seed)

    chains = generate_chains(
        start, end,
        main_iters=props.main_iterations,
        main_disp=props.displacement_scale,
        fork_prob=props.fork_probability,
        fork_max=props.fork_count_max,
        fork_reach=props.fork_reach,
        fork_iters=props.fork_iterations,
        fork_disp=props.fork_disp_scale,
        current_depth=0,
        max_depth=props.max_fork_depth,
        t_start=0.0,
        t_end=1.0,
        rng=rng
    )

    # Replace only this asset's previous output object.
    prev_obj = bpy.data.objects.get(props.generated_obj_name)
    if prev_obj:
        bpy.data.objects.remove(prev_obj, do_unlink=True)

    output_name = f"{asset_obj.name}_Lightning"

    obj = build_mesh(
        chains, output_name,
        bolt_length,
        start, end,
        add_skin=props.add_skin_modifier,
        skin_radius_pct=props.skin_radius_pct
    )
    props.generated_obj_name = obj.name

    total_verts = sum(len(c[0]) for c in chains)
    msg = f"Generated {len(chains)} chains ({total_verts} verts)"
    return obj, msg


class LIGHTNING_OT_generate(Operator):
    bl_idname = "mesh.lightning_generate"
    bl_label = "Generate Lightning"
    bl_description = "Generate lightning bolt from Start to End object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        asset_obj = context.object
        if not asset_obj:
            self.report({'ERROR'}, "Select a Lightning Asset object")
            return {'CANCELLED'}

        props = asset_obj.lightning_props
        obj, msg = run_generation(context, props)

        if obj is None:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        asset_obj.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = asset_obj
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class LIGHTNING_OT_regenerate(Operator):
    """Increment seed and regenerate — for quick variation browsing."""
    bl_idname = "mesh.lightning_regenerate"
    bl_label = "New Variation"
    bl_description = "Increment seed and regenerate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        asset_obj = context.object
        if not asset_obj:
            self.report({'ERROR'}, "Select a Lightning Asset object")
            return {'CANCELLED'}

        props = asset_obj.lightning_props
        props.seed += 1

        obj, msg = run_generation(context, props)
        if obj is None:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        asset_obj.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = asset_obj
        self.report({'INFO'}, f"Seed {props.seed} → {msg}")
        return {'FINISHED'}


class LIGHTNING_OT_create_asset(Operator):
    bl_idname = "mesh.lightning_create_asset"
    bl_label = "Create Lightning Asset"
    bl_description = "Create an Empty object that stores lightning settings"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        asset = bpy.data.objects.new("LightningAsset", None)
        context.collection.objects.link(asset)
        asset.empty_display_type = 'PLAIN_AXES'
        asset.empty_display_size = 0.3

        bpy.ops.object.select_all(action='DESELECT')
        asset.select_set(True)
        context.view_layer.objects.active = asset

        self.report({'INFO'}, "Created Lightning Asset")
        return {'FINISHED'}


# ─── Panel ──────────────────────────────────────────────────────────────────────

class LIGHTNING_PT_panel(Panel):
    bl_label = "Lightning Generator"
    bl_idname = "LIGHTNING_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lightning"

    def draw(self, context):
        layout = self.layout
        asset_obj = context.object

        row = layout.row(align=True)
        row.operator("mesh.lightning_create_asset", icon='EMPTY_AXIS', text="New Asset")

        if not asset_obj:
            layout.label(text="Select a Lightning Asset object", icon='INFO')
            return

        props = asset_obj.lightning_props
        layout.label(text=f"Asset: {asset_obj.name}", icon='OBJECT_DATA')

        # Points
        box = layout.box()
        box.label(text="Points", icon='EMPTY_AXIS')
        box.prop(props, "start_obj", text="Start")
        box.prop(props, "end_obj", text="End")

        # Main bolt
        box = layout.box()
        box.label(text="Main Bolt", icon='FORCE_CURVE')
        col = box.column(align=True)
        col.prop(props, "main_iterations")
        col.prop(props, "displacement_scale")

        # Forks
        box = layout.box()
        box.label(text="Forks", icon='OUTLINER_OB_CURVE')
        col = box.column(align=True)
        col.prop(props, "fork_probability")
        col.prop(props, "fork_count_max")
        col.prop(props, "fork_reach")
        col.prop(props, "fork_iterations")
        col.prop(props, "fork_disp_scale")
        col.prop(props, "max_fork_depth")

        # Mesh
        box = layout.box()
        box.label(text="Mesh / Output", icon='MESH_DATA')
        col = box.column(align=True)
        col.prop(props, "add_skin_modifier")
        if props.add_skin_modifier:
            col.prop(props, "skin_radius_pct")

        # Seed
        layout.separator()
        row = layout.row(align=True)
        row.prop(props, "seed")

        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.6
        row.operator("mesh.lightning_generate", icon='LIGHT', text="Generate")
        row.operator("mesh.lightning_regenerate", icon='FILE_REFRESH', text="New Variation")

        # Vertex color reminder
        layout.separator()
        col = layout.column()
        col.scale_y = 0.75
        col.label(text="Vertex attr: 'LightningTime'", icon='INFO')
        col.label(text="  0.00 = bolt start (top)")
        col.label(text="  1.00 = bolt end (tip)")


# ─── Properties ─────────────────────────────────────────────────────────────────

class LightningProperties(PropertyGroup):
    start_obj: PointerProperty(
        name="Start",
        description="Object at the top of the lightning bolt",
        type=bpy.types.Object
    )
    end_obj: PointerProperty(
        name="End",
        description="Object at the bottom / tip of the lightning bolt",
        type=bpy.types.Object
    )
    main_iterations: IntProperty(
        name="Iterations",
        description="Subdivision passes on the main bolt — more = more jagged",
        default=6, min=1, max=10
    )
    displacement_scale: FloatProperty(
        name="Displacement",
        description="Midpoint displacement amount relative to segment length",
        default=0.32, min=0.0, max=2.0, step=1, precision=3
    )
    fork_probability: FloatProperty(
        name="Fork Probability",
        description="Chance of a fork spawning at each interior vertex",
        default=0.18, min=0.0, max=1.0, step=1, precision=2,
        subtype='FACTOR'
    )
    fork_count_max: IntProperty(
        name="Max Forks / Level",
        description="Maximum forks per bolt level",
        default=6, min=0, max=30
    )
    fork_reach: FloatProperty(
        name="Fork Reach",
        description="Fork length as fraction of main bolt length",
        default=0.22, min=0.01, max=1.0, step=1, precision=2,
        subtype='FACTOR'
    )
    fork_iterations: IntProperty(
        name="Fork Iterations",
        description="Subdivision passes on forks",
        default=4, min=1, max=8
    )
    fork_disp_scale: FloatProperty(
        name="Fork Displacement",
        description="Midpoint displacement scale for forks",
        default=0.42, min=0.0, max=2.0, step=1, precision=3
    )
    max_fork_depth: IntProperty(
        name="Fork Depth",
        description="Recursion depth — 0 = no forks, 1 = forks, 2 = forks-of-forks",
        default=2, min=0, max=3
    )
    add_skin_modifier: BoolProperty(
        name="Add Skin Modifier",
        description="Add Skin + Subdivision modifiers for renderable geometry",
        default=True
    )
    skin_radius_pct: FloatProperty(
        name="Skin Radius %",
        description="Skin vertex radius as percentage of total bolt length",
        default=0.35, min=0.01, max=5.0, step=1, precision=2,
        subtype='PERCENTAGE'
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed — change for different bolt shapes",
        default=42, min=0, max=99999
    )
    generated_obj_name: StringProperty(
        name="Generated Object",
        description="Internal: output object created by this asset",
        default=""
    )


# ─── Menu Entry ─────────────────────────────────────────────────────────────────

def menu_func(self, context):
    self.layout.operator(LIGHTNING_OT_generate.bl_idname, icon='LIGHT', text="Lightning Bolt")


# ─── Registration ────────────────────────────────────────────────────────────────

classes = [
    LightningProperties,
    LIGHTNING_OT_generate,
    LIGHTNING_OT_regenerate,
    LIGHTNING_OT_create_asset,
    LIGHTNING_PT_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.lightning_props = PointerProperty(type=LightningProperties)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    del bpy.types.Object.lightning_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
