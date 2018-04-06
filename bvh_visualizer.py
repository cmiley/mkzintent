import bpy
import os

def main():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bvh_data = []

    # Changes the directory to be where the blend file is located.
    path_to_vdata = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(path_to_vdata)

    # TODO: add argparsing to select vdata file.
    vdata_path = bpy.path.abspath('visualization_data.vdata')
    with open(vdata_path, encoding='utf-8-sig') as file:
        lines = file.readlines()

    num_frames = int(lines[0])
    num_look_ahead = int(lines[1])
    bvh_load_path = lines[2].strip()

    data = []

    for line in lines[3:]:
        # X, Y, Z, dx, dy, dz = map(float, line.split(" "))
        data.append(list(map(float, line.split(" "))))


    # CUBE: Actual Position
    bpy.ops.mesh.primitive_cube_add(radius=0.5, location=(0, 0, 0))
    cube = bpy.context.object
    cube.name = 'Cube'
    bpy.context.scene.frame_current = 2
    for i in range(len(data)-num_look_ahead):
        cube.location.xyz = data[i+num_look_ahead][0], -data[i+num_look_ahead][2], data[i+num_look_ahead][1]
        bpy.ops.anim.keyframe_insert_menu(type='Location')
        bpy.context.scene.frame_current += 1

    # Change it back to the actual for when we don't have a future position (near end of loop)
    for i in range(len(data)-num_look_ahead, len(data)):
        cube.location.xyz = data[i][0], -data[i][2], data[i][1]
        bpy.ops.anim.keyframe_insert_menu(type='Location')
        bpy.context.scene.frame_current += 1

    # SPHERE: Predicted Position
    bpy.ops.mesh.primitive_ico_sphere_add(size=0.5, location=(0, 0, 0))
    sphere = bpy.context.object
    sphere.name = 'Sphere'
    bpy.context.scene.frame_current = 2

    for i in range(len(data)):
        sphere.location.xyz = data[i][0] + data[i][3], -data[i][2] - data[i][5], data[i][1] + data[i][4]
        bpy.ops.anim.keyframe_insert_menu(type='Location')
        bpy.context.scene.frame_current += 1

    # import bvh animation
    bpy.ops.import_anim.bvh(filepath=bvh_load_path)
    pose = bpy.context.object

    # deselect all objects and move playhead
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.scene.frame_end = num_frames + 2
    bpy.context.scene.frame_current = 0

    # make predicted red
    red = makeMaterial('Red', (1, 0, 0), (1, 1, 1), 1)
    setMaterial(sphere, red)

    # make actual green
    green = makeMaterial('Green', (0, 1, 0), (1, 1, 1), 1)
    setMaterial(cube, green)


def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    return mat


def setMaterial(ob, mat):
    me = ob.data
    me.materials.append(mat)


if __name__ == "__main__":
    main()
