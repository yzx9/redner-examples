from subprocess import call
import pyredner
import numpy as np
import torch
import scipy.ndimage.filters

# Optimize for material

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load the scene from a Mitsuba scene file
print('scene loading...')
scene = pyredner.load_mitsuba('scenes/living-room-3-without-texture/scene.xml')
print('scene loaded')

max_bounces = 6
args = pyredner.RenderFunction.serialize_scene(
    scene=scene,
    num_samples=512,
    max_bounces=max_bounces)

render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)

pyredner.imwrite(img.cpu(), 'results/living_room_material/target.exr')
pyredner.imwrite(img.cpu(), 'results/living_room_material/target.png')
target = pyredner.imread('results/living_room_material/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device=pyredner.get_device())

light_intensitys = []
for li, l in enumerate(scene.area_lights):
    var = torch.tensor([100.0, 200.0, 200.0],
                       device=pyredner.get_device(), requires_grad=True)
    light_intensitys.append(var)
    l.intensity = var

materials_diffuse = []
materials_specular = []
materials_roughness = []
for mi, m in enumerate(scene.materials):
    diffuse = torch.tensor(
        [0.5, 0.5, 0.5], device=pyredner.get_device(), requires_grad=True)
    specular = torch.tensor(
        [0.5, 0.5, 0.5], device=pyredner.get_device(), requires_grad=True)
    roughness = torch.tensor(
        [0.5], device=pyredner.get_device(), requires_grad=True)

    scene.materials[mi] = pyredner.Material(
        diffuse_reflectance=diffuse.abs(),
        specular_reflectance=specular.abs(),
        roughness=roughness.abs())
    materials_diffuse.append(diffuse)
    materials_specular.append(specular)
    materials_roughness.append(roughness)

args = pyredner.RenderFunction.serialize_scene(
    scene=scene,
    num_samples=512,
    max_bounces=max_bounces)

img = render(1, *args)

ctx = pyredner.RenderFunction.unpack_args([1], args)
pyredner.imwrite(img.cpu(), 'results/living_room_material/init.exr')
pyredner.imwrite(img.cpu(), 'results/living_room_material/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/living_room_material/init_diff.png')

optimizer = torch.optim.Adam(
    light_intensitys+materials_diffuse+materials_specular+materials_roughness, lr=5e-1)
for t in range(10000):
    print('iteration:', t)
    optimizer.zero_grad()

    for mi, m in enumerate(scene.materials):
        scene.materials[mi] = pyredner.Material(
            diffuse_reflectance=materials_diffuse[mi].abs(),
            specular_reflectance=materials_specular[mi].abs(),
            roughness=materials_roughness[mi].abs())

    args = pyredner.RenderFunction.serialize_scene(
        scene=scene,
        num_samples=4,
        max_bounces=max_bounces)

    img = render(t + 1, *args)

    diff = img - target
    dirac = np.zeros([7, 7], dtype=np.float32)
    dirac[3, 3] = 1.0
    dirac = torch.from_numpy(dirac)
    f = np.zeros([3, 3, 7, 7], dtype=np.float32)
    gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
    f[0, 0, :, :] = gf
    f[1, 1, :, :] = gf
    f[2, 2, :, :] = gf
    f = torch.from_numpy(f)
    if pyredner.get_use_gpu():
        f = f.cuda(device=pyredner.get_device())
    m = torch.nn.AvgPool2d(2)

    r = 256
    diff_0 = (img - target).view(1, r, r, 3).permute(0, 3, 2, 1)
    diff_1 = m(torch.nn.functional.conv2d(diff_0, f, padding=3))
    diff_2 = m(torch.nn.functional.conv2d(diff_1, f, padding=3))
    diff_3 = m(torch.nn.functional.conv2d(diff_2, f, padding=3))
    diff_4 = m(torch.nn.functional.conv2d(diff_3, f, padding=3))
    loss = diff_0.pow(2).sum() / (r*r) + \
        diff_1.pow(2).sum() / ((r/2)*(r/2)) + \
        diff_2.pow(2).sum() / ((r/4)*(r/4)) + \
        diff_3.pow(2).sum() / ((r/8)*(r/8)) + \
        diff_4.pow(2).sum() / ((r/16)*(r/16))

    loss.backward()
    optimizer.step()

    if t % 100 == 0:
        print('loss:', loss.item())
        print('intensitys: ', light_intensitys)

        # Save the intermediate render.
        pyredner.imwrite(
            img.cpu(), 'results/living_room_material/iter_{}.png'.format(t))

args = pyredner.RenderFunction.serialize_scene(
    scene=scene,
    num_samples=4,
    max_bounces=max_bounces)
img = render(602, *args)
pyredner.imwrite(img.cpu(), 'results/living_room_material/final.exr')
pyredner.imwrite(img.cpu(), 'results/living_room_material/final.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/living_room_material/final_diff.png')

call(["ffmpeg",
      "-framerate", "24",
      "-i", "results/living_room_material/iter_%d00.png",
      "-vb", "20M", "results/living_room_material/out.mp4"])
