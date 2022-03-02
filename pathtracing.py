import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pyredner
import torch
import urllib
import zipfile

if not os.path.isdir('scenes/mori_knob'):
    # wget
    filedata = urllib.request.urlretrieve(
        'https://casual-effects.com/g3d/data10/common/model/mori_knob/mori_knob.zip', 'mori_knob.zip')
    # unzip
    zip_ref = zipfile.ZipFile('mori_knob.zip', 'r')
    zip_ref.extractall('scenes/mori_knob/')
    os.remove('mori_knob.zip')

objects = pyredner.load_obj(
    'scenes/mori_knob/testObj.obj', return_objects=True)
objects[4].material.diffuse_reflectance.texels = torch.tensor(
    (0.4, 0.16, 0.04))
objects[4].material.specular_reflectance.texels = torch.tensor((0.4, 0.4, 0.4))

camera = pyredner.automatic_camera_placement(
    [objects[1], objects[4]], resolution=(768, 1024))
camera.position[1] += 2.5  # Make it higher
camera.position[2] -= 0.5  # Pull farther from the object

scene = pyredner.Scene(camera=camera, objects=objects)
scene.area_lights[0].intensity = torch.tensor((15.0, 15.0, 15.0))

img = pyredner.render_pathtracing(
    scene=scene, num_samples=(64, 4), max_bounces=6)

# Visualize img
plt.figure(figsize=(15, 15))
imshow(img.cpu())
plt.show()
