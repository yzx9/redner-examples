import pyredner
import matplotlib.pyplot as plt
import torch

# Now we load the obj file using pyredner.load_obj. Since we set return_objects=True,
# load_obj returns a list of redner "Object", which contains geometry, material, and lighting information.
objects = pyredner.load_obj('./teapot/teapot.obj', return_objects=True)

camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
scene = pyredner.Scene(camera=camera, objects=objects)

img = pyredner.render_albedo(scene)
# Visualize img
# Need to gamma compress the image for displaying.
plt.imshow(torch.pow(img, 1.0/2.2).cpu())
plt.show()
