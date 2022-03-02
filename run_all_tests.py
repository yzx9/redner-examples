import os
from subprocess import call

os.chdir("redner/tests")
g = os.walk(r".")

for path, dir_list, file_list in g:
    for file_name in file_list:
        if file_name.startswith("test_"):
            print(os.path.join(path, file_name))
            call(["python", os.path.join(path, file_name)])
