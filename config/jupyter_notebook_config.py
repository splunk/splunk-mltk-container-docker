import os
from subprocess import check_call


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return  # only do this for notebooks
    # split in directory and file name
    nb_path, nb_filename = os.path.split(os_path)
    # split out filename
    nb_name = os.path.splitext(nb_filename)[0]
    # add .py extension for target python module
    py_name = nb_name + ".py"
    # defined modules path in /srv (hardcoded to prevent notebooks subfolder relative problems)
    py_path = "/srv/app/model/"
    # notebook config path in /srv (hardcoded to prevent notebooks subfolder relative problems)
    nb_template = "~/.jupyter/jupyter_notebook_conversion.tpl"
    print("Config path: " + nb_template)
    print("Source path: " + os_path)
    print("Destination: " + py_path)
    # convert notebook to python module using the provided template
    # jupyter nbconvert --to python /srv/notebooks/Splunk_MLTK_notebook.ipynb --output-dir /src/models --template=/srv/config/jupyter_notebook_conversion.tpl
    check_call(['jupyter', 'nbconvert', '--to', 'python', nb_filename,
                '--output-dir', py_path, '--template='+nb_template], cwd=nb_path)

    if nb_filename == "algo.ipynb":
        notebook_version_file = os_path+".version"
        try:
            with open(notebook_version_file, 'r') as f:
                version = int(f.read())
        except FileNotFoundError:
            version = 0
        version += 1
        with open(notebook_version_file, "w") as f:
            f.write("%s" % version)

        python_version_file = os.path.join(py_path, py_name)+".version"
        with open(python_version_file, "w") as f:
            f.write("%s" % version)
        #print("increased source code version number to %s" % version)


c.FileContentsManager.post_save_hook = post_save

# TODO change PW to your own secret
# generate your own PW in python:
# from notebook.auth import passwd
# passwd()
c.NotebookApp.password = 'sha1:f7432152c71d:e8520c26b9d960e838d562768c1d24ef5b9b76c7'
# "Splunk4DeepLearning"
