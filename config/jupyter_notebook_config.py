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
    nb_template = "/dltk/.jupyter/jupyter_notebook_conversion.tpl"
    print("Config path: " + nb_template)
    print("Source path: " + os_path)
    print("Destination: " + py_path)
    # convert notebook to python module using the provided template
    # jupyter nbconvert --to python /srv/notebooks/Splunk_MLTK_notebook.ipynb --output-dir /src/models --template=/srv/config/jupyter_notebook_conversion.tpl
    # /opt/conda/lib/python3.7/site-packages/nbconvert/templates/python.tpl
    # /opt/conda/lib/python3.7/site-packages/nbconvert/templates/skeleton/null.tpl
    check_call(['jupyter', 'nbconvert', '--to', 'python', nb_filename,
               '--output-dir', py_path, '--template=' + nb_template], cwd=nb_path)


c.FileContentsManager.post_save_hook = post_save

# TODO change PW to your own secret
# generate your own PW in python:
# from notebook.auth import passwd
# passwd()
# passwd(os.environ['JUPYTER_PASSWD'],'sha512')
# TODO set from UI and ENV variable generated
c.ServerApp.password = 'sha1:f7432152c71d:e8520c26b9d960e838d562768c1d24ef5b9b76c7'
# "Splunk4DeepLearning"

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = int(os.getenv('JUPYTER_PORT', 8888))
if os.getenv('ENABLE_HTTPS', 'true').lower() == 'true':
    c.ServerApp.certfile = os.getenv('API_SSL_CERT', '/dltk/.jupyter/dltk.pem')
    c.ServerApp.keyfile = os.getenv('API_SSL_KEY', '/dltk/.jupyter/dltk.key')

# Fix for wigets limit default = 1000000
c.ServerApp.iopub_data_rate_limit = 20000000