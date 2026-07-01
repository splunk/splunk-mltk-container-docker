import os
import hashlib
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
    #print("Config path: " + nb_template)
    #print("Source path: " + os_path)
    #print("Destination: " + py_path)
    # convert notebook to python module using the provided template
    # jupyter nbconvert --to python /srv/notebooks/Splunk_MLTK_notebook.ipynb --output-dir /src/models --template=/srv/config/jupyter_notebook_conversion.tpl
    # /opt/conda/lib/python3.7/site-packages/nbconvert/templates/python.tpl
    # /opt/conda/lib/python3.7/site-packages/nbconvert/templates/skeleton/null.tpl
    check_call(['jupyter', 'nbconvert', '--to', 'python', nb_filename,
               '--output-dir', py_path, '--template=' + nb_template], cwd=nb_path)


c.FileContentsManager.post_save_hook = post_save

DEFAULT_JUPYTER_PASSWORD_SHA256 = '8bd001ab84cb8c74a23ca56471f830222b74afa08423487d9ddd1eba0f695ae7'
jupyter_password = os.getenv('JUPYTER_PASSWD')
if not jupyter_password:
    raise RuntimeError('JUPYTER_PASSWD must be set to start Jupyter Lab.')
if hashlib.sha256(jupyter_password.encode('utf-8')).hexdigest() == DEFAULT_JUPYTER_PASSWORD_SHA256:
    raise RuntimeError('JUPYTER_PASSWD must not use the bundled default password hash.')
c.ServerApp.password = jupyter_password

c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = int(os.getenv('JUPYTER_PORT', 8888))
if os.getenv('ENABLE_HTTPS', 'true').lower() == 'true':
    c.ServerApp.certfile = os.getenv('API_SSL_CERT', '/dltk/.jupyter/dltk.pem')
    c.ServerApp.keyfile = os.getenv('API_SSL_KEY', '/dltk/.jupyter/dltk.key')

# Fix for wigets limit default = 1000000
c.ServerApp.iopub_data_rate_limit = 20000000

# try fix async errors / UI issues
#c.ServerApp.kernel_manager_class = 'notebook.services.kernels.kernelmanager.AsyncMappingKernelManager'
c.ServerApp.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.AsyncMappingKernelManager'
