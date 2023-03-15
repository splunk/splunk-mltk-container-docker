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
    # py_path = "/srv/app/model/"

    # TODO test - change the py_path
    py_path = "/home/jovyan/app/model/"

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

#c.ServerApp.password = u'sha1:f7432152c71d:e8520c26b9d960e838d562768c1d24ef5b9b76c7'
# default PW from app or the provided PW hash from the user defined hashed password in the ENV var
c.ServerApp.password = os.getenv('JUPYTER_PASSWD','sha1:f7432152c71d:e8520c26b9d960e838d562768c1d24ef5b9b76c7')

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
