import base64
import io 
import os

#todo
# path should incorporate model name, algo name and cell metadata?
# create a package with this and the other functions required
# summary command should have mode options?
# need to pass hostname from DLTK into container for absolute URI reference

def SplunkGenerateGraphicsObjects(model,key,plot,graphics_path="/srv/app/graphics/"):
    pic_IObytes = io.BytesIO()

    path=graphics_path+key+".png"

    if hasattr(plot,'fig'):
        plot.fig.savefig(pic_IObytes, format='png')
        plot.fig.savefig(path, format='png')
    elif hasattr(plot,'figure'):
        plot.figure.savefig(pic_IObytes, format='png')
        plot.figure.savefig(path, format='png')

    pic_IObytes.seek(0)
    pic_base64 = base64.b64encode(pic_IObytes.read())

    if "graphics" not in model:
        model["graphics"] = {}

    model["graphics"][key] = {}
    model["graphics"][key]["base64"]=pic_base64
    model["graphics"][key]["container_local_path"]=path
    model["graphics"][key]["external_relative_path"]="graphics/"+key+".png"
        
    return model