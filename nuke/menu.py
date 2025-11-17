print('Menu file loaded successsfully')
import importlib
import nuke

import matte2Roto 

def run_matte_tool():
    importlib.reload(matte2Roto)
    matte2Roto.run()


# add a button in nuke menu
nuke.menu("Nuke").addCommand(
    "StudioTools/Matte -> Shapes",
    "run_matte_tool()"
)