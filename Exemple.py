import PySiWrap as ps
from importlib import reload as r
r(ps)
import copy
import matplotlib.pyplot as plt
import numpy as np

#A simple exemple whre we change the dates and which species is moddeled
mod = ps.Model("Exemple", pathToSirane="SIRANE")
print(f"the modeling is performed form the {mod.inputs.DATE_DEB} to {mod.inputs.DATE_FIN}")
#Change begening and end dates
mod.inputs.DATE_DEB = "01/01/2017 07:00:00"
mod.inputs.DATE_FIN = "01/01/2017 08:00:00"
print(f"the modeling is performed form the {mod.inputs.DATE_DEB} to {mod.inputs.DATE_FIN}")
#desactivate the modelisation of PM and PM2.5
especes = ps.read_dat(mod.inputs.FICH_ESPECES, modInput=mod)
especes.loc[["PM","PM25"],"Flag"] = 0
mod.get_cachedFiles(data=True)
mod.add_cachedFiles(especes, mod.inputs.FICH_ESPECES)
mod.get_cachedFiles(data=True)
mod.run("Windows")


###
#An exemple to see the effect of removing the O3 background concentration
###
#First, create the two models (default one and the one with no O3 background concentration)
mod_def = copy.deepcopy(mod) #you can copy and deepcopy any models, but a deep copy makes no interference between them while copy does
mod_def.name = "Default"
mod_def.inputs.FICH_DIR_RESUL = f"RESULT_{mod_def.name}"

mod_LowEmissions = copy.deepcopy(mod_def)
mod_LowEmissions.name = "LowEmissions"
mod_LowEmissions.inputs.FICH_DIR_RESUL = f"RESULT_{mod_LowEmissions.name}"
C_Background = ps.read_dat(mod_LowEmissions.inputs.FICH_POLL_FOND, modInput=mod_LowEmissions)
C_Background.loc[:,"O3"] = 0
mod_LowEmissions.add_cachedFiles(C_Background, mod_LowEmissions.inputs.FICH_POLL_FOND)

#onces they are created, you can run them
mod_def.run("Windows") #this one open a new terminal
mod_LowEmissions.run("Windows_Silent") #this one run SIRAN on the background

#plot the mean NO2 concentrations and the absolute difference
CNO2_def,extent,_ = ps.transph_grd("GRILLE_STAT/Conc_NO2_Moy.grd", modResult=mod_def)
CNO2_NoBackground,extent,_ = ps.transph_grd("GRILLE_STAT/Conc_NO2_Moy.grd", modResult=mod_LowEmissions)

fig, axs = plt.subplots(2,2,figsize=(7,5), layout="constrained")
fig.suptitle('Mean NO2 concentrations on 1st Jan 2017 7:00 to 8:00')
map = axs[0,0].imshow(CNO2_def,extent = extent)
plt.colorbar(map,ax=axs[0,0])
axs[0,0].set_title("Default")
map1 = axs[0,1].imshow(CNO2_NoBackground,extent = extent)
plt.colorbar(map1,ax=axs[0,1])
axs[0,1].set_title("No O3 background")
max_diff = np.max(np.abs(CNO2_NoBackground-CNO2_def))
map2 = axs[1,0].imshow(CNO2_NoBackground-CNO2_def,extent = extent, vmin=-max_diff,vmax=max_diff, cmap = "RdBu_r")
plt.colorbar(map2,ax=axs[1,0])
axs[1,0].set_title("No O3 background - default")
fig.delaxes(axs[1,1])
fig.savefig("ExempleFigs/ZeroO3background.png")
plt.show()
