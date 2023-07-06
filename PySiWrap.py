import subprocess
from os import makedirs
from os.path import exists,isfile

import io,glob,warnings
import pandas as pd
import numpy as np
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:
    from osgeo import gdal
except:
    print("Gdal not found, please install it to use the transph_grd() function !")

pathToSirane = "SIRANE/"
fatalErrorStr = "ERREUR FATALE"

def _get_inputKeys(input_key_file):
    """
    A function to get the input Keys
    """
    with io.open(f"{input_key_file}",'r',encoding='utf8') as f:
        data = f.read()
    data = pd.Series(data.split(sep="\n"))
    def sepdf(str_val):
        str_val = str_val.split(";")
        if len(str_val) != 7:
            None #return ["","","","","","",""]
        else :
            return str_val
    strip = lambda x : x.strip()
    sprip_v = np.vectorize(strip) #to remove spaces in front and at the back
    data = sprip_v(np.array(data.apply(sepdf).dropna().to_list()))
    data = pd.DataFrame(data[1:,1:], index = data[1:,0], columns = data[0,1:])
    data["Key"] = data.index
    return data

def read_dat(filePath, Nindex = 0, Ncol = 0, modInput = None, modResult = None):
    """
    Reads SIRANE input files (.dat files), supporting temporal series and other types.
    Parameters:
        filePath (str): The path to the .dat file.
        Nindex (int): The index column number.
        Ncol (int): The column names number.
        modInput (Model, optional): The model used to make the filePath relative to the input file. Default is None.
        modResult (Model, optional): The model used to make the filePath relative to the result file. Default is None.
    Returns:
        pd.DataFrame: The data table.
    """
    if modInput is not None:
        filePath = f"{modInput.pathToSirane}/{modInput.inputs.FICH_DIR_INPUT}/{filePath}"
    if modResult is not None:
        filePath = f"{modResult.pathToSirane}/{modResult.inputs.FICH_DIR_RESUL}/{filePath}"

    data = pd.read_csv(filePath, sep='\t',index_col=Nindex,header=Ncol,na_values=["ND", "-9999", "-999"])
    if type(data.index[0]) is str:
        if "/" in data.index[0] and ":" in data.index[0]:
            data.index = pd.to_datetime(data.index, format="%d/%m/%Y %H:%M")
    return data

class Model(object):
    """
    Represents a SIRANE model.
    Parameters:
        name (str): The name of the model.
        pathToSirane (str, optional): The relative path to the folder containing SIRANE.exe. Default is {pathToSirane}.
        InputFile (str, optional): The donnees.dat file relative to the input folder. Default is "Donnees.dat".
    """
    def __init__(self, name, pathToSirane=pathToSirane, InputFile = "Donnees.dat"):
        super(Model, self).__init__()
        self._lst_changedFiles = []
        self._lst_cachedFiles=[]
        self.name = name
        self.pathToSirane = pathToSirane
        self.InputFile = InputFile
        self._inputs_full = self._read_input(self.InputFile)
        self.inputs = self._inputs_full.Value

        self._site_meteo_full = self._read_site(self.inputs.FICH_SITE_METEO)
        self.site_meteo = self._site_meteo_full.Value
        self._site_disp_full = self._read_site(self.inputs.FICH_SITE_DISP)
        self.site_disp = self._site_disp_full.Value

    def __repr__(self):
        return f"PySiWrap.Model named '{self.name}' at '{self.pathToSirane}'"

    def run(self, method= "Windows"):
        """
        Launches SIRANE from a Python file.
        Parameters:
            method (str): The method to launch Sirane (case sensitive). Can be one of the following:
                - 'Windows': Opens another terminal with Sirane.
                - 'Windows_Silent': Does not open a terminal (usually slightly faster).
                - 'wine': Runs on Linux-based systems with Wine.
        Returns:
            bool: True if the model has finished running; False if an error occurred due to inputs.
        """

        #Safeguard if the inputs have not been well set
        try:
            self._inputs_full.Value == self.inputs
            self.site_meteo == self._site_meteo_full.Value
            self.site_disp == self._site_disp_full.Value
        except :
            reflist = pd.concat([self._inputs_full, self._site_meteo_full, self._site_disp_full])
            modlist = pd.concat([self.inputs, self.site_meteo, self.site_disp])
            isnotinlist = np.vectorize(lambda x : not x in reflist.index)
            raise ValueError(f"You cannot add an imput that as not been set in the default parameter ! Those are not present: {modlist.index[isnotinlist(modlist.index)].to_list()}")
            return False
        if method not in ['Windows','Windows_Silent','wine']:
            list_of_methodes= "','".join(['Windows','Windows_Silent','wine'])
            raise ValueError(f"{method} is not a valid running method. Use one of the following : '{list_of_methodes}'")

        self._write_all_cachedFiles()
        self._write_input(self._inputs_full, self.InputFile)
        self._write_site(self._site_meteo_full, self.inputs.FICH_SITE_METEO)
        self._write_site(self._site_disp_full, self.inputs.FICH_SITE_DISP)

        self.log_filePath = f"{self.inputs.FICH_DIR_RESUL}/log.txt"
        self.pathToInput = f"{self.inputs.FICH_DIR_INPUT}/{self.InputFile}"
        makedirs(f"{self.pathToSirane}/{self.inputs.FICH_DIR_RESUL}", exist_ok=True)
        #make a copy of all the input file that have changed in the result file
        self._export_changedFiles(f"{self.pathToSirane}/{self.inputs.FICH_DIR_RESUL}/INPUT/")
        if method=="Windows": #run in an other terminal
            a = subprocess.run(["start", f"SIRANE - {self.name}","/wait", "cmd", f"/k cd {self.pathToSirane} & sirane {self.pathToInput} {self.log_filePath} & exit /b"], shell = True, capture_output=True)
        elif method=="wine":
            os.chdir(self.pathToSirane)
            a = subprocess.run(["wine", "sirane.exe", self.pathToInput, self.log_filePath], capture_output=True)
            os.chdir("..")
        elif method== "Windows_Silent":
            a = subprocess.run(["cd",self.pathToSirane, "&", "sirane", self.pathToInput, self.log_filePath], shell = True, capture_output=True)
        else:
            return False
        #back to default parameter on all the input files
        self._del_changedFiles()
        #See if a fatal error occured
        if fatalErrorStr in self.get_log(printlogs=False):
            warnings.warn("".join(self.get_log(printlogs=False).split("#################")[-2:]), SiraneError)
        return not fatalErrorStr in self.get_log(printlogs=False)

    def get_log(self, printlogs=True):
        with io.open(f"{self.pathToSirane}/{self.log_filePath}",'r',encoding='utf8') as f:
            logs = f.read()
        if printlogs==True:
            print(logs)
            return None
        else:
            return logs

    """
    This section of code is responsible for handling operations related to the input file of Sirane.
    """
    def _read_input(self, input_file):
        """
        Reads data from an input file in Python.
        Parameters:
            input_file (str): The input file relative to the input folder.
        Returns:
            pd.Dataframe : The data from the input file as a pandas DataFrame.
        """
        with io.open(f"{self.pathToSirane}/INPUT/{input_file}",'r',encoding='utf8') as f:
            data = f.read()
        data = pd.Series(data.split(sep="\n"))
        def sepdf(str_val):
            if "=" in str_val:
                return str_val.split("=")
            else :
                return [str_val,""]
            return 'oups should not be there'
        strip = lambda x : x.strip()
        sprip_v = np.vectorize(strip) #to remove spaces in front and at the back
        data = sprip_v(np.array(data.apply(sepdf).to_list()))
        data = pd.DataFrame(data, columns=["Name","Value"])
        # add the keyyysss
        keys = _get_inputKeys("DefaultParam/Donnees_def.dat")
        data = data.merge(keys, how="right", right_on="Description", left_on="Name").reindex(columns=["Description","Key","Value"])
        data.index = data.Key
        del data["Key"]
        return data

    def _write_input(self, data, input_dat, keepDef=True):
        """
        Writes data to the input file.
        Parameters:
            data (pd.DataFrame of str): The data to be written in the file.
            input_dat (str): The input file path relative to the input folder.
            keepDef (bool, optional): Whether to keep the default data in the file. Default is True.
        Returns:
            None
        """
        if keepDef==True:
            if input_dat in self._get_changedFiles():
                None
            else :
                self._add_changedFiles(input_dat)
        data.dropna().to_csv(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{input_dat}",sep="=", header = False, index=False)
        return None

    """
    This section of code is responsible for handling operations related to reading and writing site-specific files, including dispersion and meteorology data.
    """
    def _read_site(self,site_file):
        """
        Reads a site file in Python.
        Parameters:
            site_file (str): The site file path relative to the input folder.
        Returns:
            pd.DataFrame : The data from the site file as a pandas DataFrame
        """
        with io.open(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{site_file}",'r',encoding='utf8') as f:
            data = f.read()
        data = pd.Series(data.split(sep="\n"))
        def sepdf(str_val):
            if "=" in str_val:
                return str_val.split("=")
            else :
                return [str_val,""]
            return 'oups should not be there'
        strip = lambda x : x.strip()
        sprip_v = np.vectorize(strip) #to remove spaces in front and at the back
        data = sprip_v(np.array(data.apply(sepdf).to_list()))
        data = pd.DataFrame(data, columns=["Name","Value"])
        # add the keyyysss
        keys = _get_inputKeys("DefaultParam/Site_def.dat")
        data = keys.merge(data, how="left", left_on="Description", right_on="Name").reindex(columns=["Description","Key","Value"])
        data.index = data.Key
        del data["Key"]
        #data.Value = data.Value.astype(float)
        return data

    def _write_site(self, data, site_file, keepDef=True):
        """
        Writes data to a site file.
        Parameters:
            data (df): The data to be written in the file.
            site_file (str): The site file path relative to the input folder.
            keepDef (bool, optional): Whether to keep the default data in the file. Default is True.
        Returns:
            None
        """
        if keepDef==True:
            if site_file in self._get_changedFiles():
                None
            else :
                self._add_changedFiles(site_file)
        data.dropna().to_csv(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{site_file}",sep="=", header = False, index=False)
        return None

    """
        This part alow to copy a data file to a temporary file and therfore keep a default value of the data
    """
    def _add_changedFiles(self, PathToFile):
        """
        Creates safely modifiable copies of the files specified in `PathToFile`.
        Parameters:
            PathToFile (str or list of str): The path(s) to the file(s) to be copied.
        Returns:
            None
        """
        def add(PathToFile):
            PathToFile = str(PathToFile)
            if PathToFile in self._lst_changedFiles:
                raise Exception(f"{PathToFile} is already in a modfifiable way")
            elif exists(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{PathToFile}-def")  :
                raise Exception(f"{PathToFile}-def already exist but {PathToFile} is not in a modfifiable way. Use 'PySiWrap.force_all_changedFiles_to_def(self)' to change to default file")
            else :
                shutil.copy(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{PathToFile}", f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{PathToFile}-def")
                self._lst_changedFiles.append(PathToFile)
        if type(PathToFile) is list:
            for path in PathToFile:
                add(path)
        else :
            add(PathToFile)

    def _del_changedFiles(self, PathToFile=None, force_to_def = False):
        """
        Reverts safely modifiable files to their default state and makes them not safely modifiable.
        Parameters:
            PathToFile (str or None, optional): The path(s) to the file(s) or folder(s) to be reverted. If None, all files are changed. Default is None.
            force_to_def (bool, optional): Whether to change the file to the default state even if it is not registered as safely modifiable. Default is False.
        Returns:
            None
        """
        def rem(path):
            path = str(path)
            if path in self._lst_changedFiles:
                #replace file
                shutil.move(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{path}-def",f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{path}")
                self._lst_changedFiles.remove(path)
                return None
            elif force_to_def :
                shutil.move(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{path}-def",f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{path}")
                return None
            else:
                raise Exception(f"{path} is not in a modfifiable way")
        if PathToFile is None:
            PathToFile = self._get_changedFiles()
        if type(PathToFile) is list:
            for path in PathToFile:
                rem(path)
        else :
            rem(PathToFile)

    def _get_changedFiles(self):
        """
        Retrieves the paths of all safely modifiable files that have been changed.
        Returns:
            list: A copy of the list containing the paths to the safely modifiable files that have been changed.
        """
        return self._lst_changedFiles.copy()

    def get_cachedFiles(self, data=False):
            """
            Retrieves the paths or data of safely modifiable files that will be changed.
            Parameters:
                data (bool, optional): If True, returns all the safely modifiable files instead of just their paths. Default is False.
            Returns:
                If data=True:
                    List of [data, filePath, sep, index]
                Else:
                    List of filePath
            """
        if data:
            return self._lst_cachedFiles.copy()
        else:
            return [a[1] for a in self._lst_cachedFiles].copy()

    def _export_changedFiles(self, pathToExport, files = None):
        """
        Exports the safely modifiable files to the specified `pathToExport`.
        Parameters:
            pathToExport (str): The file path to export the safely modifiable files. If the file exists, it will remove everything from it.
            files (list or None, optional): The list of specific files to export. If None, exports all safely modifiable files. Default is None.
        Returns:
            None
        """
        if files is None:
            files = self._get_changedFiles()
        #rmv all the files in the path to export
        shutil.rmtree(pathToExport, ignore_errors=True)
        def exp(PathToFile):
            dir = "".join([str(i)+"/" for i in f"{pathToExport}/{PathToFile}".split("/")[:-1]])
            if dir != "/" :
                makedirs(dir, exist_ok=True)
            shutil.copy(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{PathToFile}", f"{pathToExport}/{PathToFile}")

        if type(files) is list:
            for path in files:
                exp(path)
        else :
            exp(files)

    def add_cachedFiles(self, data, filePath, sep='\t', index=True, na_rep=-9999):
        """
        Saves the modified inputs without writing them directly.
        Parameters:
            data (df): The data to be saved.
            filePath (str): The path to the .dat file relative to the input file.
            sep (str, optional): The separator used in the file. Default is '\t'.
            index (bool, optional): If True, the index will be included in the file. Default is True.
            na_rep (str, int, ..., optional): The value used to replace NaN values in the file (e.g., "NULL", -9999, ...). Default is -9999.
        Returns:
            None
        """
        if filePath in [a[1] for a in self._lst_cachedFiles]:
            self.delete_cachedFiles(filePath)
        self._lst_cachedFiles.append([data, filePath, sep, index, na_rep])
        return None

    def delete_cachedFiles(self, filePath):
        """
        Deletes the modified inputs and reverts them to their original state.
        Parameters:
            filePath (str): The path to the .dat file relative to the input file.
        Returns:
            None
        """
        self._lst_cachedFiles = [i for i in self._lst_cachedFiles if i[1] != filePath]
        return None

    def _write_all_cachedFiles(self):
        """
        Writes all modified inputs to their respective files.
        Parameters:
            None
        Returns:
            None
        """
        for element in self._lst_cachedFiles:
            self._write_cachedFiles(data = element[0],filePath= element[1], sep= element[2], index= element[3],na_rep= element[4])

    def _write_cachedFiles(self,data,filePath, sep='\t', index=True,na_rep="",keepDef=True):
        """
        Writes time series (or any SIRANE input files) data to a .dat file.
        Parameters:
            data (df): The data to be saved.
            filePath (str): The path to the .dat file relative to the input file.
            sep (str, optional): The separator used in the file. Default is '\t'.
            index (bool, optional): If True, the index will be included in the file. Default is True.
            na_rep (str, optional): The value used to replace NaN values in the file. Default is an empty string.
            keepDef (bool, optional): If True, the data will be written in a safe way, preserving a default value as '{filePath}-def'. Default is True.
        Returns:
            None
        """
        if keepDef==True:
            if filePath in self._get_changedFiles():
                None
            else :
                self._add_changedFiles(filePath)
        data.to_csv(f"{self.pathToSirane}/{self.inputs.FICH_DIR_INPUT}/{filePath}",sep=sep,index=index, date_format='%d/%m/%Y %H:%M', na_rep=na_rep)


class Output(object):
    """
    A class that handles air quality model output for multiple measurement stations.
    Parameters:
        Model: An instance of the PySiWrap model class.
        AddHourOfDay (bool, optional): Adds hour of the day, day of the month, and month as variables. Default is False.
        Relative_Residual (bool, optional): Calculates residuals relative to observations: (Obs-Mod)/(Obs+Mod)*2. Default is True.
    Attributes:
        mod: An instance of an air quality model class.
        alldata: A DataFrame containing reception data for all measurement stations.
        list_of_Recept: A list of measurement station names with reception data.
        list_of_species: A list of species names for which data is available in alldata.
        list_of_VarMeteo: A list of meteorological variable names.
    Methods:
        scatterplots: Displays scatter plots for specified species and receptors.
        qQplots: Displays quantile-quantile plots for each combination of species and receptors.
        residualplots: Displays residual box plots for different meteorological variables for each species.
        indicators: Computes 'FB', 'MG', 'NMSE', 'VG', 'R', and 'FAC2' indicators for data for different receptors and species.
    """
    def __init__(self, Model, AddHourOfDay = False, Relative_Residual=True):
        """
        Initialise une instance de la classe Output.
        """
        super(Output, self).__init__()
        self.mod = Model
        if self.mod.inputs.FICH_RECEPT in self.mod.get_cachedFiles():
            ls = list([i[0] for i in self.mod.get_cachedFiles(data=True) if i[1]==self.mod.inputs.FICH_RECEPT][0].index)
        else:
            ls = list(read_dat(self.mod.inputs.FICH_RECEPT, modInput=self.mod).index)
        self.list_of_Recept = [i for i in ls if isfile(f"{self.mod.pathToSirane}/{self.mod.inputs.FICH_DIR_RESUL}/RECEPT/Recept_{i}.dat")]

        data = [read_dat(f"RECEPT/Recept_{i}.dat", modResult=self.mod) for i in self.list_of_Recept]
        self.alldata = pd.concat(data, keys=self.list_of_Recept).reset_index(level=0)
        self.alldata = self.alldata.rename(columns = {'level_0':'Recept'})
        self.list_of_species=[sp for sp in set([i.split("_")[0] for i in self.alldata.columns if (i not in ["Recept", "Date"])]) if len(self.alldata.loc[:,[f"{sp}_Mod", f"{sp}_Mes"]].dropna()) != 0]
        self.list_of_species= np.sort(self.list_of_species)
        for species in self.list_of_species:
            if Relative_Residual:
                self.alldata[f"{species}_Res"] = (self.alldata[f"{species}_Mod"] / self.alldata[f"{species}_Mes"]) #/ (self.alldata[f"{species}_Mes"])
            else:
                self.alldata[f"{species}_Res"] = self.alldata[f"{species}_Mod"] - self.alldata[f"{species}_Mes"]
        self.list_of_VarMeteo = None

        self.AddHourOfDay = AddHourOfDay
        if AddHourOfDay:
            self.alldata["Heure"] = self.alldata.index.hour
            self.alldata["Jour_semaine"] = self.alldata.index.dayofweek
            self.alldata["Jour_mois"] = self.alldata.index.day
            self.alldata["Mois"] = self.alldata.index.month

        meteo = read_dat(self.mod.inputs.FICH_METEO,modInput=self.mod)
        meteo = meteo.loc[self.alldata.index,:]
        meteo2 = read_dat("METEO/Resul_Meteo.dat",modResult=self.mod).loc[self.alldata.index,["Hcla","Ustar","SigmaTheta","Cld","H0","Lmo","Thetastar","k1","k3"]]
        meteo = pd.concat([meteo, meteo2], axis=1)
        self.list_of_VarMeteo = list(meteo.columns)
        if self.AddHourOfDay: #ajout de l'heure comme variable "météo"
            self.list_of_VarMeteo += ["Heure","Jour_semaine","Jour_mois","Mois"]
        self.alldata = pd.concat([self.alldata, meteo], axis=1)

    def __repr__(self):
        return f"PySiWrap.Output for model named '{self.mod.name}'"

    def scatterplots(self, list_of_Recept=None, list_of_species=None):
        """
        Displays scatter plots for the specified species and receptors.
        Parameters:
            list_of_Recept (list, optional): A list of receptor identifiers for which scatter plots should be displayed. If not provided, all receptors will be included.
            list_of_species (list, optional): A list of species for which scatter plots should be displayed. If not provided, all species will be included.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the scatter plots.
            axs (numpy.ndarray): Array of sub-plot axes for each species.
        """
        if list_of_Recept is None:
            list_of_Recept = self.list_of_Recept
        if list_of_species is None:
            list_of_species = self.list_of_species
        if len(list_of_Recept) <= 10:
            sns.set_palette(sns.color_palette("tab10"), len(list_of_Recept))
        else:
            sns.set_palette(sns.color_palette("tab20"), len(list_of_Recept))
        if len(list_of_species)<=0 :
            return (None,None)
        fig, axs = plt.subplots(1,len(list_of_species))
        if len(list_of_Recept) > 1:
            legend = True
        else:
            legend = False

        filter_recept = self.alldata.Recept.apply(lambda x : x in list_of_Recept)
        for id,species in enumerate(list_of_species):
            min = self.alldata.loc[filter_recept,[f"{species}_Mod", f"{species}_Mes"]].min().min()
            max = self.alldata.loc[filter_recept,[f"{species}_Mod", f"{species}_Mes"]].max().max()
            min = min - (max-min)/20
            max = max + (max-min)/20
            axs[id].plot([min,max],[min,max], ls="--", c=".5")
            axs[id] = sns.scatterplot(data= self.alldata[filter_recept], y=f"{species}_Mod", x=f"{species}_Mes", hue="Recept", ax=axs[id], legend=legend, s=10)
            axs[id].set(xlim=(min, max), ylim=(min, max))
            axs[id].set_aspect("equal")
            if (id == 0) and (len(list_of_Recept) > 1):
                legend = False
                axs[id].legend(title="Recepteurs")
        return fig, axs

    def qQplots(self, list_of_Recept=None, list_of_species=None):
        """
        Displays quantile-quantile plots for each combination of species and receptors.
        Parameters:
            list_of_Recept (list of str, optional): List of receptors to include in the plot. If None, all receptors will be displayed.
            list_of_species (list of str, optional): List of species names to display. If None, all species will be displayed.
        Returns:
            fig (matplotlib.figure.Figure): The Figure object containing the plot.
            axs (numpy.ndarray): An array of AxesSubplot objects representing the subplots in the figure.
        """
        if list_of_Recept is None:
            list_of_Recept = self.list_of_Recept
        if list_of_species is None:
            list_of_species = self.list_of_species
        quantile = lambda series: np.quantile(series.dropna(), np.arange(0,1,1/len(series)))# - series.min())/(series.max()-series.min())
        if len(list_of_Recept)*len(list_of_species) == 0:
            return (None,None)
        fig, axs = plt.subplots(len(list_of_Recept),len(list_of_species))
        if type(axs) != type(np.ndarray([])):
            d = np.ndarray(shape=(2,2),dtype=object )
            d[0,0] = axs
            axs = d
        elif len(axs.shape) == 1:
            if len(list_of_Recept) == 1 :
                d = np.ndarray(shape=(2,axs.shape[0]),dtype=object)
                d[0,:] = axs
                axs = d
            else:
                d = np.ndarray(shape=(axs.shape[0],2),dtype=object)
                d[:,0] = axs
                axs = d

        for n_el, species in enumerate(list_of_species):
            for n_rec, recept in enumerate(list_of_Recept):
                if len(self.alldata.loc[self.alldata["Recept"] == recept,[f"{species}_Mes",f"{species}_Mod"]].dropna()) == 0:
                    fig.delaxes(axs[n_rec, n_el])
                else:
                    observed_data = quantile(self.alldata.loc[self.alldata["Recept"] == recept,f"{species}_Mes"])
                    predicted_data = quantile(self.alldata.loc[self.alldata["Recept"] == recept,f"{species}_Mod"])
                    min = np.array((observed_data.min(),predicted_data.min())).min()
                    max = np.array((observed_data.max(),predicted_data.max())).max()
                    min = min - (max-min)/20
                    max = max + (max-min)/20
                    axs[n_rec, n_el].scatter(observed_data, predicted_data, color='blue')
                    axs[n_rec, n_el].plot([min,max],[min,max], ls="--", c=".5")
                    axs[n_rec, n_el].set(xlim=(min, max), ylim=(min, max), ylabel= 'Predicted Concentration', xlabel='Observed Concentration', title=f"{recept}: {species}")
                    axs[n_rec, n_el].set_aspect("equal")

        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #fig.set_size_inches(10, 10)
        fig.subplots_adjust(hspace=0)
        return fig, axs

    def residualplots(self, list_of_VarMeteo = None, list_of_species=None, list_of_Recept=None, max_bins=35):
        """
        Displays residual box plots for each species based on different meteorological variables.
        Parameters:
            list_of_VarMeteo (list of str, optional): List of meteorological variable names to display. If None, all meteorological variables will be displayed.
            list_of_species (list of str, optional): List of species names to display. If None, all species will be displayed.
            list_of_Recept (list of str, optional): List of receptor names to consider. If None, all receptors will be considered.
            max_bins (int, optional): Maximum number of bins per plot.
        Returns:
            fig (matplotlib.figure.Figure): The matplotlib Figure object containing the residual plots.
            axs (numpy.ndarray): The numpy array containing the axes for each residual plot.
        """
        if list_of_VarMeteo is None:
            list_of_VarMeteo = self.list_of_VarMeteo
        if list_of_species is None:
            list_of_species = self.list_of_species
        if list_of_Recept is None:
            list_of_Recept = self.list_of_Recept

        filter_recept = self.alldata.Recept.apply(lambda x : x in list_of_Recept)
        fig, axs = plt.subplots(len(list_of_VarMeteo),len(list_of_species))
        fig.set_size_inches(12.4 * len(list_of_species) /2 ,17.5 * len(list_of_VarMeteo) / 5)
        if type(axs) != type(np.ndarray([])):
            d = np.ndarray(shape=(2,2),dtype=object )
            d[0,0] = axs
            axs = d
        elif len(axs.shape) == 1:
            if len(list_of_VarMeteo) == 1 :
                d = np.ndarray(shape=(2,axs.shape[0]),dtype=object)
                d[0,:] = axs
                axs = d
            else:
                d = np.ndarray(shape=(axs.shape[0],2),dtype=object)
                d[:,0] = axs
                axs = d
        def nbr_bins(data):
            nbr_diff = len(set(data))
            if (nbr_diff>0)&(nbr_diff < max_bins):
                return nbr_diff
            else:
                return int(max_bins)
        #sns.set_palette(sns.color_palette("tab20"))
        for n_var, variable in enumerate(list_of_VarMeteo):
            nbr_xvalue = len(set(self.alldata[filter_recept][variable]))
            if (nbr_xvalue>0)&(nbr_xvalue <= max_bins):
                self.alldata[f"{variable}_Group"] = self.alldata[filter_recept][variable]
                retbins = np.sort(np.array(list(set(self.alldata[filter_recept][variable]))))
                xlables = mticker.AutoLocator().tick_values(retbins.min(),retbins.max())
                xticks  = (xlables-retbins[1])/(retbins[-1]-retbins[1])*len(retbins[2:])+1
            else:
                self.alldata[f"{variable}_Group"],retbins = pd.cut(self.alldata[filter_recept][variable], nbr_bins(self.alldata[filter_recept][variable]), retbins=True)
                xlables = mticker.AutoLocator().tick_values(retbins.min(),retbins.max())
                xticks = (xlables-retbins[1])/(retbins[-1]-retbins[1])*len(retbins[2:])+0.5
            for n_el, species in enumerate(list_of_species):
                if len(self.alldata.loc[filter_recept,[f"{species}_Res"]].dropna()) == 0:
                    fig.delaxes(axs[n_rec, n_el])
                else:
                    axs[n_var, n_el] = sns.boxplot(data = self.alldata[filter_recept], x=f"{variable}_Group", y=f"{species}_Res", orient="v", ax = axs[n_var, n_el], color = "palegreen")
                    axs[n_var, n_el].set_xticks(xticks)
                    axs[n_var, n_el].set_xticklabels(xlables)
                    axs[n_var, n_el].set_xlabel(f"{variable}")
        return fig,axs

    def indicateurs(self, list_of_Recept=None, list_of_species=None, yearlyMQI=False):
        """
        Computes the 'FB', 'MG', 'NMSE', 'VG', 'R', and 'FAC2' indicators for data for different receptors and species.
        Parameters:
            list_of_Recept (list, optional): The list of receptors for which to compute the indicators. If None, computes indicators for all receptors.
            list_of_species (list, optional): The list of species for which to compute the indicators. If None, computes indicators for all species.
            yearlyMQI (bool, optional): If True, computes the annual MQI (Model Quality Index) for each species and indicator.
        Returns:
            dict: A dictionary containing, for each species, a DataFrame with the indicators for each receptor.
        """
        if list_of_Recept is None:
            list_of_Recept = self.list_of_Recept
        if list_of_species is None:
            list_of_species = self.list_of_species
        output = {}
        for esp in list_of_species:
            output[esp] = pd.DataFrame([calculate_indicators(self.alldata[self.alldata.Recept == rec], f"{esp}_Mod", f"{esp}_Mes", esp, yearlyMQI=yearlyMQI) for rec in list_of_Recept], index = list_of_Recept)
        return output

def transph_grd(filePath_grd, filePath_tif = None, crs="EPSG:31370", modInput = None, modResult=None):
    """
    Reads and transforms a .grd file into a .tif file.
    Parameters:
        filePath_grd (str): The path to the grd file (cannot be rotated).
        filePath_tif (str, optional): The path where the tif file should be written. If set to None, no file conversion will be performed.
        crs (str, optional): The coordinate reference system (CRS) of the projection. Used only when transforming to a tiff. The default value is Belgian Lambert 72.
        modInput: The model for which the filePath_grd is relative to the input file. Default is None.
        modResult: The model for which the filePath_grd is relative to the result file. Default is None.
    Returns:
        tuple: A tuple containing the grid data as np.array, the extend as a list, and the GDAL object of the grid file.
    """
    if modInput is not None:
        filePath_grd = f"{modInput.pathToSirane}/{modInput.inputs.FICH_DIR_INPUT}/{filePath_grd}"
    if modResult is not None:
        filePath_grd = f"{modResult.pathToSirane}/{modResult.inputs.FICH_DIR_RESUL}/{filePath_grd}"

    t = gdal.Open(filePath_grd)
    value = t.ReadAsArray()
    GT = t.GetGeoTransform()
    extent = [GT[0], GT[0] + value.shape[1] * GT[1], GT[3] + value.shape[0] * GT[5],  GT[3]]
    if filePath_tif != None:
        t.SetProjection(crs) #To set the Projection
        t = gdal.Translate(filePath_tif, t)
    return (value, extent, t)

def force_all_changedFiles_to_def(model):
    """
    Restores all modified files in another instance (ending with "-def") to their default values.
    Parameters:
        model (PySiWrap.Model): The model instance that contains files with "-def" in the INPUTS.
    Returns:
        list: A list of all files that have been restored to their default values.
    """
    Paths = [i[:-4].split("INPUT\\")[-1] for i in glob.glob(f"{model.pathToSirane}\\{model.inputs.FICH_DIR_INPUT}/**/*-def", recursive=True)]
    [model._del_changedFiles(PathToFile=i, force_to_def = True) for i in Paths]
    return Paths

#indcateurs stats:
def calculate_indicators(df, C_p, C_o, esp, LOQ = None, yearlyMQI = False):
    """
    Calculates the 'FB', 'MG', 'NMSE', 'VG', 'R', 'FAC2', 'NAD', and MQI indicators for data pairs.
    Parameters:
        df (pd.DataFrame): A DataFrame with observed and predicted data pairs. The data pairs are on the same row.
        C_p (str): The column name for the predicted data by the model.
        C_o (str): The column name for the observed data.
        esp (str): The name of the species concerned (can be [NO2, O3, PM10, PM2.5]; if other, MQI is np.nan).
        LOQ (float, optional): The limit of quantification of the detectors. If None, NAD_t is np.nan.
        yearlyMQI (bool, optional): If True, the annual MQI (Model Quality Indicators) is also provided.
    Returns:
        dict: A dictionary containing the values of the indicators. For MQI, if more than 25% of the data pairs are missing, the values are negative (and cannot be used according to FAIRMOD initative standard).
    """
    df = df[[C_p,C_o]].copy()
    df.loc[df.isna().sum(axis=1)>0,:] = np.nan
    #df = df.dropna()
    # calculer les moyennes et écart-types
    C_p_mean = df[C_p].mean()
    C_o_mean = df[C_o].mean()
    C_p_std = df[C_p].std()
    C_o_std = df[C_o].std()
    # calculer FB
    FB = (C_o_mean - C_p_mean) / (0.5 * (C_o_mean + C_p_mean))
    # calculer MG
    MG = np.exp(np.log(df[C_o]).mean() - np.log(df[C_p]).mean())
    # calculer NMSE
    NMSE = ((df[C_o] - df[C_p])**2).mean() / (C_o_mean * C_p_mean)
    # calculer VG
    VG = np.exp(((np.log(df[C_o]) - np.log(df[C_p]))**2).mean())
    # calculer R
    R = (((df[C_o] - C_o_mean) * (df[C_p] - C_p_mean)).mean()) / (C_o_std * C_p_std)
    # calculer FAC2
    FAC2 = ((df[C_p] / df[C_o] >= 0.5) & (df[C_p] / df[C_o] <= 2.0)).mean()
    #Calculer NAD
    NAD = (df[C_o] - df[C_p]).abs().mean()/(C_p_mean + C_o_mean)

    #Calculer le MQI
    def U(Oi, esp):
        """
        Calculates the measurement uncertainty.
        Parameters:
            Oi (float): The observed measurement value.
            esp (str): The name of the species for which the uncertainty is calculated.
        Returns:
            float: The measurement uncertainty.
        """
        params = pd.DataFrame({"NO2" : {"Ur" : 0.24, "RV": 200, "a" : 0.2, "Np":5.2, "Nnp":5.5},
                               "O3"  : {"Ur" : 0.18, "RV": 120, "a" : 0.79, "Np":11, "Nnp":3},
                               "PM10": {"Ur" : 0.28, "RV": 50, "a" : 0.25, "Np":20, "Nnp":1.5},
                               "PM2.5":{"Ur" : 0.36, "RV": 25, "a" : 0.5, "Np":20, "Nnp":1.5}})
        return params[esp]["Ur"] * ((1-params[esp]["a"]**2) * Oi**2 + params[esp]["a"]**2 * params[esp]["RV"]**2)**0.5

    if esp in ["NO2", "PM10", "PM2.5"]:
        MQI = ((df[C_o] - df[C_p])**2).sum()**0.5 / 2 / (U(df[C_o], esp)**2).sum()**0.5
        if (df[[C_o,C_p]].isna().sum(axis=1)>=1).sum()/len(df) > 0.25:
            MQI = - MQI
    elif esp == "O3":
        #maximum journalier des moyennes glissantes sur huit heures
        dfO3 = df[[C_o,C_p]].rolling(8, min_periods=6).mean()
        dfO3["date"] = dfO3.index.date
        dfO3_day = dfO3.groupby("date").max()
        dfO3_day.loc[dfO3.groupby("date").count().sum(axis=1)>=18,:] = np.nan
        MQI = (np.sum(dfO3_day[C_o] - dfO3_day[C_p])**2)**0.5 / 2 / (U(dfO3_day[C_o], esp)**2).sum()**0.5
        if dfO3[C_o].isna().sum()/len(dfO3) > 0.25:
            MQI = - MQI
    else :
        MQI = np.nan

    #Calculer le MQI Annuel
    def U_year(Oi,esp):
        """
        Calculates the measurement uncertainty.
        Parameters:
            Oi (float): The observed measurement value.
            esp (str): The name of the species for which the uncertainty is calculated.
        Returns:
            float: The measurement uncertainty.
        """
        params = pd.DataFrame({"NO2" : {"Ur" : 0.24, "RV": 200, "a" : 0.2, "Np":5.2, "Nnp":5.5},
                               "O3"  : {"Ur" : 0.18, "RV": 120, "a" : 0.79, "Np":11, "Nnp":3},
                               "PM10": {"Ur" : 0.28, "RV": 50, "a" : 0.25, "Np":20, "Nnp":1.5},
                               "PM2.5":{"Ur" : 0.36, "RV": 25, "a" : 0.5, "Np":20, "Nnp":1.5}})
        return params[esp]["Ur"] * ((1-params[esp]["a"]**2) * Oi**2 / params[esp]["Np"] + params[esp]["a"]**2 * params[esp]["RV"]**2 /params[esp]["Nnp"])**0.5
    if esp in ["NO2", "O3", "PM10", "PM2.5"] and yearlyMQI:
        MQI_y = np.abs(C_p_mean - C_o_mean)/2/U_year(C_o_mean,esp)
        if (df[[C_o,C_p]].isna().sum(axis=1)>=1).sum()/len(df) > 0.25:
            MQI_y = - MQI_y
    else :
        MQI_y = np.nan

    return {'FB': FB, 'MG': MG, 'NMSE': NMSE, 'VG': VG, 'R': R, 'FAC2': FAC2, "NAD" : NAD, "MQI": MQI, "MQI_y":MQI_y}


class SiraneError(UserWarning):
    """
    A custom error Warning class for SIRANE fatal error
    """
    pass
