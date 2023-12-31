![Python](https://img.shields.io/badge/python-3.8-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

# PySiWrap : A python wrapper for SIRANE

PySiWrap is a Python wrapper for the SIRANE urban dispersion model. It aims to simplify and accelerate the modeling process of SIRANE by automating parameter changes and storing modifications alongside the SIRANE outputs. This wrapper restores the original parameters, allowing users to perform consecutive model runs with unmodified data (useful for sensitivity analysis).

In addition, PySiWrap provides automated graph generation, statistics and export to FAIRMOD Delta tool to facilitate result analysis.

## Features

- Automates parameter changes for the SIRANE chemical dispersion model
- Stores parameter modifications alongside SIRANE outputs
- Restores original parameters for consecutive model runs with unmodified data
- Provides automated graph generation for result analysis

## Requirements
- Have a working version of SIRANE on your computer (see Q&A 1). Tested with SIRANE 2.2 rev146 & rev155
- Python. Tested with python 3.10.6
- Python libraries (see requirements.txt)

## Installation
1. Clone the PySiWrap repository:
```
git clone https://github.com/Turtle6665/PySiWrap.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```
Note that osgeo/gdal is not mandatory. Its only used when calling the transph_grd() function. It's goal is to read and the grd files (grid files generated by SIRANE) and transform it into a tiff.

3. Setup the [DefaultParam files](DefaultParam/):
To fill up this file, simply copy and paste the lines after the line 16 from the SIRANE SETTINGS folder to these two folders. Only the first two column (Description and Keyword) are used by PySiWrap, the others are there only to simplify the setup work.

As new versions of SIRANE sometimes adds/removes/change the keywords in those files, make sure that the following one are present. These are directly used by PySiWrap and are therefore mandatory. If they are not present in the SETTINGS file, this version of SIRANE might not be compatible with PySiWrap.

List of required keywords in DefaultParam files:
* "FICH_DIR_INPUT": Path to the INPUT folder
* "FICH_DIR_RESUL": Path to the RESULT folder
* "FICH_SITE_METEO": Path of the file that provides of the characteristics of the meteorological site
* "FICH_SITE_DISP": Path of the file that provides of the characteristics of the dispersion site

## Usage
1. Import the PySiWrap module in your Python script:
```python
import PySiWrap as ps
```
2. Create a PySiWrap Model, change the parameters, run it and more !
Check the exemple.py file

## Model class principle


## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request to the [GitHub repository](https://github.com/Turtle6665/PySiWrap).

## Q&A :
1. Why can I not use PySiWrap without having access to SIRANE ?
I don't have the right to share the SIRANE program. You therefore need to have the SIRANE executable to be able to use this library. To gain access to SIRANE, go to the [SIRANE website](http://air.ec-lyon.fr/SIRANE/index.php).

2. Is PySiWrap linked to the SIRANE Development ?
No, PySiWrap is not linked or affiliated to the "Laboratoire de Mécanique des Fluides et d’Acoustique" (LMFA), the development team behind SIRANE. PySiWrap was originally constructed as a side-project during my master's thesis on air pollution in Brussels. I needed a way to more easily change parameters and run a high amount of a SIRANE simulation without the need to be behind my screen.

3. Should a path be relative or absolute ?
All path used within PySiWrap can be relative, but not all can be absolute. In fact, all the path to input files must be relative to the input folder.
When using PySiWrap I would recommend to only use relative path (maybe except for the "pathToSirane")

4. Wicth OS is PySiWrap working on ?
Currently, I only have access to a Windows version of SIRANE. It's therefore intended to work with this OS. For Linux, I tested it using [Wine](https://www.winehq.org/) and the windows SIRANE executable. As it worked, a parameter to the run function has been added to be able to use Wine.
If you have another version of SIRANE not yet taken into account, you can create a pull request/open an issue.


## License
This project is released under the MIT license. See [LICENSE](LICENSE) for more details.

## Acknowledgment
This script has originally been created within a master thesis supported financially by [Bloomberg Philanthropies](https://www.bloomber.org).
