# topic_models_using_rosetta

This repository is intended to serve as a tutorial for running topic models on a sample of 5 text documents using the Python packages rosetta. 

To follow the tutorial:

1. Clone or download this repository to your local machine.
2. Install the rosetta package (updated for Python 3) from the following forked branch https://github.com/chsuong/rosetta/tree/update. For example, use the following command on your terminal if you have pip installed: 
```
pip install git+https://github.com/chsuong/rosetta.git@update
```
3. Install python packages ```os```, ```panda```, and ```random```.
4. Install Vowpal Wabbit to be used via the terminal of your local machine. For example, install it using the following command on your terminal if you have brew installed:
```
brew install vowpal-wabbit
```
4. Open the IPython Notebook "topic_models_using_rosetta.ipynb" or the Python script "topic_models_using_rosetta.py" in Python 3. 
5. Change the file paths in the notebook and the script for your own use.

##Notes: 
- data/raw is the directory where the documents are stored as text files. 

- This tutorial is based on the tutorial at https://github.com/columbia-applied-data-science/rosetta.
- See https://github.com/columbia-applied-data-science/rosetta for more tutorials and information about the rosetta package. 

