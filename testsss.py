import numpy as np
import seaborn as sns
import matplotlib

#  go Settings -> Tools -> Python Scientific -> uncheck 'Show plots in tool window option'
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
np.random.seed(0)


bird= sns.load_dataset("penguins")
g = sns.FacetGrid(bird, col="island", hue="species")
g.map(plt.scatter, "flipper_length_mm", "body_mass_g", alpha=.6)
g.add_legend()
plt.show()