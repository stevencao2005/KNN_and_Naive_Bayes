import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd
#  go Settings -> Tools -> Python Scientific -> uncheck 'Show plots in tool window option'
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv
import sklearn as sk
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

np.random.seed(0)

seeds = np.genfromtxt(r"C:\\Xiwu\\S\\uci_intern\\pythonProject\\the_final_project\\seeds_dataset.txt", delimiter=None)
names = ['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel', 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Wheat Type']
df_seeds = pd.DataFrame(seeds, columns=names)

#------------------------------

sns.relplot(data=df_seeds, x='Area', y='Perimeter', hue="Wheat Type", kind="scatter")
plt.show()

#------------------------------
#area to perimeter relationship........
g= sns.FacetGrid(data=df_seeds, col="Wheat Type", hue="Wheat Type")
g.map(plt.scatter, "Area", "Perimeter", alpha=.6)
g.add_legend()
plt.show()


#-----------------------------
#area to wheat type in a good way to see relationships as long as it is not densely populated
sns.stripplot(x="Wheat Type", y="Area", hue= "Wheat Type", data=df_seeds, palette="Set1")
plt.show()


#----------------------------
#same as the previous one, just making the dots not as densely populated if it was.
sns.swarmplot(x="Wheat Type", y="Area", hue="Wheat Type",data=df_seeds, palette="Set1", dodge=True)
plt.show()




#---------------------------
#similar to the FacetGrid and map function except that it has a best line fit
sns.lmplot(x="Area", y="Perimeter", hue="Wheat Type",order=2, col="Wheat Type",markers = ['o','x','+'],palette="Set1",data=df_seeds)
plt.show()


#---------------------------
#visualise a scatter plot and histogram side
#by side to get a holistic view of the spread of data points and also the frequency of observation at the same time.
sns.set_palette("gist_rainbow_r")
sns.jointplot(x="Area", y="Perimeter", kind="hex",data=df_seeds )
plt.show()


#-----
#same thing but just also plotting the datapoints
g = sns.jointplot(x="Area", y="Perimeter", kind="hex",data=df_seeds )
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.set_axis_labels("Area (in gram)", "Perimeter ( in mm)")
plt.show()

# ---------------------------
# the diagonals show the distribution of the values. The higher a curve is, the more of the values that are at that value
sns.pairplot(df_seeds, hue="Wheat Type")
plt.show()
i=1


Y = seeds[:,-1]
X = seeds[:,0:-1]



X,Y = sk.utils.shuffle(X,Y) # shuffle data randomly
sns.set()
Xtr,Xva,Ytr,Yva = sk.model_selection.train_test_split(X,Y, test_size=0.2)

# Slicing the data points in 3 different label groups of 0, 1, 2
X_groups = X[Y==1], X[Y==2], X[Y==3]
fig, axes = plt.subplots(3,7,figsize=(10,5), constrained_layout=True)


cols = ['Feature {}'.format(col) for col in range(1, 8)]
rows = ['Class {}'.format(row) for row in range(1,4)]
colors = ['c', 'm', 'y']
for group_idx, Xi in enumerate(X_groups):
    for j,ax in enumerate(axes[group_idx]):
        ax.hist(Xi[:,j], color=colors[group_idx])
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation='vertical', size='large')

plt.suptitle("Histograms of each feature of 3 classes", fontsize=14)
plt.show()

print('Mean of each feature: %s' %np.mean(X, axis=0))
print('Variance of each feature: %s' %np.var(X, axis=0))
print('Standard deviation of each feature: %s' %np.std(X, axis=0))


fig, axs = plt.subplots(1, 6, figsize=(15, 10), constrained_layout=True)

for j, ax in enumerate(axs):
    ax.plot(X[Y == 1, j], X[Y == 1, j + 1], 'o', color='r', label='Yama')
    ax.plot(X[Y == 2, j], X[Y == 2, j + 1], 'o', color='g', label='Rosa')
    ax.plot(X[Y == 3, j], X[Y == 3, j + 1], 'o', color='b', label='Canadian')

    ax.set_title('Scatterplot of Feature %s and %s' % (j + 1, (j + 2)))
    ax.set_xlabel('Feature %s' % (j + 1))
    ax.set_ylabel('Feature %s' % (j + 2))
    ax.legend()

plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
knn = KNeighborsClassifier()

#-----------------------------
#showing error rate to see which k value is the best.
#according to the graph, it is the valueof K = 10.
K=[1,2,5,10,50,100,140]
errTrain = np.zeros(len(K))
errTrainB = np.zeros(len(K))
for i in range(len(K)):
    knn = KNeighborsClassifier(n_neighbors=K[i])
    knn.fit(Xtr, Ytr) # where K is an integer, e.g. 1 for nearest neighborprediction
    Yhat = knn.predict(Xtr)        #TODO: predict results on training data
    errTrain[i] = np.mean(Yhat.reshape(Ytr.shape) != Ytr)   # TODO: count what fraction of predictions are wrong
    Yvahat = knn.predict(Xva)
    errTrainB[i] = np.mean(Yvahat.reshape(Yva.shape) != Yva)
plt.semilogx(K,errTrain, label = "Training error", color = 'red')
plt.semilogx(K,errTrainB, label = "Validation error", color = 'green')
plt.legend(loc="best")
plt.ylabel("Error value")
plt.xlabel("K value")
plt.title("Error rate functions based on K value")
plt.show()


#---------------------------
#accuracy of different K values
k_range= range(1,100)
scores = {}
score_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, Ytr) # where K is an integer, e.g. 1 for nearest neighbor prediction
    Yvahat = knn.predict(Xva)
    scores[k] = metrics.accuracy_score(Yva, Yvahat)
    score_list.append(metrics.accuracy_score(Yva, Yvahat))
plt.plot(k_range, score_list)
plt.xlabel("Value of K for KNN")
plt.ylabel("accuracy")
plt.show()


#---------------------------------

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(Xtr, Ytr) # where K is an integer, e.g. 1 for nearest neighbor prediction
Yvahat = knn.predict(Xva)
accuracy = (Xva.shape[0] - ((Yva != Yvahat).sum()))/(Xva.shape[0])




from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(Xtr, Ytr).predict(Xva)
accuracy = (Xva.shape[0] - ((Yva != y_pred).sum()))/(Xva.shape[0])

i=1