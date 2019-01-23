import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as model
import numpy
from PIL import Image
import os
from prettytable import PrettyTable
from mlxtend import plotting

__author__ = "Davide L. Manna"
__copyright__ = "Copyright 2018"
__credits__ = ["Davide L. Manna, Lorenzo Santolini"]

verbose = False
### GLOBALS ###
X = []
labels = []
colors_dict = {"dog": "orange", "house": "#A50104", "guitar": "blue", "person": "green"}
handles = []
colors = []
index = 0
mi = 0
stdDev = 0
wm = True
ws = True
nComp1 = 60
nComp2 = 6
nComp3 = 2
nCompLast = 6
firstPCAdone = False
components_backup = []
fig = None

def printScatters():
    """ Displays Projected Data Scatter Plot """
    global X_proj
    scats = plt.figure(figsize=(30, 10))
    ax = scats.add_subplot(131)
    ax.scatter(X_proj[:, 0], X_proj[:, 1], c=colors)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax = scats.add_subplot(132)
    ax.scatter(X_proj[:, 2], X_proj[:, 3], c=colors)
    ax.set_xlabel("Component 3")
    ax.set_ylabel("Component 4")

    ax = scats.add_subplot(133)
    ax.scatter(X_proj[:, 10], X_proj[:, 11], c=colors)
    ax.set_xlabel("Component 11")
    ax.set_ylabel("Component 12")

    scats.legend(handles=handles)

    scats.show()
    scats.savefig("scatterplots.png")

def performPca():
    """ Performs PCA on 4 different number of PCs and displays the reconstructed images VS the original one"""
    global pca
    #global pca1
    #global pca2
    #global pca3
    global X_proj
    global fig
    global X_invLast
    global X_inv1
    global X_inv2
    global X_inv3
    global firstPCAdone
    global components_backup


    #pca1 = PCA(nComp1)
    #pca2 = PCA(nComp2)
    #pca3 = PCA(nComp3)

    if not firstPCAdone:
        pca = PCA()
        scaled = preprocessing.scale(X, with_mean=wm, with_std=ws)
        pca.fit(scaled)
        print("[OK] PCA fitting")
        components_backup = pca.components_
        compLast = components_backup[-nCompLast:]
        comp1 = components_backup[:nComp1]
        comp2 = components_backup[:nComp2]
        comp3 = components_backup[:nComp3]
        print("[OK] Components Taken")
        X_proj = pca.transform(scaled)
        print("[OK] PCA transform")
        firstPCAdone = True
    else:
        compLast = components_backup[-nCompLast:]
        comp1 = components_backup[:nComp1]
        comp2 = components_backup[:nComp2]
        comp3 = components_backup[:nComp3]
        print("[OK] Components Taken")

    """ # METHOD 1
    pca.components_ = compLast
    X_projLast = pca.transform(scaled)
    X_invLast = pca.inverse_transform(X_projLast)*stdDev+mi
    print("[OK] PCA last",nCompLast)
    
    pca.components_ = comp1
    X_proj1 = pca.transform(scaled)
    X_inv1 = pca.inverse_transform(X_proj1)*stdDev+mi
    print("[OK] PCA",nComp1)

    pca.components_ = comp2
    X_proj2 = pca.transform(scaled)
    X_inv2 = pca.inverse_transform(X_proj2)*stdDev+mi
    print("[OK] PCA",nComp2)

    pca.components_ = comp3
    X_proj3 = pca.transform(scaled)
    X_inv3 = pca.inverse_transform(X_proj3)*stdDev+mi
    print("[OK] PCA",nComp3)
    """

    """ "# METHOD 2
    pca.components_ = compLast
    X_projLast = pca.transform(scaled)
    X_invLast = pca.inverse_transform(X_projLast)*stdDev+mi
    print("[OK] PCA last",nCompLast)
    
    X_proj1 = pca1.fit_transform(scaled)
    X_inv1 = pca1.inverse_transform(X_proj1)*stdDev+mi
    print("[OK] PCA",nComp1)

    X_proj2 = pca2.fit_transform(scaled)
    X_inv2 = pca2.inverse_transform(X_proj2)*stdDev+mi
    print("[OK] PCA",nComp2)

    X_proj3 = pca3.fit_transform(scaled)
    X_inv3 = pca3.inverse_transform(X_proj3)*stdDev+mi
    print("[OK] PCA",nComp3)
    """

    # METHOD 3 --- improvement: save transposed version of X_proj

    print("[PCA] performing Projected Matrix transposition and selection")
    tmp = numpy.transpose(X_proj)[:nComp1]
    X_proj1 = numpy.transpose(tmp)
    print("[PCA] done")
    pca.components_ = comp1
    X_inv1 = pca.inverse_transform(X_proj1)*stdDev+mi
    print("[OK] PCA",nComp1)


    print("[PCA] performing Projected Matrix transposition and selection")
    tmp = numpy.transpose(X_proj)[:nComp2]
    X_proj2 = numpy.transpose(tmp)
    print("[PCA] done")
    pca.components_ = comp2
    X_inv2 = pca.inverse_transform(X_proj2) * stdDev + mi
    print("[OK] PCA", nComp2)


    print("[PCA] performing Projected Matrix transposition and selection")
    tmp = numpy.transpose(X_proj)[:nComp3]
    X_proj3 = numpy.transpose(tmp)
    print("[PCA] done")
    pca.components_ = comp3
    X_inv3 = pca.inverse_transform(X_proj3) * stdDev + mi
    print("[OK] PCA", nComp3)


    print("[PCA] performing Projected Matrix transposition and selection")
    tmp = numpy.transpose(X_proj)[-nCompLast:]
    X_projLast = numpy.transpose(tmp)
    print("[PCA] done")
    pca.components_ = compLast
    X_invLast = pca.inverse_transform(X_projLast) * stdDev + mi
    print("[OK] PCA Last", nCompLast)

    print("\nDone!\n")
    return

def displayImageReconstruction():
    imgIndex = int(input("Which image to reconstruct? > "))
    global fig
    # Printing out reconstruction
    fig = plt.figure(figsize=(50, 10))

    ax = fig.add_subplot(151)
    ax.axis('off')
    ax.imshow(numpy.reshape(X[imgIndex], (227, 227, 3)))
    ax.set_title("Original image")

    ax = fig.add_subplot(152)
    ax.imshow(numpy.reshape(X_inv1[imgIndex, :]/255.0, (227, 227, 3)))
    ax.axis('off')
    ax.set_title(str(nComp1) + " PCs")

    ax = fig.add_subplot(153)
    ax.imshow(numpy.reshape(X_inv2[imgIndex, :]/255.0, (227, 227, 3)))
    ax.axis('off')
    ax.set_title(str(nComp2) + " PCs")

    ax = fig.add_subplot(154)
    ax.imshow(numpy.reshape(X_inv3[imgIndex, :]/255.0, (227, 227, 3)))
    ax.axis('off')
    ax.set_title(str(nComp3) + " PCs")

    ax = fig.add_subplot(155)
    ax.imshow(numpy.reshape(X_invLast[imgIndex, :]/255.0, (227, 227, 3)))
    ax.axis('off')
    ax.set_title("Last " + str(nCompLast) + " PCs")

    fig.show()
    print("\nDone!\n")


def savePCsImages(name="PCsDifferences.png"):
    global fig
    fig.savefig(name)


def printCumVariance():
    var = numpy.cumsum(numpy.round(pca.explained_variance_ratio_, 3) * 100)
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Cumulative Variance Analysis')
    plt.xlim(0, 60)
    plt.ylim(30, 100.5)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    plt.show()
    #plt.savefig("CumulativeVariance.png")

def setNComp():
    global nComp1
    global nComp2
    global nComp3
    global nCompLast
    global wm
    global ws
    nComp1 = int(input(f"N. of Components for first PCA  >\t"))
    nComp2 = int(input(f"N. of Components for second PCA  >\t"))
    nComp3 = int(input(f"N. of Components for third PCA  >\t"))
    nCompLast = int(input(f"N. of Last Components for fourth PCA  > "))
    c = input("With mean? y/n > ").lower()
    if c == "y" :
        wm = True
    else :
        wm = False
    c = input("With stdDev? y/n > ").lower()
    if c == "y" :
        ws = True
    else :
        ws = False

    print("\nDone!\n")

def classify():
    print("GaussianNB classifier started")
    if verbose: print("[INFO] Splitting Training Set from Test Set")
    X_train, X_test, Y_train, Y_test = model.train_test_split(X, labels, test_size=0.1)
    #nComponents = int(input("[DONE] Splitting\n[INPUT] N. of Components to use for pca? "))
    #pca = PCA(nComponents)
    pca = PCA(4)
    if verbose: print("[INFO] Performing PCA fit for", 4,"components")
    pca.fit(X_train)
    if verbose: print("[DONE] Fitting\n[INFO] Performing dimensionality reduction on train and test sets")

    X_train_proj = pca.transform(X_train)
    tmp12 = numpy.transpose(X_train_proj)[:2]
    tmp34 = numpy.transpose(X_train_proj)[2:]
    X_train_proj12 = numpy.transpose(tmp12)
    X_train_proj34 = numpy.transpose(tmp34)

    X_test_proj = pca.transform(X_test)
    tmp12 = numpy.transpose(X_test_proj)[:2]
    tmp34 = numpy.transpose(X_test_proj)[2:]
    X_test_proj12 = numpy.transpose(tmp12)
    X_test_proj34 = numpy.transpose(tmp34)

    if verbose: print("[DONE] Dim. reduction")

    #Experiment 0

    print("\n#########################################\n"
          "#### EXPERIMENT 0 - Original Features ###\n"
          "#########################################\n")
    if verbose: print("[INFO] Fitting Gaussian Naive-Bayes Classifier")
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    score_o = gnb.score(X_test, Y_test) * 100
    if verbose: print("[DONE] Fitting")  # \n[INFO] Predicted Labels Accuracy: {:.3f}%".format(score12))
    if verbose: print("[INFO] Predicting labels for test set...")
    prediction = gnb.predict(X_test)
    i = 0
    correct = 0
    T = PrettyTable()
    T.field_names = ["PREDICTED", "ACTUAL", "RESULT"]
    CT = PrettyTable()
    CT.field_names = ["CLASS", "CORRECT", "WRONG", "ACTUAL", "TOTAL PREDICTED", "CORRECT VS ACTUAL", "CORRECT VS TOTAL",
                      "% PREDICTIONS"]
    labels_stats = {"dog": [0, 0, 0, 0, 0, 0, 0], "house": [0, 0, 0, 0, 0, 0, 0], "guitar": [0, 0, 0, 0, 0, 0, 0],
                    "person": [0, 0, 0, 0, 0, 0, 0]}
    for s in prediction:
        actual = Y_test[i]
        res = True if s == actual else False
        T.add_row([s, actual, res])
        if res:
            correct = correct + 1
            v0 = labels_stats[s][0]
            v3 = labels_stats[s][3]
            labels_stats[s][0] = v0 + 1
            labels_stats[s][3] = v3 + 1
        else:
            v1 = labels_stats[s][1]
            v3 = labels_stats[s][3]
            labels_stats[s][1] = v1 + 1
            labels_stats[s][3] = v3 + 1
        v2 = labels_stats[actual][2]
        labels_stats[actual][2] = v2 + 1
        i = i + 1
    c = input("Do you want to print the predicted values? (y/n)\n> ")
    if c.lower() == "y":
        print(T)
    c = input("Do you want to print the predicted class' stats? (y/n)\n> ")
    if c.lower() == "y":
        for k, v in labels_stats.items():
            tmp = v[0] / v[2] * 100
            v[4] = tmp
            if v[3] != 0:
                tmp = v[0]/v[3] * 100
            else:
                tmp = 0
            v[5] = tmp
            tmp = v[3] / i * 100
            v[6] = tmp
            row = numpy.concatenate(([k], v))
            numpy.set_printoptions(precision=3)
            CT.add_row(row)
        print(CT)
    print("[RESULTS]\n\t|_\tTotal predictions:", i, "\n\t|_\tCorrect predictions:", correct)
    print("\t|_\tRatio: {:.3f}%".format((correct * 100 / i)))

    c_ = ["orange", "#A50104", "blue", "green"]
    cmap = mcolors.ListedColormap(c_[:len(numpy.unique(Y_train))])
    label_map = {"dog": 0, "guitar": 1, "house": 2, "person": 3}
    colors_ = numpy.array([colors_dict[l] for l in Y_test])

    """
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xvalues, yvalues = numpy.meshgrid(numpy.arange(x_min, x_max, 20), numpy.arange(y_min, y_max, 20))
    z = gnb.predict(numpy.c_[xvalues.ravel(), yvalues.ravel()])
    z = numpy.array([label_map[s] for s in z])
    z = numpy.reshape(z, xvalues.shape)
    """
    fig = plt.figure(figsize=(30, 10))

    """
    ax = fig.add_subplot(131)
    CS = ax.contourf(xvalues, yvalues, z, alpha=0.65, cmap=cmap)
    ax.axis('off')
    ax.scatter(x=X_test[:, 0], y=X_test[:, 1], c=colors_, s=20, edgecolor='k')
    ax.set_title("Decision Boundaries Obtained w/ Original Features and Test Data Points")
    """

    #Experiment 1

    print("\n#########################################\n"
          "### EXPERIMENT 1 - Components 1 and 2 ###\n"
          "#########################################\n")
    if verbose: print("[INFO] Fitting Gaussian Naive-Bayes Classifier")
    gnb = GaussianNB()
    gnb.fit(X_train_proj12, Y_train)
    score12 = gnb.score(X_test_proj12, Y_test) * 100
    if verbose: print("[DONE] Fitting")#\n[INFO] Predicted Labels Accuracy: {:.3f}%".format(score12))
    if verbose: print("[INFO] Predicting labels for test set...")
    prediction = gnb.predict(X_test_proj12)
    i = 0
    correct = 0
    T = PrettyTable()
    T.field_names = ["PREDICTED","ACTUAL","RESULT"]
    CT = PrettyTable()
    CT.field_names = ["CLASS", "CORRECT", "WRONG", "ACTUAL", "TOTAL PREDICTED", "CORRECT VS ACTUAL", "CORRECT VS TOTAL", "% PREDICTIONS"]
    labels_stats = {"dog":[0,0,0,0,0,0,0], "house":[0,0,0,0,0,0,0], "guitar":[0,0,0,0,0,0,0], "person":[0,0,0,0,0,0,0]}
    for s in prediction:
        actual = Y_test[i]
        res = True if s == actual else False
        T.add_row([s,actual,res])
        if res:
            correct = correct +1
            v0 = labels_stats[s][0]
            v3 = labels_stats[s][3]
            labels_stats[s][0] = v0+1
            labels_stats[s][3] = v3+1
        else:
            v1 = labels_stats[s][1]
            v3 = labels_stats[s][3]
            labels_stats[s][1] = v1+1
            labels_stats[s][3] = v3+1
        v2 = labels_stats[actual][2]
        labels_stats[actual][2] = v2+1
        i = i+1
    c = input("Do you want to print the predicted values? (y/n)\n> ")
    if c.lower() == "y":
        print(T)
    c = input("Do you want to print the predicted class' stats? (y/n)\n> ")
    if c.lower() == "y":
        for k,v in labels_stats.items():
            tmp = v[0]/v[2] * 100
            v[4] = tmp
            if v[3] != 0:
                tmp = v[0]/v[3] * 100
            else:
                tmp = 0
            v[5] = tmp
            tmp = v[3]/i * 100
            v[6] = tmp
            row = numpy.concatenate(([k],v))
            numpy.set_printoptions(precision=3)
            CT.add_row(row)
        print(CT)
    print("[RESULTS]\n\t|_\tTotal predictions:", i, "\n\t|_\tCorrect predictions:", correct)
    print("\t|_\tRatio: {:.3f}%".format((correct * 100 / i)))

    c_ = ["orange","#A50104","blue","green"]
    cmap = mcolors.ListedColormap(c_[:len(numpy.unique(Y_train))])
    label_map = {"dog": 0, "guitar": 1, "house": 2, "person": 3}
    colors_ = numpy.array([colors_dict[l] for l in Y_test])

    x_min, x_max = X_train_proj[:, 0].min() - 1, X_train_proj[:, 0].max() + 1
    y_min, y_max = X_train_proj[:, 1].min() - 1, X_train_proj[:, 1].max() + 1
    xvalues, yvalues = numpy.meshgrid(numpy.arange(x_min, x_max, 20), numpy.arange(y_min, y_max, 20))
    z = gnb.predict(numpy.c_[xvalues.ravel(), yvalues.ravel()])
    z = numpy.array([label_map[s] for s in z])
    z = numpy.reshape(z, xvalues.shape)

    #fig = plt.figure(figsize=(30, 15))

    ax = fig.add_subplot(132)
    CS = ax.contourf(xvalues, yvalues, z, alpha=0.65, cmap=cmap)
    ax.axis('off')
    ax.scatter(x=X_test_proj12[:,0],y=X_test_proj12[:,1],c=colors_, s=20, edgecolor='k')
    ax.set_title("Decision Boundaries Obtained w/ Components 1-2 and Test Data Points")

    #Experiment 2

    print("\n#########################################\n"
          "### EXPERIMENT 2 - Components 3 and 4 ###\n"
          "#########################################\n")
    if verbose: print("[INFO] Fitting Gaussian Naive-Bayes Classifier")
    gnb = GaussianNB()
    gnb.fit(X_train_proj34, Y_train)
    score34 = gnb.score(X_test_proj34, Y_test) * 100
    if verbose: print("[DONE] Fitting")#\n[INFO] Predicted Labels Accuracy: {:.3f}%".format(score34))
    if verbose: print("[INFO] Predicting labels for test set...")
    prediction = gnb.predict(X_test_proj34)
    i = 0
    correct = 0
    T = PrettyTable()
    T.field_names = ["PREDICTED", "ACTUAL", "RESULT"]
    CT = PrettyTable()
    CT.field_names = ["CLASS", "CORRECT", "WRONG", "ACTUAL", "TOTAL PREDICTED", "CORRECT VS ACTUAL", "CORRECT VS TOTAL",
                      "% PREDICTIONS"]
    labels_stats = {"dog": [0, 0, 0, 0, 0, 0, 0], "house": [0, 0, 0, 0, 0, 0, 0], "guitar": [0, 0, 0, 0, 0, 0, 0],
                    "person": [0, 0, 0, 0, 0, 0, 0]}
    for s in prediction:
        actual = Y_test[i]
        res = True if s == actual else False
        T.add_row([s, actual, res])
        if res:
            correct = correct + 1
            v0 = labels_stats[s][0]
            v3 = labels_stats[s][3]
            labels_stats[s][0] = v0 + 1
            labels_stats[s][3] = v3 + 1
        else:
            v1 = labels_stats[s][1]
            v3 = labels_stats[s][3]
            labels_stats[s][1] = v1 + 1
            labels_stats[s][3] = v3 + 1
        v2 = labels_stats[actual][2]
        labels_stats[actual][2] = v2 + 1
        i = i + 1
    c = input("Do you want to print the predicted values? (y/n)\n> ")
    if c.lower() == "y":
        print(T)
    c = input("Do you want to print the predicted class' stats? (y/n)\n> ")
    if c.lower() == "y":
        for k, v in labels_stats.items():
            tmp = v[0] / v[2] * 100
            v[4] = tmp
            if v[3] != 0:
                tmp = v[0]/v[3] * 100
            else:
                tmp = 0
            v[5] = tmp
            tmp = v[3] / i * 100
            v[6] = tmp
            row = numpy.concatenate(([k], v))
            numpy.set_printoptions(precision=3)
            CT.add_row(row)
        print(CT)
    print("[RESULTS]\n\t|_\tTotal predictions:", i, "\n\t|_\tCorrect predictions:", correct)
    print("\t|_\tRatio: {:.3f}%".format((correct * 100 / i)))

    z = gnb.predict(numpy.c_[xvalues.ravel(), yvalues.ravel()])
    z = numpy.array([label_map[s] for s in z])
    z = numpy.reshape(z, xvalues.shape)

    ax = fig.add_subplot(133)
    ax.contourf(xvalues, yvalues, z, alpha=0.65, cmap=cmap)
    ax.axis('off')
    ax.scatter(x=X_test_proj34[:, 0], y=X_test_proj34[:, 1], c=colors_, s=20, edgecolor='k')
    ax.set_title("Decision Boundaries Obtained w/ Components 3-4 and Test Data Points")

    bar_labels = ["Originals", "Comp12", "Comp34"]
    scores = [score_o, score12, score34]
    ax = fig.add_subplot(131)
    po, p12, p34 = ax.bar(numpy.arange(1,4), scores, width=0.6)
    po.set_facecolor('m')
    p12.set_facecolor("g")
    p34.set_facecolor("b")
    ax.set_xticks(numpy.arange(1,4))
    ax.set_xticklabels(bar_labels)
    ax.set_ylim([0,100])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy of Classifiers")

    print("\n ------------ SUMMARY ------------ \n")
    print("Experiment 0 prediction accuracy: {:.3f}%".format(score_o))
    print("Experiment 1 prediction accuracy: {:.3f}%".format(score12))
    print("Experiment 2 prediction accuracy: {:.3f}%\n".format(score34))

    fig.legend(handles=handles)
    fig.show()
    fig.savefig("decision_boundaries.png")


def printMenu():
    print("Select one of the following options:"
          "\n\t1. Perform PCA"
          "\n\t2. Set number of components and standardization"
          "\n\t3. Display Cumulative Variance"
          "\n\t4. Save PCA Image Reconstructions"
          "\n\t5. Print Scatter Plot"
          "\n\t6. Print Image Reconstruction"
          "\n\t7. Run Gaussian Naive-Bayes Classifier"
          "\n\t8. Exit")


switch = {
    "1": performPca,
    "2": setNComp,
    "3": printCumVariance,
    "4": savePCsImages,
    "5": printScatters,
    "6": displayImageReconstruction,
    "7": classify,
    "8": exit
}


if __name__ == '__main__':

    for k, v in colors_dict.items():
        handles.append(mpatches.Patch(color=v, label=k))
    for root, subFolders, fs in os.walk("PACS_homework"):
        for subRoot in subFolders:
            for sR, subSubFolders, files in os.walk(root + "/" + subRoot):
                for file in files:
                    img_array = numpy.asarray(
                        Image.open(root + "/" + subRoot + "/" + file))  # save img as array of 227*227*3 => x*y*rgb
                    flattened = img_array.ravel()  # flatten the image vector
                    X.append(flattened)
                    labels.append(subRoot)
                    colors.append(colors_dict[subRoot])

    mi = numpy.mean(X)
    stdDev = numpy.std(X)

    while True:
        printMenu()
        c = input("> ")
        if c in switch:
            try:
                switch[c]()
                input("Press any key to continue...")
                sys.stdout.flush()
            except (IOError, ValueError, TypeError, RuntimeError) as err:
                print("Error: {0}".format(err))
        else:
            print("Operation not available\n")
    exit()

    """
    fig = plt.figure(figsize=(50,10))
    plt.axis('off')
    ax = fig.add_subplot(151)
    ax.axis('off')
    ax.imshow(numpy.reshape(X[700],(227,227,3)))
    ax.set_title("Original image")

    X = preprocessing.scale(X, with_std=True, with_mean=True)

    stdDev2 = numpy.std(X)
    pca = PCA()
    pca60 = PCA(250)
    pca6 = PCA(6)
    pca2 = PCA(2)

    pca.fit(X)
    pca.components_ = pca.components_[-6:]
    X_projl6 = pca.transform(X)

    #tmp = numpy.transpose(X_proj)[:250]
    #X_proj60 = numpy.transpose(tmp)
    X_proj60 = pca60.fit_transform(X)
    comp60 = pca.components_[:250]

    #tmp = numpy.transpose(X_proj)[:6]
    #X_proj6 = numpy.transpose(tmp)
    X_proj6 = pca6.fit_transform(X)
    comp6 = pca.components_[:6]

    #tmp = numpy.transpose(X_proj)[:2]
    #X_proj2 = numpy.transpose(tmp)
    X_proj2 = pca2.fit_transform(X)
    comp2 = pca.components_[:2]

    tmp = numpy.transpose(X_proj)[-6:0]
    X_projl6 = numpy.transpose(tmp)
    compl6 = pca.components_[-6:0]
    """
    # cumulative variance
    """
    var = numpy.cumsum(numpy.round(pca.explained_variance_ratio_,3)*100)
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Cumulative Variance Analysis')
    plt.xlim(0, 60)
    plt.ylim(30, 100.5)
    plt.style.context('seaborn-whitegrid')

    plt.plot(var)
    plt.savefig("CumulativeVariance.png")
    exit()
    """
    # To display reconstructed img
    """
    #pca.components_ = comp60
    X_inv = pca60.inverse_transform(X_proj60)*stdDev +mi
    ax = fig.add_subplot(152)
    ax.imshow(numpy.reshape(X_inv[700,:]/255,(227,227,3)))
    ax.axis('off')
    ax.set_title("60 PCs")
    print("X_inv.shape = ",X_inv.shape)
    print("X_proj60.shape = ",X_proj60.shape)
    print("pca60.components_.shape = ",pca60.components_.shape)
    print(X_inv[400,:])

    #pca.components_ = comp6
    X_inv = pca6.inverse_transform(X_proj6)*stdDev +mi
    ax = fig.add_subplot(153)
    ax.imshow(numpy.reshape(X_inv[700, :]/255, (227, 227, 3)))
    ax.axis('off')
    ax.set_title("6 PCs")

    #pca.components_ = comp2
    X_inv = pca2.inverse_transform(X_proj2)*stdDev +mi
    ax = fig.add_subplot(154)
    ax.imshow(numpy.reshape(X_inv[700, :]/255, (227, 227, 3)))
    ax.axis('off')
    ax.set_title("2 PCs")

    #pca.components_ = compl6
    X_inv = pca.inverse_transform(X_projl6)*stdDev +mi
    ax = fig.add_subplot(155)
    ax.imshow(numpy.reshape(X_inv[700, :]/255, (227, 227, 3)))
    ax.axis('off')
    ax.set_title("last 6 PCs")

    fig.show()
    fig.savefig("PCsDifferences_img400.png")
    exit()

    plt.axis('off')
    plt.imshow(numpy.reshape(X_inv[0], (227,227,3)).astype(numpy.uint8))
    plt.show()
    exit()
    """
