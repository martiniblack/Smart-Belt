import csv
import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from skimage import util
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from scipy import signal

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def extractDataFromFile(file) :
    data = []
    with open(file) as tsv:
        for column in zip(*[line for line in csv.reader(tsv, dialect="excel-tab")]):
            data.append(list(map(int, column)))
    return data

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def findAverageDPAndDF(windowSlices) :
    DP_list = []
    DF_list = []
    for i in range(windowSlices.shape[0]) :
        fft = np.fft.fft(windowSlices[i])
        n = windowSlices[i].size
        freq = np.fft.fftfreq(n, d=0.001)
        DP_list.append(np.max(np.abs(fft)))
        DF_list.append(freq[np.argmax(np.abs(fft))])

    DP_AVG = np.mean(DP_list)
    DF_AVG = np.mean(DF_list)
    DP_STD = np.std(DP_list)
    DF_STD = np.std(DF_list)

    return DP_AVG, DF_AVG, DP_STD, DF_STD

def findPercentageFeature(windowSlices) :
    NORM_PC = 0
    BRODY_PC = 0
    TACHY_PC = 0
    OTHER_PC = 0

    for i in range(windowSlices.shape[0]) :
        fft = np.fft.fft(windowSlices[i])
        n = windowSlices[i].size
        freq = np.fft.fftfreq(n, d=0.001)


        DF = freq[np.argmax(np.abs(fft))]
        if (DF < (0.5/60.0)) or (DF>(4.5/60.0)) :
            OTHER_PC +=1
        elif (DF >= (0.5/60.0)) and (DF < (2/60.0)) :
            BRODY_PC +=1
        elif (DF >= (2/60.0)) and (DF <= (4/60.0)) :
            NORM_PC +=1
        else :
            TACHY_PC  +=1

    OTHER_PC = float(OTHER_PC/float(windowSlices.shape[0]))
    BRODY_PC = float(BRODY_PC/float(windowSlices.shape[0]))
    NORM_PC = float(NORM_PC/float(windowSlices.shape[0]))
    TACHY_PC = float(TACHY_PC/float(windowSlices.shape[0]))

    return NORM_PC, BRODY_PC, TACHY_PC, OTHER_PC

def findOverallDPAndDF(raw_data_norm_bef) :
    fft = np.fft.fft(raw_data_norm_bef)
    n = raw_data_norm_bef.size
    freq = np.fft.fftfreq(n, d=0.001)
    return np.max(np.abs(fft)), freq[np.argmax(np.abs(fft))]


def extractFeaturesFromFile(file) :
    # STEP 0 : extracting raw data
    raw_data_norm_bef = np.array(extractDataFromFile(file))
    print("STEP 0 : Raw data extraction ... DONE ")

    # STEP 1 : normalize (standardize) raw data
    raw_data_norm_bef = preprocessing.scale(raw_data_norm_bef)
    print("STEP 1 : Raw data standardization ... DONE ")

    # STEP 2 : Low pass filtering with butterworth
    raw_data_norm_bef = butter_lowpass_filter(raw_data_norm_bef, 1.5, 1000, order=3)
    print("STEP 2 : Butterworth low path filtering ... DONE ")

    # STEP 3 : Window of 4min, for each window calculate DF and DP with single sided fast Fourier transform (fft), then we take the average
    winLen = 4*60*1000
    step = int(0.25*winLen)

    sliceList = []
    FeatureList = []
    for i in range(raw_data_norm_bef.shape[0]) :
        sliceList.append(util.view_as_windows(raw_data_norm_bef[i], window_shape=(winLen,), step=step))
        DP_AVG, DF_AVG, DP_STD, DF_STD = findAverageDPAndDF(sliceList[i])
        NORM_PC, BRODY_PC, TACHY_PC, OTHER_PC = findPercentageFeature(sliceList[i])

        DP_OA, DF_OA = findOverallDPAndDF(raw_data_norm_bef[i])

        featureVec = []
        featureVec.append(DP_AVG)
        featureVec.append(DF_AVG)
        featureVec.append(DP_STD)
        featureVec.append(DF_STD)
        featureVec.append(NORM_PC)
        featureVec.append(BRODY_PC)
        featureVec.append(TACHY_PC)
        featureVec.append(OTHER_PC)
        featureVec.append(DP_OA)
        featureVec.append(DF_OA)
        FeatureList.append(featureVec)

    print("STEP 3 : Feature extraction ... DONE ")
    return FeatureList

def extractFeatureForLabel(file_bef, file_aft) :
    print("Before recording data treatment ... ")
    bef = extractFeaturesFromFile(file_bef)
    print("Before recording data treatment ... DONE")
    print("After recording data treatment ... ")
    aft = extractFeaturesFromFile(file_aft)
    print("After recording data treatment ... DONE")

    # Concatenation of before and after features
    feat = np.concatenate((bef, aft), axis=1)

    # DP_AVG_after / DP_AVG_BEF feature
    AFT_BEF_RATIO = np.array(aft)[:, 0] / np.array(bef)[:, 0]
    AFT_BEF_RATIO.reshape(AFT_BEF_RATIO.shape[0], 1)

    # Adding to feature vector
    feat = np.concatenate((feat, AFT_BEF_RATIO[:, None]), axis=1)

    return feat, bef, aft

def getFeatureNameByIndex(index) :
    if index == 0 :
        name = "DP_AVG_BEFORE"
    elif index == 1 :
        name = "DF_AVG_BEFORE"
    elif index == 2:
        name = "DP_STD_BEFORE"
    elif index == 3:
        name = "DF_STD_BEFORE"
    elif index == 4:
        name = "NORM_PC_BEFORE"
    elif index == 5:
        name = "BRODY_PC_BEFORE"
    elif index == 6:
        name = "TACHY_PC_BEFORE"
    elif index == 7:
        name = "OTHER_PC_BEFORE"
    elif index == 8 :
        name = "DP_OVERALL_BEFORE"
    elif index == 9 :
        name = "Df_OVERALL_BEFORE"
    elif index == 10 :
        name = "DP_AVG_AFTER"
    elif index == 11 :
        name = "DF_AVG_AFTER"
    elif index == 12:
        name = "DP_STD_AFTER"
    elif index == 13:
        name = "DF_STD_AFTER"
    elif index == 14:
        name = "NORM_PC_AFTER"
    elif index == 15:
        name = "BRODY_PC_AFTER"
    elif index == 16:
        name = "TACHY_PC_AFTER"
    elif index == 17:
        name = "OTHER_PC_AFTER"
    elif index == 18:
        name = "DP_OVERALL_AFTER"
    elif index == 19:
        name = "DF_OVERALL_AFTER"
    elif index == 20:
        name = "DF_AFTER_BEFORE_RATIO"
    else :
        name = "UNKNOWN"

    return name

def boxPlotFeatures(norm_feat, pd_feat):

    for i in range(norm_feat.shape[1]) :
        fig1, ax1 = plt.subplots()
        ax1.set_title(getFeatureNameByIndex(i))
        ax1.boxplot([norm_feat[:, i], pd_feat[:, i]])

    plt.show()

def normalizeFeatureVec(feat) :
    for i in range(feat.shape[1]) :
        feat[:,i] /= np.max(feat[:,i] )
    return feat

def classify(feat, labels) :
    loo = LeaveOneOut()
    loo.get_n_splits(feat)
    print("JACK KNIFE")
    clf = svm.SVC(kernel='linear', C=1000)

    accuracy = 0
    tot = 0
    for train_index, test_index in loo.split(feat):
        X_train, X_test = feat[train_index], feat[test_index]
        y_train, y_test = np.array(labels)[train_index.astype(int)], np.array(labels)[test_index]
        clf.fit(X_train, y_train)
        tot += 1
        print(clf.score(X_test, y_test))
        if clf.score(X_test, y_test) == 1 :
            accuracy += 1

    accuracy /= float(tot)
    print(accuracy)


def main():

    print("Normal data treatment ... ")
    norm_feat, norm_bef, norm_aft = extractFeatureForLabel("C:/Users/pierr/Documents/Mines ParisTech/Module 4/IoT & Healthcare/project_01/project data/norm_bef.txt",
                                       "C:/Users/pierr/Documents/Mines ParisTech/Module 4/IoT & Healthcare/project_01/project data/norm_aft.txt")
    print("Normal data treatment ... DONE")

    print("Parkinson Disease data treatment ... ")
    pd_feat, pd_bef, pd_aft = extractFeatureForLabel("C:/Users/pierr/Documents/Mines ParisTech/Module 4/IoT & Healthcare/project_01/project data/pd_bef.txt",
                                       "C:/Users/pierr/Documents/Mines ParisTech/Module 4/IoT & Healthcare/project_01/project data/pd_aft.txt")
    print("Parkinson Disease data treatment ... DONE")

    boxPlotFeatures(norm_feat, pd_feat)

    norm_feat_relevant = []
    pd_feat_relevant = []

    for i in range (norm_feat.shape[1]) :
        stat, pvalue = mannwhitneyu(norm_feat[:, i], pd_feat[:, i]) #wilcoxon(norm_feat[:8, i], pd_feat[:, i])
        print("p value " + str(pvalue) + " found for feature " + getFeatureNameByIndex(i))
        if pvalue < 0.04:
            #print("Relevant feature found for feature " + getFeatureNameByIndex(i) )
            norm_feat_relevant.append(norm_feat[:,i])
            pd_feat_relevant.append(pd_feat[:,i])

    labels = []
    for j in range(np.array(norm_feat_relevant).shape[1]) :
        labels.append(0)

    for k in range(np.array(pd_feat_relevant).shape[1]):
        labels.append(1)



    feat = np.concatenate((norm_feat_relevant, pd_feat_relevant), axis=1).T
    #print(feat.shape)
    #print(labels)
    feat = normalizeFeatureVec(feat)

    classify(feat, labels)

if __name__ == "__main__":
    main()