import pandas as pd
import sklearn.ensemble as es
import sklearn.tree as tr
import sklearn.neighbors as n
import sklearn.neural_network as nn
import sklearn.svm as s
import sklearn.utils as u
import sklearn.feature_selection as fs
import time as t
import numpy as np
import sklearn.metrics as m
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import seaborn as sb
import warnings as w
w.filterwarnings('ignore')

countR = 0
count21R = 0
countD = 0
count21D = 0
countK = 0
count21K = 0
countS = 0
count21S = 0
countM = 0
count21M = 0
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0


#Start of Reading Data


traindata = pd.read_csv("E:\\bukc\\FYP\\Data\\NSL-KDD\\KDDTest+.csv")
testdata21 = pd.read_csv("E:\\bukc\\FYP\\Data\\NSL-KDD\\KDDTest-21+.csv")
testdata = pd.read_csv("E:\\bukc\\FYP\\Data\\NSL-KDD\\KDDTrain+.csv")


#End of Reading Data


#Start of Preprocessing

traindata = u.shuffle(traindata)
testdata = u.shuffle(testdata)
testdata21 = u.shuffle(testdata21)

# itrain = int(len(traindata) * 0.25)
# itest = int(len(testdata) * 0.25)
# itest21 = int(len(testdata21) * 0.25)
# traindata = traindata[0:itrain+1]
# testdata = testdata[0:itest+1]
# testdata21 = testdata21[0:itest21+1]

atk_dict = {
    'normal': 'Normal',
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',
    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',    
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

atk_correct = {"Normal":1, "DoS":2, "Probe":3, "R2L":4, "U2R":5}

protocol_correct = {"tcp": 1, "icmp" : 2, "udp" : 3}

flag_correct = {"SF":1, "REJ":2, "RSTR":3, "RSTO":4, "S0":5, "S1":6, "S2":7, "S3":8, "OTH":9, "SH":10, "RSTOS0":11}


print("\nTrain data difficulty level count graph: \n")
t.sleep(1)
ax = sb.countplot(x='Difficulty Level', data=testdata)
plt.show()

print("\nTest data difficulty level count graph:\n")
t.sleep(1)
ax = sb.countplot(x='Difficulty Level', data=traindata)
plt.show()

print("\nTest data without difficulty level 21 difficulty level count graph:\n")
ax = sb.countplot(x='Difficulty Level', data=testdata21)
plt.show()

print("\nTrain data protocol count graph: \n")
t.sleep(1)
ax = sb.countplot(x='Protocol_Type', data=testdata)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(testdata)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()

print("\nTest data protocol count graph:\n")
t.sleep(1)
ax = sb.countplot(x='Protocol_Type', data=traindata)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(traindata)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()

print("\nTest data without difficulty level 21 protocol count graph:\n")
t.sleep(1)
ax = sb.countplot(x='Protocol_Type', data=testdata21)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(testdata21)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()


traindata = traindata.drop(["Duration"], axis=1)
testdata = testdata.drop(["Duration"], axis=1)
testdata21 = testdata21.drop(["Duration"], axis=1)


traindata = traindata.drop(["Service"], axis=1)
testdata = testdata.drop(["Service"], axis=1)
testdata21 = testdata21.drop(["Service"], axis=1)


traindata = traindata.drop(["Dst_Bytes"], axis=1)
testdata = testdata.drop(["Dst_Bytes"], axis=1)
testdata21 = testdata21.drop(["Dst_Bytes"], axis=1)


traindata = traindata.drop(["Land"], axis=1)
testdata = testdata.drop(["Land"], axis=1)
testdata21 = testdata21.drop(["Land"], axis=1)


traindata = traindata.drop(["Wrong_Fragment"], axis=1)
testdata = testdata.drop(["Wrong_Fragment"], axis=1)
testdata21 = testdata21.drop(["Wrong_Fragment"], axis=1)


traindata = traindata.drop(["Urgent"], axis=1)
testdata = testdata.drop(["Urgent"], axis=1)
testdata21 = testdata21.drop(["Urgent"], axis=1)


traindata = traindata.drop(["Hot"], axis=1)
testdata = testdata.drop(["Hot"], axis=1)
testdata21 = testdata21.drop(["Hot"], axis=1)


traindata = traindata.drop(["Num_Failed_Logins"], axis=1)
testdata = testdata.drop(["Num_Failed_Logins"], axis=1)
testdata21 = testdata21.drop(["Num_Failed_Logins"], axis=1)


traindata = traindata.drop(["Logged_In"], axis=1)
testdata = testdata.drop(["Logged_In"], axis=1)
testdata21 = testdata21.drop(["Logged_In"], axis=1)


traindata = traindata.drop(["Num_Compromised"], axis=1)
testdata = testdata.drop(["Num_Compromised"], axis=1)
testdata21 = testdata21.drop(["Num_Compromised"], axis=1)


traindata = traindata.drop(["Root_Shell"], axis=1)
testdata = testdata.drop(["Root_Shell"], axis=1)
testdata21 = testdata21.drop(["Root_Shell"], axis=1)


traindata = traindata.drop(["Su_Attempted"], axis=1)
testdata = testdata.drop(["Su_Attempted"], axis=1)
testdata21 = testdata21.drop(["Su_Attempted"], axis=1)


traindata = traindata.drop(["Num_Root"], axis=1)
testdata = testdata.drop(["Num_Root"], axis=1)
testdata21 = testdata21.drop(["Num_Root"], axis=1)


traindata = traindata.drop(["Num_File_Creations"], axis=1)
testdata = testdata.drop(["Num_File_Creations"], axis=1)
testdata21 = testdata21.drop(["Num_File_Creations"], axis=1)


traindata = traindata.drop(["Num_Shells"], axis=1)
testdata = testdata.drop(["Num_Shells"], axis=1)
testdata21 = testdata21.drop(["Num_Shells"], axis=1)


traindata = traindata.drop(["Num_Outbound_Cmds"], axis=1)
testdata = testdata.drop(["Num_Outbound_Cmds"], axis=1)
testdata21 = testdata21.drop(["Num_Outbound_Cmds"], axis=1)


traindata = traindata.drop(["Is_Host_Login"], axis=1)
testdata = testdata.drop(["Is_Host_Login"], axis=1)
testdata21 = testdata21.drop(["Is_Host_Login"], axis=1)


traindata = traindata.drop(["Count"], axis=1)
testdata = testdata.drop(["Count"], axis=1)
testdata21 = testdata21.drop(["Count"], axis=1)


traindata = traindata.drop(["Serror_Rate"], axis=1)
testdata = testdata.drop(["Serror_Rate"], axis=1)
testdata21 = testdata21.drop(["Serror_Rate"], axis=1)


traindata = traindata.drop(["Srv_Serror_Rate"], axis=1)
testdata = testdata.drop(["Srv_Serror_Rate"], axis=1)
testdata21 = testdata21.drop(["Srv_Serror_Rate"], axis=1)


traindata = traindata.drop(["Rerror_Rate"], axis=1)
testdata = testdata.drop(["Rerror_Rate"], axis=1)
testdata21 = testdata21.drop(["Rerror_Rate"], axis=1)


traindata = traindata.drop(["Srv_Rerror_Rate"], axis=1)
testdata = testdata.drop(["Srv_Rerror_Rate"], axis=1)
testdata21 = testdata21.drop(["Srv_Rerror_Rate"], axis=1)


traindata = traindata.drop(["Same_Srv_Rate"], axis=1)
testdata = testdata.drop(["Same_Srv_Rate"], axis=1)
testdata21 = testdata21.drop(["Same_Srv_Rate"], axis=1)


traindata = traindata.drop(["Dst_Host_Count"], axis=1)
testdata = testdata.drop(["Dst_Host_Count"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Count"], axis=1)


traindata = traindata.drop(["Dst_Host_Srv_Count"], axis=1)
testdata = testdata.drop(["Dst_Host_Srv_Count"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Srv_Count"], axis=1)


traindata = traindata.drop(["Dst_Host_Same_Srv_Rate"], axis=1)
testdata = testdata.drop(["Dst_Host_Same_Srv_Rate"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Same_Srv_Rate"], axis=1)


traindata = traindata.drop(["Dst_Host_Diff_Srv_Rate"], axis=1)
testdata = testdata.drop(["Dst_Host_Diff_Srv_Rate"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Diff_Srv_Rate"], axis=1)


traindata = traindata.drop(["Dst_Host_Serror_Rate"], axis=1)
testdata = testdata.drop(["Dst_Host_Serror_Rate"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Serror_Rate"], axis=1)


traindata = traindata.drop(["Dst_Host_Rerror_Rate"], axis=1)
testdata = testdata.drop(["Dst_Host_Rerror_Rate"], axis=1)
testdata21 = testdata21.drop(["Dst_Host_Rerror_Rate"], axis=1)


traindata = traindata.replace({"Class": atk_dict})
testdata = testdata.replace({"Class": atk_dict})
testdata21 = testdata21.replace({"Class": atk_dict})


# traindata = traindata[traindata.Class != "U2R"]
# testdata = testdata[testdata.Class != "U2R"]
# testdata21 = testdata21[testdata21.Class != "U2R"]
# traindata = traindata[traindata.Class != "R2L"]
# testdata = testdata[testdata.Class != "R2L"]
# testdata21 = testdata21[testdata21.Class != "R2L"]
# traindata = pd.DataFrame(data = traindata)
# testdata = pd.DataFrame(data = testdata)
# testdata21 = pd.DataFrame(data = testdata21)


normTest = testdata[testdata.Class == "Normal"]
normTest21 = testdata21[testdata21.Class == "Normal"]


DoSTest = testdata[testdata.Class == "DoS"]
DoSTest21 = testdata21[testdata21.Class == "DoS"]


probeTest = testdata[testdata.Class == "Probe"]
probeTest21 = testdata21[testdata21.Class == "Probe"]


R2LTest = testdata[testdata.Class == "R2L"]
R2LTest21 = testdata21[testdata21.Class == "R2L"]


U2RTest = testdata[testdata.Class == "U2R"]
U2RTest21 = testdata21[testdata21.Class == "U2R"]


print("Train data attack count graph: \n")
t.sleep(1)
ax = sb.countplot(x='Class', data=testdata)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(testdata)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()

print("\nTest data attack count graph:\n")
t.sleep(1)
ax = sb.countplot(x='Class', data=traindata)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(traindata)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()

print("\nTest data without difficulty level 21 attack count graph:\n")
t.sleep(1)
ax = sb.countplot(x='Class', data=testdata21)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(testdata21)), (p.get_x() + 0.22, p.get_height() + 2))
plt.show()


traindata = traindata.replace({"Class": atk_correct})
testdata = testdata.replace({"Class": atk_correct})
testdata21 = testdata21.replace({"Class": atk_correct})


traindata = traindata.replace({"Protocol_Type": protocol_correct})
testdata = testdata.replace({"Protocol_Type": protocol_correct})
testdata21 = testdata21.replace({"Protocol_Type": protocol_correct})

traindata = traindata.replace({"Flag": flag_correct})
testdata = testdata.replace({"Flag": flag_correct})
testdata21 = testdata21.replace({"Flag": flag_correct})


for col in traindata.columns:
    if traindata[col].dtype == type(object):
        le = pp.LabelEncoder()
        traindata[col] = le.fit_transform(traindata[col])
        
for col in testdata.columns:
    if testdata[col].dtype == type(object):
        le = pp.LabelEncoder()
        testdata[col] = le.fit_transform(testdata[col])
        
for col in testdata21.columns:
    if testdata21[col].dtype == type(object):
        le = pp.LabelEncoder()
        testdata21[col] = le.fit_transform(testdata21[col])


#End of Preprocessing


#Start of Feature Selection


featsTrainD = traindata.values[:,0:12]

lblsTrainD = traindata.values[:,12]


featsTrain = traindata.values[:,0:13]

featsTest = testdata.values[:,0:13]

lblsTrain = traindata.values[:,13]

lblsTest = testdata.values[:,13]

featsTest21 = testdata21.values[:,0:13]

lblsTest21 = testdata21.values[:,13]



#End of Feature Selection


#Start of Random Forest Model


modelR = es.RandomForestClassifier(n_estimators = 100, criterion = "entropy")
model21R = es.RandomForestClassifier(n_estimators = 100, criterion="entropy")
modelR.fit(featsTrain, lblsTrain)
lblsPredR = modelR.predict(featsTest)
for a,b in zip(lblsTest, lblsPredR):
    if a == b:
        countR += 1
        if a == 1:
            countNorm += 1
        elif a == 2:
            countDoS += 1
        elif a == 3:
            countProbe += 1
        elif a == 4:
            countR2L += 1
        elif a == 5:
            countU2R += 1
accR = (round(countR/(len(featsTest)), 3)) * 100
normaccR = (round(countNorm / len(normTest), 3)) * 100
DoSaccR = (round(countDoS / len(DoSTest), 3)) * 100
ProbeaccR = (round(countProbe / len(probeTest), 3)) * 100
R2LaccR = (round(countR2L / len(R2LTest), 3)) * 100
U2RaccR = (round(countU2R / len(U2RTest), 3)) * 100
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0
print("\n\n\tWith NSL-KDD train and test data using Random Forest\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest, lblsPredR),"\n\n")
print(m.classification_report(lblsTest, lblsPredR))
print("\nAccuracy = ", accR, "%\n\nAccuracy of normal = ", normaccR, "%\nAccuracy of DoS = ", DoSaccR, "%\nAccuracy of Probe = ", ProbeaccR, "%\nAccuracy of R2L = ", R2LaccR, "%\nAccuracy of U2R = ", U2RaccR, "%\n")
print("-------------------------------------------------------\n")

# model21R.fit(featsTrain, lblsTrain)
# lblsPred21R = model21R.predict(featsTest21)
# for a,b in zip(lblsTest21, lblsPred21R):
#     if a == b:
#         count21R += 1
#         if a == 1:
#             countNorm += 1
#         elif a == 2:
#             countDoS += 1
#         elif a == 3:
#             countProbe += 1
#         elif a == 4:
#             countR2L += 1
#         elif a == 5:
#             countU2R += 1
# acc21R = (round(count21R/(len(featsTest21)), 3)) * 100
# normacc21R = (round(countNorm / len(normTest21), 3)) * 100
# DoSacc21R = (round(countDoS / len(DoSTest21), 3)) * 100
# Probeacc21R = (round(countProbe / len(probeTest21), 3)) * 100
# R2Lacc21R = (round(countR2L / len(R2LTest21), 3)) * 100
# U2Racc21R = (round(countU2R / len(U2RTest21), 3)) * 100
# countNorm = 0
# countDoS = 0
# countProbe = 0
# countR2L = 0
# countU2R = 0
# print("\tWith NSL-KDD test data without difficulty level 21 using Random Forest\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest21, lblsPred21R),"\n\n")
# print(m.classification_report(lblsTest21, lblsPred21R))
# print("\nAccuracy = ", acc21R, "%\n\nAccuracy of normal = ", normacc21R, "%\nAccuracy of DoS = ", DoSacc21R, "%\nAccuracy of Probe = ", Probeacc21R, "%\nAccuracy of R2L = ", R2Lacc21R, "%\nAccuracy of U2R = ", U2Racc21R, "%\n")
# print("-------------------------------------------------------\n")

rfecvR = fs.RFECV(estimator = modelR, scoring = "accuracy").fit(featsTest, lblsTest)
plt.figure()
plt.xlabel("Features selected")
plt.ylabel("Cross Validation Score")
plt.title('RFECV Random Forest')
plt.plot(range(1, len(rfecvR.grid_scores_) + 1), rfecvR.grid_scores_)


#End of Random Forest Model


#Start of Decision Tree Model


modelD = tr.DecisionTreeClassifier(criterion = "entropy", max_depth = 5)
model21D = tr.DecisionTreeClassifier(criterion = "entropy", max_depth = 5)
modelD.fit(featsTrain, lblsTrain)
lblsPredD = modelD.predict(featsTest)
for a,b in zip(lblsTest, lblsPredD):
    if a == b:
        countD += 1
        if a == 1:
            countNorm += 1
        elif a == 2:
            countDoS += 1
        elif a == 3:
            countProbe += 1
        elif a == 4:
            countR2L += 1
        elif a == 5:
            countU2R += 1
accD = (round(countD/(len(featsTest)), 3)) * 100
normaccD = (round(countNorm / len(normTest), 3)) * 100
DoSaccD = (round(countDoS / len(DoSTest), 3)) * 100
ProbeaccD = (round(countProbe / len(probeTest), 3)) * 100
R2LaccD = (round(countR2L / len(R2LTest), 3)) * 100
U2RaccD = (round(countU2R / len(U2RTest), 3)) * 100
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0
print("\n\tWith NSL-KDD train and test data using Decision Tree\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest, lblsPredD),"\n\n")
print(m.classification_report(lblsTest, lblsPredD))
print("\nAccuracy = ", accD, "%\n\nAccuracy of normal = ", normaccD, "%\nAccuracy of DoS = ", DoSaccD, "%\nAccuracy of Probe = ", ProbeaccD, "%\nAccuracy of R2L = ", R2LaccD, "%\nAccuracy of U2R = ", U2RaccD, "%\n")
print("-------------------------------------------------------\n")

# model21D.fit(featsTrain, lblsTrain)
# lblsPred21D = model21D.predict(featsTest21)
# for a,b in zip(lblsTest21, lblsPred21D):
#     if a == b:
#         count21D += 1
#         if a == 1:
#             countNorm += 1
#         elif a == 2:
#             countDoS += 1
#         elif a == 3:
#             countProbe += 1
#         elif a == 4:
#             countR2L += 1
#         elif a == 5:
#             countU2R += 1
# acc21D = (round(count21D/(len(featsTest21)), 3)) * 100
# normacc21D = (round(countNorm / len(normTest21), 3)) * 100
# DoSacc21D = (round(countDoS / len(DoSTest21), 3)) * 100
# Probeacc21D = (round(countProbe / len(probeTest21), 3)) * 100
# R2Lacc21D = (round(countR2L / len(R2LTest21), 3)) * 100
# U2Racc21D = (round(countU2R / len(U2RTest21), 3)) * 100
# countNorm = 0
# countDoS = 0
# countProbe = 0
# countR2L = 0
# countU2R = 0
# print("\tWith NSL-KDD test data without difficulty level 21 using Decision Tree\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest21, lblsPred21D),"\n\n")
# print(m.classification_report(lblsTest21, lblsPred21D))
# print("\nAccuracy = ", acc21D, "%\n\nAccuracy of normal = ", normacc21D, "%\nAccuracy of DoS = ", DoSacc21D, "%\nAccuracy of Probe = ", Probeacc21D, "%\nAccuracy of R2L = ", R2Lacc21D, "%\nAccuracy of U2R = ", U2Racc21D, "%\n")
# print("-------------------------------------------------------\n")

rfecvD = fs.RFECV(estimator = modelD, scoring = "accuracy").fit(featsTest, lblsTest)
plt.figure()
plt.xlabel("Features selected")
plt.ylabel("Cross Validation Score")
plt.title('RFECV Decision Tree')
plt.plot(range(1, len(rfecvD.grid_scores_) + 1), rfecvD.grid_scores_)


#End of Decision Tree Model


#Start of K-Neighbors Model


modelK = n.KNeighborsClassifier(n_neighbors = 100)
model21K = n.KNeighborsClassifier(n_neighbors = 100)
modelK.fit(featsTrain, lblsTrain)
lblsPredK = modelK.predict(featsTest)
for a,b in zip(lblsTest, lblsPredK):
    if a == b:
        countK += 1
        if a == 1:
            countNorm += 1
        elif a == 2:
            countDoS += 1
        elif a == 3:
            countProbe += 1
        elif a == 4:
            countR2L += 1
        elif a == 5:
            countU2R += 1
accK = (round(countK/(len(featsTest)), 3)) * 100
normaccK = (round(countNorm / len(normTest), 3)) * 100
DoSaccK = (round(countDoS / len(DoSTest), 3)) * 100
ProbeaccK = (round(countProbe / len(probeTest), 3)) * 100
R2LaccK = (round(countR2L / len(R2LTest), 3)) * 100
U2RaccK = (round(countU2R / len(U2RTest), 3)) * 100
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0
print("\n\tWith NSL-KDD train and test data using K-Neighbors\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest, lblsPredK),"\n\n")
print(m.classification_report(lblsTest, lblsPredK))
print("\nAccuracy = ", accK, "%\n\nAccuracy of normal = ", normaccK, "%\nAccuracy of DoS = ", DoSaccK, "%\nAccuracy of Probe = ", ProbeaccK, "%\nAccuracy of R2L = ", R2LaccK, "%\nAccuracy of U2R = ", U2RaccK, "%\n")
print("-------------------------------------------------------\n")

# model21K.fit(featsTrain, lblsTrain)
# lblsPred21K = model21K.predict(featsTest21)
# for a,b in zip(lblsTest21, lblsPred21K):
#     if a == b:
#         count21K += 1
#         if a == 1:
#             countNorm += 1
#         elif a == 2:
#             countDoS += 1
#         elif a == 3:
#             countProbe += 1
#         elif a == 4:
#             countR2L += 1
#         elif a == 5:
#             countU2R += 1
# acc21K = (round(count21K/(len(featsTest21)), 3)) * 100
# normacc21K = (round(countNorm / len(normTest21), 3)) * 100
# DoSacc21K = (round(countDoS / len(DoSTest21), 3)) * 100
# Probeacc21K = (round(countProbe / len(probeTest21), 3)) * 100
# R2Lacc21K = (round(countR2L / len(R2LTest21), 3)) * 100
# U2Racc21K = (round(countU2R / len(U2RTest21), 3)) * 100
# countNorm = 0
# countDoS = 0
# countProbe = 0
# countR2L = 0
# countU2R = 0
# print("\tWith NSL-KDD test data without difficulty level 21 using K-Neighbors\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest21, lblsPred21K),"\n\n")
# print(m.classification_report(lblsTest21, lblsPred21K))
# print("\nAccuracy = ", acc21K, "%\n\nAccuracy of normal = ", normacc21K, "%\nAccuracy of DoS = ", DoSacc21K, "%\nAccuracy of Probe = ", Probeacc21K, "%\nAccuracy of R2L = ", R2Lacc21K, "%\nAccuracy of U2R = ", U2Racc21K, "%\n")
# print("-------------------------------------------------------\n")


#End of K-Neighbors Model


#Start of MLP Model


modelM = nn.MLPClassifier()
model21M = nn.MLPClassifier()
modelM.fit(featsTrain, lblsTrain)
lblsPredM = modelM.predict(featsTest)
for a,b in zip(lblsTest, lblsPredM):
    if a == b:
        countM += 1
        if a == 1:
            countNorm += 1
        elif a == 2:
            countDoS += 1
        elif a == 3:
            countProbe += 1
        elif a == 4:
            countR2L += 1
        elif a == 5:
            countU2R += 1
accM = (round(countM/(len(featsTest)), 3)) * 100
normaccM = (round(countNorm / len(normTest), 3)) * 100
DoSaccM = (round(countDoS / len(DoSTest), 3)) * 100
ProbeaccM = (round(countProbe / len(probeTest), 3)) * 100
R2LaccM = (round(countR2L / len(R2LTest), 3)) * 100
U2RaccM = (round(countU2R / len(U2RTest), 3)) * 100
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0
print("\n\tWith NSL-KDD train and test data using MLP Classifier\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest, lblsPredM),"\n\n")
print(m.classification_report(lblsTest, lblsPredM))
print("\nAccuracy = ", accM, "%\n\nAccuracy of normal = ", normaccM, "%\nAccuracy of DoS = ", DoSaccM, "%\nAccuracy of Probe = ", ProbeaccM, "%\nAccuracy of R2L = ", R2LaccM, "%\nAccuracy of U2R = ", U2RaccM, "%\n")
print("-------------------------------------------------------\n")

# model21M.fit(featsTrain, lblsTrain)
# lblsPred21M = model21M.predict(featsTest21)
# for a,b in zip(lblsTest21, lblsPred21M):
#     if a == b:
#         count21M += 1
#         if a == 1:
#             countNorm += 1
#         elif a == 2:
#             countDoS += 1
#         elif a == 3:
#             countProbe += 1
#         elif a == 4:
#             countR2L += 1
#         elif a == 5:
#             countU2R += 1
# acc21M = (round(count21M/(len(featsTest21)), 3)) * 100
# normacc21M = (round(countNorm / len(normTest21), 3)) * 100
# DoSacc21M = (round(countDoS / len(DoSTest21), 3)) * 100
# Probeacc21M = (round(countProbe / len(probeTest21), 3)) * 100
# R2Lacc21M = (round(countR2L / len(R2LTest21), 3)) * 100
# U2Racc21M = (round(countU2R / len(U2RTest21), 3)) * 100
# countNorm = 0
# countDoS = 0
# countProbe = 0
# countR2L = 0
# countU2R = 0
# print("\tWith NSL-KDD test data without difficulty level 21 using MLP Classifier\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest21, lblsPred21M),"\n\n")
# print(m.classification_report(lblsTest21, lblsPred21M))
# print("\nAccuracy = ", acc21M, "%\n\nAccuracy of normal = ", normacc21M, "%\nAccuracy of DoS = ", DoSacc21M, "%\nAccuracy of Probe = ", Probeacc21M, "%\nAccuracy of R2L = ", R2Lacc21M, "%\nAccuracy of U2R = ", U2Racc21M, "%\n")
# print("-------------------------------------------------------\n")


#End of MLP Model


#Start of SVM Model


modelS = s.SVC(gamma = "auto", cache_size = 7000)
model21S = s.SVC(gamma = "auto", cache_size = 7000)
modelS.fit(featsTrain, lblsTrain)
lblsPredS = modelS.predict(featsTest)
for a,b in zip(lblsTest, lblsPredS):
    if a == b:
        countS += 1
        if a == 1:
            countNorm += 1
        elif a == 2:
            countDoS += 1
        elif a == 3:
            countProbe += 1
        elif a == 4:
            countR2L += 1
        elif a == 5:
            countU2R += 1
accS = (round(countS/(len(featsTest)), 3)) * 100
normaccS = (round(countNorm / len(normTest), 3)) * 100
DoSaccS = (round(countDoS / len(DoSTest), 3)) * 100
ProbeaccS = (round(countProbe / len(probeTest), 3)) * 100
R2LaccS = (round(countR2L / len(R2LTest), 3)) * 100
U2RaccS = (round(countU2R / len(U2RTest), 3)) * 100
countNorm = 0
countDoS = 0
countProbe = 0
countR2L = 0
countU2R = 0
print("\n\tWith NSL-KDD train and test data using SVM\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest, lblsPredS),"\n\n")
print(m.classification_report(lblsTest, lblsPredS))
print("\nAccuracy = ", accS, "%\n\nAccuracy of normal = ", normaccS, "%\nAccuracy of DoS = ", DoSaccS, "%\nAccuracy of Probe = ", ProbeaccS, "%\nAccuracy of R2L = ", R2LaccS, "%\nAccuracy of U2R = ", U2RaccS, "%\n")
print("-------------------------------------------------------\n")

# model21S.fit(featsTrain, lblsTrain)
# lblsPred21S = model21S.predict(featsTest21)
# for a,b in zip(lblsTest21, lblsPred21S):
#     if a == b:
#         count21S += 1
#         if a == 1:
#             countNorm += 1
#         elif a == 2:
#             countDoS += 1
#         elif a == 3:
#             countProbe += 1
#         elif a == 4:
#             countR2L += 1
#         elif a == 5:
#             countU2R += 1
# acc21S = (round(count21S/(len(featsTest21)), 3)) * 100
# normacc21S = (round(countNorm / len(normTest21), 3)) * 100
# DoSacc21S = (round(countDoS / len(DoSTest21), 3)) * 100
# Probeacc21S = (round(countProbe / len(probeTest21), 3)) * 100
# R2Lacc21S = (round(countR2L / len(R2LTest21), 3)) * 100
# U2Racc21S = (round(countU2R / len(U2RTest21), 3)) * 100
# countNorm = 0
# countDoS = 0
# countProbe = 0
# countR2L = 0
# countU2R = 0
# print("\tWith NSL-KDD test data without difficulty level 21 using SVM\n\nConfusion Matrix: \n", m.confusion_matrix(lblsTest21, lblsPred21S),"\n\n")
# print(m.classification_report(lblsTest21, lblsPred21S))
# print("\nAccuracy = ", acc21S, "%\n\nAccuracy of normal = ", normacc21S, "%\nAccuracy of DoS = ", DoSacc21S, "%\nAccuracy of Probe = ", Probeacc21S, "%\nAccuracy of R2L = ", R2Lacc21S, "%\nAccuracy of U2R = ", U2Racc21S, "%\n")
# print("-------------------------------------------------------\n")


#End of SVM Model


#Prediction Testing


choice = str(input("\nDo you want to test specific input (y or n): "))
    
if(choice.lower()=="n"):
    print("Exiting..")

elif(choice.lower()=='y'):
    print("\n1. TCP      2. ICMP\n3. UDP")
    a = int(input("Enter Protocol Type: "))
    if a <= 0 or a > 11:
        while a <= 0 or a > 3:
            print("\nWrong Input!... Try Again\n")
            a = int(input("\nEnter choice for status flag: "))
            if a > 0 and a <= 3:
                break
    print("\n1. SF      2. REJ\n3. RSTR      4. RSTO\n5. S0      6. S1\n7.S2      8. S3\n9. OTH      10. SH\n11. RSTOS0")
    b = int(input("\nEnter choice for status flag: "))
    if b <= 0 or b > 11:
        while b <= 0 or b > 11:
            print("\nWrong Input!... Try Again\n")
            b = int(input("\nEnter choice for status flag: "))
            if b > 0 and b <= 11:
                break        
    c = int(input("\nEnter Number of Source Bytes sent (int): "))
    d = int(input("\nEnter num_access_files (int): "))
    e = int(input("\nEnter 1 if the login is guest or 0 if it is not: "))
    if e < 0 or e > 1:
        while e < 0 or e > 1:
            print("\nWrong Input!... Try Again\n")
            e = int(input("Enter 1 if login is guest or 0 if it is not: "))
            if e >= 0 and e <= 1:
                break
    f = int(input("\nEnter srv count (int): "))
    g = float(input("\nEnter diff srv rate (in decimal form) : "))
    h = float(input("\nEnter srv diff host rate (in decimal form) : "))
    i = float(input("\nEnter dst host same source port rate (in decimal form) : "))
    j = float(input("\nEnter dst host srv diff host rate (in decimal form) : "))
    k = float(input("\nEnter dst host srv serror rate (in decimal form) : "))
    m = float(input("Enter dst host srv rerror rate (in decimal form) : "))
    
    modelRDiff = es.RandomForestClassifier(n_estimators = 100, criterion = "entropy")
    modelDDiff = tr.DecisionTreeClassifier(criterion = "entropy", max_depth = 5)
    modelKDiff = n.KNeighborsClassifier(n_neighbors = 100)
    modelMDiff = nn.MLPClassifier()
    modelSDiff = s.SVC(gamma="auto", cache_size = 7000)

    modelRDiff.fit(featsTrainD, lblsTrainD)
    modelDDiff.fit(featsTrainD, lblsTrainD)
    modelKDiff.fit(featsTrainD, lblsTrainD)
    modelMDiff.fit(featsTrainD, lblsTrainD)
    modelSDiff.fit(featsTrainD, lblsTrainD)

    arrDiff = np.array([a, b, c, d, e, f, g, h, i, j, k, m])
    lR = int(modelRDiff.predict(arrDiff.reshape(1, -1)))
    lD = int(modelDDiff.predict(arrDiff.reshape(1, -1)))
    lK = int(modelKDiff.predict(arrDiff.reshape(1, -1)))
    lM = int(modelMDiff.predict(arrDiff.reshape(1, -1)))
    lS = int(modelSDiff.predict(arrDiff.reshape(1, -1)))
    
    arrR = np.array([a, b, c, d, e, f, g, h, i, j, k, lR, m])
    arrD = np.array([a, b, c, d, e, f, g, h, i, j, k, lD, m])
    arrK = np.array([a, b, c, d, e, f, g, h, i, j, k, lK, m])
    arrM = np.array([a, b, c, d, e, f, g, h, i, j, k, lM, m])
    arrS = np.array([a, b, c, d, e, f, g, h, i, j, k, lS, m])
    
    pPredR = modelR.predict(arrR.reshape(1, -1))
    print("\n\nUsing Random Forest Difficulty Level Prediction: ", lR, "\n")
    
    if(pPredR == 1):
        t.sleep(1)
        print("\nUsing Random Forest: Attack detection predicts normal connection\n\n")
    elif(pPredR == 2):
        t.sleep(1)
        print("\nUsing Random Forest: Attack detection predicts possible DoS attack!\n\n")
    elif(pPredR == 3):
        t.sleep(1)
        print("\nUsing Random Forest: Attack detection predicts possible Probe attack!\n\n")
    elif(pPredR == 4):
        t.sleep(1)
        print("\nUsing Random Forest: Attack detection predicts R2L attack!\n\n")
    elif(pPredR == 5):
        t.sleep(1)
        print("\nUsing Random Forest: Attack detection predicts U2R attack!\n\n")
    
    pPredD = modelD.predict(arrD.reshape(1, -1))
    print("Using Decision Tree Difficulty Level Prediction: ", lD, "\n")
    
    if(pPredD == 1):
        t.sleep(1)
        print("\nUsing Decision Tree: Attack detection predicts normal connection\n\n")
    elif(pPredD == 2):
        t.sleep(1)
        print("\nUsing Decision Tree: Attack detection predicts possible DoS attack!\n\n")
    elif(pPredD == 3):
        t.sleep(1)
        print("\nUsing Decision Tree: Attack detection predicts possible Probe attack!\n\n")
    elif(pPredD == 4):
        t.sleep(1)
        print("\nUsing Decision Tree: Attack detection predicts R2L attack!\n\n")
    elif(pPredD == 5):
        t.sleep(1)
        print("\nUsing Decision Tree: Attack detection predicts U2R attack!\n\n")

    pPredK = modelK.predict(arrK.reshape(1, -1))
    print("Using K-Neighbors Difficulty Level Prediction: ", lK, "\n")
    
    if(pPredK == 1):
        t.sleep(1)
        print("\nUsing K-Neighbors: Attack detection predicts normal connection\n\n")
    elif(pPredK == 2):
        t.sleep(1)
        print("\nUsing K-Neighbors: Attack detection predicts possible DoS attack!\n\n")
    elif(pPredK == 3):
        t.sleep(1)
        print("\nUsing K-Neighbors: Attack detection predicts possible Probe attack!\n\n")
    elif(pPredK == 4):
        t.sleep(1)
        print("\nUsing K-Neighbors: Attack detection predicts R2L attack!\n\n")
    elif(pPredK == 5):
        t.sleep(1)
        print("\nUsing K-Neighbors: Attack detection predicts U2R attack!\n\n")

    pPredM = modelM.predict(arrM.reshape(1, -1))
    print("Using MLP Clssifier Difficulty Level Prediction: ", lM, "\n")

    if(pPredM == 1):
        t.sleep(1)
        print("\nUsing MLP Classifier: Attack detection predicts normal connection\n")
    elif(pPredM == 2):
        t.sleep(1)
        print("\nUsing MLP Classifier: Attack detection predicts possible DoS attack!\n")
    elif(pPredM == 3):
        t.sleep(1)
        print("\nUsing MLP Classifier: Attack detection predicts possible Probe attack!\n")
    elif(pPredM == 4):
        t.sleep(1)
        print("\nUsing MLP Classifier: Attack detection predicts R2L attack!\n")
    elif(pPredM == 5):
        t.sleep(1)
        print("\nUsing MLP Classifier: Attack detection predicts U2R attack!\n")
        
    pPredS = modelS.predict(arrS.reshape(1, -1))
    print("Using SVM Difficulty Level Prediction: ", lS, "\n")

    if(pPredS == 1):
        t.sleep(1)
        print("\nUsing SVM: Attack detection predicts normal connection\n")
    elif(pPredS == 2):
        t.sleep(1)
        print("\nUsing SVM: Attack detection predicts possible DoS attack!\n")
    elif(pPredS == 3):
        t.sleep(1)
        print("\nUsing SVM: Attack detection predicts possible Probe attack!\n")
    elif(pPredS == 4):
        t.sleep(1)
        print("\nUsing SVM: Attack detection predicts R2L attack!\n")
    elif(pPredS == 5):
        t.sleep(1)
        print("\nUsing SVM: Attack detection predicts U2R attack!\n")


#End of Prediction Testing