import os
import numpy as np
import cv2
import glob
import sklearn.svm as svm
import joblib
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from skimage import filters
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import itertools

#计算训练集图片特征向量
def get_tz(path):
    data_vec = np.float32([]).reshape(0, 256)#存放训练集图片的特征
    labels = np.float32([])
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img=cv2.imread(im)
            # 提取color特征
            hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
            # f0 = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
            # f1 = cv2.calcHist([img], [1], None, [256], [0.0, 255.0])
            # f2 = cv2.calcHist([img], [2], None, [256], [0.0, 255.0])
            #
            # plt.plot(range(256), f0, label='R')
            # plt.plot(range(256), f1, label='G')
            # plt.plot(range(256), f2, label='B')
            # plt.show()
            hhist = hist.reshape((1,256))
            #print(hhist.shape)
            # hist size:256
            data_vec = np.append(data_vec,hhist,axis=0)
            labels = np.append(labels,idx)

    return data_vec,labels

if __name__ == "__main__":
    train_path = './7/train1+'
    test_path = './7/test1+'
    data_vec,labels = get_tz(train_path)
    # data_vect, labelst = get_tz(test_path)
    # X = np.vstack((data_vec, data_vect))

    # num = data_vec.shape[0]
    # labelss = labels.reshape((num, 1))
    # data = np.hstack((data_vec, labelss))

    #SVM分类器
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(data_vec, labels)

    # 计算训练集的正确率
    data_vec, labels = get_tz(train_path)
    res = clf.predict(data_vec)
    num_test = data_vec.shape[0]
    # print(num_test)
    acc = 0
    for i in range(num_test):
        if labels[i] == res[i]:
            # print(i)
            # print(labels[i], res[i], sep='--')
            acc = acc + 1
        else:
            print(i)
            print(labels[i], res[i], sep='--')
    print('acc: ' + str(acc) + '/' + str(num_test) + '=' + str(acc / num_test))

    # 计算测试集的正确率
    data_vec, labels = get_tz(test_path)
    res = clf.predict(data_vec)
    num_test = data_vec.shape[0]
    # print(num_test)
    acc = 0
    for i in range(num_test):
        if labels[i] == res[i]:
            # print(i)
            # print(labels[i], res[i], sep='--')
            acc = acc + 1
        else:
            print(i)
            print(labels[i], res[i], sep='--')
    print('acc: ' + str(acc) + '/' + str(num_test) + '=' + str(acc / num_test))




    # #KNN分类器
    # knn = KNeighborsClassifier(n_neighbors=2)
    # knn.fit(data_vec,labels)
    # #计算训练集的正确率
    # res = knn.predict(data_vec)
    # num1 = data_vec.shape[0]
    # acc1 = 0
    # for i in range(num1):
    #     if labels[i] == res[i]:
    #         acc1 = acc1 + 1
    # print('acc: ' + str(acc1) + '/' + str(num1) + '=' + str(acc1 / num1))
    # #计算测试集的正确率
    # tdata,tlabels = get_tz(test_path)
    # tres = knn.predict(tdata)
    # num2 = tdata.shape[0]
    # acc2 = 0
    # for i in range(num2):
    #     if tlabels[i] == tres[i]:
    #         acc2 = acc2+1
    # print('acc：' + str(acc2) + '/' + str(num2) + '=' + str(acc2 / num2))





    # # 贝叶斯分类器
    # bayes = GaussianNB()
    # #bayes = MultinomialNB()
    # bayes.fit(data_vec, labels)
    # # 计算训练集的正确率
    # res = bayes.predict(data_vec)
    # print(res)
    # num1 = data_vec.shape[0]
    # acc1 = 0
    # for i in range(num1):
    #     if labels[i] == res[i]:
    #         acc1 = acc1 + 1
    # print('acc: ' + str(acc1) + '/' + str(num1) + '=' + str(acc1 / num1))
    # #计算测试集的正确率
    # tdata,tlabels = get_tz(test_path)
    # tres = bayes.predict(tdata)
    # print(tres)
    # num2 = tdata.shape[0]
    # acc2 = 0
    # for i in range(num2):
    #     if tlabels[i] == tres[i]:
    #         acc2 = acc2+1
    # print('acc：' + str(acc2) + '/' + str(num2) + '=' + str(acc2 / num2))


