import os
import numpy as np
import cv2
import glob
import sklearn.svm as svm
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from numpy import linalg as la
from sklearn.decomposition import PCA
from skimage import feature as ft

def siftorsurf(img):
    #选择使用SIFT还是SURF
    sift = cv2.xfeatures2d.SURF_create()
    #sift = cv2.xfeatures2d.SIFT_create()
    #计算图片的特征点和特征点描述
    keypoints, features = sift.detectAndCompute(img, None)
    return features


#计算图片的特征向量
def get_tz(path,centers):
    jzdata = np.float32([]).reshape(0, 50)
    labels = np.float32([])
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img=cv2.imread(im)
            img_f = siftorsurf(img)
            featVec = np.zeros((1, 50))
            for i in range(0, img_f.shape[0]):
                fi = img_f[i]
                diffMat = np.tile(fi, (50, 1)) - centers
                sqSum = (diffMat ** 2).sum(axis=1)
                dist = sqSum ** 0.5
                sortedIndices = dist.argsort()
                idx = sortedIndices[0]
                featVec[0][idx] += 1
            data_vec = np.append(jzdata,featVec,axis=0)
            labels = np.append(labels,idx)
    return jzdata,labels


#计算训练集图片特征向量
def hog_vec(path):
    data_vec1 = []
    labels = np.float32([])
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    #wsl = 0
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            img=cv2.imread(im, cv2.IMREAD_GRAYSCALE )
            # 提取hog特征
            features = ft.hog(img,feature_vector=True,visualize=False)

            features = features.tolist()
            data_vec1.append(features)
            labels = np.append(labels,idx)
            #wsl += 1
            #print(wsl)
    data_vec=np.array(data_vec1)
    return data_vec,labels



if __name__ == "__main__":
    train_path = './15/train2++'
    test_path = './15/test2++'

    # 建立大小自定义的词袋
    cate = [train_path + '/' + x for x in os.listdir(train_path) if os.path.isdir(train_path + '/' + x)]
    features = np.float32([]).reshape(0, 64)
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = cv2.imread(im)
            img_f = siftorsurf(img)
            features = np.append(features, img_f, axis=0)
    # 设置词典的大小
    wordsize = 50
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(features, wordsize, None, criteria, 20, flags)
    #构建训练集特征向量
    data_vec1,labels = get_tz(train_path,centers)
    data_vec2,labels = hog_vec(train_path)

    #直接特征融合
    data_vec = np.hstack((data_vec1, data_vec2))
    print(1)
    print(data_vec.shape)


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
    # tdata1,tlabels = get_tz(test_path,centers)
    # tdata2,tlabels = hog_vec(test_path)
    # tdata = np.hstack((tdata1, tdata2))
    # tres = knn.predict(tdata)
    # num2 = tdata.shape[0]
    # acc2 = 0
    # for i in range(num2):
    #     if tlabels[i] == tres[i]:
    #         acc2 = acc2+1
    # print('acc：' + str(acc2) + '/' + str(num2) + '=' + str(acc2 / num2))


    # #贝叶斯分类器
    # bayes = GaussianNB()
    # bayes.fit(data_vec,labels)
    # #计算训练集的正确率
    # res = bayes.predict(data_vec)
    # num1 = data_vec.shape[0]
    # acc1 = 0
    # for i in range(num1):
    #     if labels[i] == res[i]:
    #         acc1 = acc1 + 1
    # print('acc: ' + str(acc1) + '/' + str(num1) + '=' + str(acc1 / num1))
    # #计算测试集的正确率
    # tdata1,tlabels = get_tz(test_path,centers)
    # tdata2,tlabels = hog_vec(test_path)
    # tdata = np.hstack((tdata1, tdata2))
    # #tdata2 = PCA(tdata,0.99)
    # tres = bayes.predict(tdata)
    # num2 = tdata.shape[0]
    # acc2 = 0
    # for i in range(num2):
    #     if tlabels[i] == tres[i]:
    #         acc2 = acc2+1
    # print('acc：' + str(acc2) + '/' + str(num2) + '=' + str(acc2 / num2))