import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# 数据读入
def load_data(path):
    data = pd.read_csv(path)
    X = data.values[:,1:]
    y = data[['label']].values
    y = y.flatten()

    # 划分训练集，测试集
    X = X / 255
    return X, y

# 定义对PCA参数分析的函数
def n_components_analysis(n, X_train, y_train, X_test, y_test):

    start = time.time()
    # PCA降维实现
    pca = PCA(n_components=n, svd_solver='full')
    print("PCA特征降维所选参数:{}".format(n))
    pca.fit(X_train)
    x_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)

    # 放入支持向量机进行训练
    svc = svm.SVC()
    svc.fit(x_train_pca, y_train)
    accuracy = svc.score(x_test_pca, y_test)

    end = time.time()
    print("准确率：{}, 花费时间：{}, 保留的信息量的比例:{}, PCA特征降维后训练集保留的特征数:{}"
          .format(accuracy, int(end-start), pca.explained_variance_ratio_.sum(), x_train_pca.shape[1]))
    return accuracy

# 定义PCA降维参数选择可视化的函数
def n_components_choice(X_train, y_train, X_test, y_test):
    # 记录参数选择不同对应的accuracy
    n_s = np.linspace(0.50, 0.90, num=10)  # n从0.50到0.90，共取10个
    accuracy = [] # 记录每一次准确率
    for n in n_s:
        result = n_components_analysis(n, X_train, y_train, X_test, y_test)
        accuracy.append(result)
    # 对选择不同的参数可视化
    plt.xlabel('choice of different n_components', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.ylim(0.95, 1)
    plt.plot(n_s, np.array(accuracy), 'r', color='grey', marker='o', markersize=8, markerfacecolor='red')
    plt.show()
    return accuracy,n_s

# 对数据进行PCA降维
def feature_decomposition(X_train,X_test,a):

    pca = PCA(n_components=a, svd_solver='full', whiten=True).fit(X_train)
    train_x = pca.fit_transform(X_train)
    test_x = pca.transform(X_test)
    print("目前训练集特征数:{}".format(train_x.shape[1]))
    return train_x, test_x

def comparison_kernel(train_x, test_x, y_train, y_test):

    kernel = ['linear', 'poly', 'sigmoid', 'rbf']
    ACC = []
    # 对不同的核函数进行选择
    for i in kernel:
        print('选用的核函数为{}时'.format(i))
        start = time.time()
        svc = svm.SVC(kernel = i, C = 10)
        svc.fit(train_x, y_train)
        end = time.time()
        ACC.append(svc.score(test_x, y_test))
        print('accuracy:', svc.score(test_x, y_test))
        print('所用的时间：{}s'.format(int(end-start)))
    # 对选择的不同核函数进行可视化
    plt.xlabel('choice of different kernel', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.plot(kernel, np.array(ACC),'-p', color='grey', marker = 'o', markersize=8,markerfacecolor='red')
    plt.ylim(0.80, 1)
    plt.show()
    return ACC

# 对数据集可视化
def number_plot(X,y):

    X1 = np.zeros([len(X), 28, 28])#将原始数据恢复为原始维度
    for i in range(0, len(X)):
        X1[i] = X[i].reshape(28, 28)
    images_and_labels = list(zip(X1, y))

    for index, (image, label) in enumerate(images_and_labels[:8]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        # image是要绘制的图像或者数组，cmp是颜色图谱，
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# 可视化ROC曲线
def ROC_plot(n_classes, y_test, y_score):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lw = 1
    colors = ['blue', 'red', 'green', 'black', 'yellow', 'orange', 'grey', 'brown', 'cyan', 'darkblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.3f})'
                                                          ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()
'''
def look_plot(n):
    num = X[n].reshape(28,28)
    plt.imshow(num)
    plt.axis("off")
    plt.show()
'''
if __name__ == '__main__':

    # 读取数据并划分测试集、训练集
    X, y =load_data('E:\\python\\train.csv')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=111)
    print("X train shape: {}, X test shape: {} ".format(X_train.shape, X_test.shape))  # 训练集29400个数据,测试集12600个数据

    # 对手写数字可视化
    number_plot(X, y)
    # look_plot(3)查看第四张图片

    # 传递多个参数，寻找合理的n_components
    accuracy, n_s= n_components_choice(X_train, y_train, X_test, y_test)
    choice_num = round(n_s[accuracy.index(max(accuracy))],2)

    # 使用PCA对数据进行降维
    train_x,test_x = feature_decomposition(X_train,X_test,choice_num)

    # 使用支持向量机对模型进行训练并比较了不同核函数选择对分类器预测性能的影响
    ACC=comparison_kernel(train_x, test_x, y_train, y_test)

    # 对最终模型画ROC曲线
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_test.shape[1]
    svc = svm.SVC(kernel='rbf', C=10, probability=True)
    svc.fit(train_x, y_train)
    y_score = svc.predict_proba(test_x)
    ROC_plot(n_classes, y_test, y_score)








