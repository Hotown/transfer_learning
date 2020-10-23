import numpy as np
import sklearn
import scipy
import scipy.io as sio
from sklearn.neighbors import KNeighborsClassifier


def kernal(type, X1, X2, gamma):
    '''

    :param type: linear or rbg
    :param X1:
    :param X2:
    :param gamma: the parm of rbg
    :return:
    '''

    K = None
    if not type:
        return X1

    elif type == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)

    elif type == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)

    return K


class TCA:
    def __init__(self, type, dim=30, lamb=1, gamma=1):
        '''

        :param dim: 降维后的维度
        :param lamb: 正则项系数
        '''
        self.type = type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        :param Xs: source domain
        :param Xt: target domain
        :return:
        '''
        X = np.hstack((Xs.T, Xt.T))
        X = X / np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e * e.T
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernal(self.type, X, None, self.gamma)
        # K= kernal('rbf', X, None, gamma=self.gamma)
        Sa = np.linalg.multi_dot([K, L, K.T]) + self.lamb * np.eye(n)
        Sb = np.linalg.multi_dot([K, H, K.T])

        w, v = scipy.linalg.eig(Sa, Sb)
        ind = np.argsort(w)
        Phi = v[:, ind[:self.dim]]
        Z = np.dot(Phi.T, K)
        Z = Z / np.linalg.norm(Z, axis=0)
        Xs_, Xt_ = Z[:, :ns].T, Z[:, ns:].T
        return Xs_, Xt_

    def train(self, Xs, Xt, Ys, Yt):
        '''

        :param Xs: Source Domain Instance
        :param Xt: Source Domain Label
        :param Ys:
        :param Yt:
        :return:
        '''
        Xs_, Xt_ = self.fit(Xs, Xt)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(Xs_, Ys.flatten())
        y_ = classifier.predict(Xt_)
        acc = sklearn.metrics.accuracy_score(Yt, y_)
        return acc, y_


if __name__ == '__main__':
    domains = ['Caltech10_SURF_L10.mat', 'amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat']
    for i in [2]:
        for j in [3]:
            if i != j:
                src, tar = '../data/surf/' + domains[i], '../data/surf/' + domains[j]
                src_domain, tar_domain = sio.loadmat(src), sio.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']
                tca = TCA(type='linear', dim=30, lamb=1, gamma=1)
                acc, y_ = tca.train(Xs, Xt, Ys, Yt)
                print('Acc:{:4f}'.format(acc))
