import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,KFold
from sklearn.svm import SVC


def create_proto(file0_loc, file1_loc):
    df0 = pd.read_csv(file0_loc)
    df1 = pd.read_csv(file1_loc)
    cols = []
    for n in range(0,120):
        col = np.asarray(df0.iloc[:, n])
        cols.append(col)
    for n in range(0,120):
        col = np.asarray(df1.iloc[:, n])
        cols.append(col)
    return cols

c_vals = []
acc_scores_tuning = []

X = np.stack(create_proto('data/dataset1.csv', 'data/dataset2.csv'))
y= np.stack(np.append(np.array([0]*120),np.array([1]*120)))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/6)

cv_outer = KFold(n_splits=6)

for train_idx,val_idx in cv_outer.split(X_train,y_train):
    train_data,val_data = X_train[train_idx],X_train[val_idx]
    train_target,val_target = y_train[train_idx],y_train[val_idx]

    #show_chanWeights(train_data[0])

    model = SVC(kernel='linear',probability=True)
    cv_inner = StratifiedKFold(n_splits=5)
    params = {'class_weight': [None,'balanced'],'C': [10 ** x for x in range(-9,6)]}
    gd_search = GridSearchCV(model,params,scoring='roc_auc',n_jobs=-1,cv=cv_inner).fit(train_data,train_target)
    best_model = gd_search.best_estimator_

    classifier = best_model.fit(train_data,train_target)
    y_pred_prob = classifier.predict_proba(val_data)[:,1]
    auc = metrics.roc_auc_score(val_target,y_pred_prob)

    print("Val Acc:",auc,"Best GS Acc:",gd_search.best_score_,"Best Params:",gd_search.best_params_)

    c_vals.append(gd_search.best_params_)
    acc_scores_tuning.append(gd_search.best_score_)

# Training final model
highest_acc_val_index = acc_scores_tuning.index(max(acc_scores_tuning))
best_c_val = c_vals[highest_acc_val_index]
print(best_c_val)
model = SVC(kernel='linear',probability=True,C=best_c_val['C'],class_weight=best_c_val['class_weight']).fit(X_train,y_train)
y_pred_prob = model.predict_proba(X_test)[:,1]
weights_to_brain = [abs(x) for x in np.asarray(model.coef_)][0]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
roc_score = metrics.roc_auc_score(y_test,y_pred_prob)
print("AUC",roc_score)
plt.plot(fpr,tpr,label="optimal C value: "+str(best_c_val['C'])+" and auc="+str(roc_score))
plt.legend(loc=4)
plt.title("ROC Curve for Optimal Classifier")
plt.show()



def gen_svm_boundary(X_new, Y_new, clf):
    # figure number
    fignum = 1
    clf.fit(X_new, Y_new)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=40,
                facecolors='none', zorder=10, edgecolors='k')
    plt.title('SVM Decision Boundary with linear kernel')
    plt.scatter(X_new[:, 0], X_new[:, 1], c=Y_new, zorder=10 ,marker="x", cmap='RdYlGn',
                edgecolors='k')

    plt.axis('tight')
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000

    XX, YY = np.mgrid[x_min:x_max:10j, y_min:y_max:10j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('SVM Decision Boundary')
    fignum = fignum + 1
    plt.show()


def gen_svm_3d_boundary(X,Y, model):
    # make it binary classification problem
    X = X[np.logical_or(Y == 0,Y == 1)]
    Y = Y[np.logical_or(Y == 0,Y == 1)]
    clf = model.fit(X,Y)
    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x,y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    tmp = np.linspace(-5,5,30)
    x,y = np.meshgrid(tmp,tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot3D(X[Y == 0,0],X[Y == 0,1],X[Y == 0,2],'ob')
    ax.plot3D(X[Y == 1,0],X[Y == 1,1],X[Y == 1,2],'sr')
    ax.plot_surface(x,y,z(x,y))
    ax.view_init(30,30)
    ax.set_title("3D projection of first three features")
    plt.show()


gen_svm_boundary(X_train[:, :2],y_train, model)
gen_svm_3d_boundary(X_train[:, :3],y_train, model)