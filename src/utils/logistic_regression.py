"""
Logistic regression on Sam's featurization.
"""

import scipy.io
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

data = scipy.io.loadmat('../../data/all_molecules_sam_feats.mat')
x, y = data['X'], data['Y'].ravel()

# Should match the split we've been using
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.25, random_state=int(sum(map(ord, 'alvin'))))

scipy.io.savemat('../../data/train_molecules_sam_feats.mat', {'X': X_train, 'Y': y_train})
scipy.io.savemat('../../data/test_molecules_sam_feats.mat', {'X': X_test, 'Y': y_test})

lr = sklearn.linear_model.LogisticRegression(verbose=1)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(sklearn.metrics.accuracy_score(y_pred, y_test))
print(sklearn.metrics.confusion_matrix(y_pred, y_test))
