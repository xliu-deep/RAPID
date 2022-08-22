
# from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import warnings
from math import sqrt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


import seaborn as sns
import matplotlib.pyplot as plt


class BarrierRegression():
	def __init__(self):
		self.names = ["Nearest Neighbors", "SVM", "Gaussian Process",
					"Decision Tree", "Random Forest", "MLP","XGBoost","Linear Regression"]
		knn = [KNeighborsRegressor(),{'n_neighbors':[2,3,4,5]}]
		svm = [SVR(),{'C': [1, 10], 'kernel': ('linear', 'rbf')}]
		gp = [GaussianProcessRegressor(random_state=0),{"kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]}]
		dr = [DecisionTreeRegressor(random_state=0),{'splitter':['best', 'random'],
										'max_depth':[2,3,4,5]}]
		rf = [RandomForestRegressor(random_state=0),{'n_estimators': [50, 150, 250, 350],
										'max_features': ['sqrt', 0.25, 0.5],
										'min_samples_split': [2, 4, 6]}]
		mlp = [MLPRegressor(random_state=0),{'hidden_layer_sizes': [(10,30,10),(20,)],
								'activation': ['tanh', 'relu'],
								'solver': ['sgd', 'adam'],
								'alpha': [0.0001, 0.05],
								'max_iter':[2000],
								'learning_rate': ['constant','adaptive']}]
		xgboost = [XGBRegressor(eval_metric='mlogloss',random_state=0),{'learning_rate': [0.0001, 0.001, 0.01, 0.1],
									'n_estimators': [100, 300, 500],
									'max_depth': [3,5,7,9],
									'min_child_weight': [1,3,5]}]
		rg = [Ridge(random_state=0),{'alpha':[0.1,0.25,.5]}]

		self.regressors = [knn,svm,gp,dr,rf,mlp,xgboost,rg]


	def chooseparameter(self,X_train,y_train):
		warnings.filterwarnings("ignore")
		result = []

		for name, [regressor,parameters] in zip(self.names, self.regressors):

			clf = GridSearchCV(estimator=regressor,param_grid=parameters,
								scoring='r2',cv=KFold(n_splits=5, shuffle=True))
			clf.fit(X_train,y_train)
			pd.DataFrame.from_dict(clf.cv_results_).to_csv('%s_GridSearchCV.csv'%name)
			
			# Test set predict prob
			# print(clf.best_params_)
			# print(clf.best_score_)
			# print(clf.cv_results_)
			clf_best = regressor.set_params(**clf.best_params_)
			clf.fit(X_train,y_train)
			test_pred = clf.predict(X_test)
			train_pred = clf.predict(X_train)
			
			# save predicted label
			np.savetxt('%s_y_train_pred.txt'%name,train_pred)
			np.savetxt('%s_y_test_pred.txt'%name,test_pred)

			# evaluation metric
			test_r2 = r2_score(y_test, test_pred)
			train_r2 = r2_score(y_train, train_pred)
			test_rmse = sqrt(mean_squared_error(y_test, test_pred))
			train_rmse = sqrt(mean_squared_error(y_train, train_pred))

			result.append([name,clf.best_params_,clf.best_score_,train_r2,
					train_rmse,test_r2,test_rmse])

		# Save model comparision result
		result = np.array(result)
		print(result)
		df = pd.DataFrame(result,columns = ['name','best_params','best_cv_r2','train_r2',
					'train_rmse','test_r2','test_rmse'])
		df.to_csv('MultipleModels_comparision_r2.csv')


	def tSNE(df):
		X = df.values
		y = df.index
		print(X)
		print(y)
		X_embedded = manifold.TSNE(n_components=2, perplexity=20,init='pca', random_state=0).fit_transform(X)
		df = pd.DataFrame(X_embedded, columns = ['t-SNE_1','t-SNE_2'])
		df.to_csv('tSNE.csv',index=None)


X_train_data = pd.read_csv('data/X_train.csv',index_col=0,header=0)
X_train = X_train_data.values

X_test_data = pd.read_csv('data/X_test.csv',index_col=0,header=0)
X_test = X_test_data.values

y_train_data = pd.read_csv('data/y_train.csv',index_col=None,header=0)
y_train = y_train_data.values.ravel()

y_test_data = pd.read_csv('data/y_test.csv',index_col=None,header=0)
y_test = y_test_data.values.ravel()

BarrierRegression().chooseparameter(X_train,y_train)
# ClassifyThyCarcino().Shapley_importance(X_train,y_train,X_test)

