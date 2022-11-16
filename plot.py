import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def plot_regression():
	name = 'Decision Tree'
	train_pred = np.loadtxt('result/%s_y_train_pred.txt'%name)
	test_pred = np.loadtxt('result/%s_y_test_pred.txt'%name)
	predictied = np.concatenate((train_pred, test_pred), axis=None)
	
	y_train_data = pd.read_csv('data/y_train.csv',index_col=None,header=0)
	y_train = y_train_data.values.ravel()
	y_test_data = pd.read_csv('data/y_test.csv',index_col=None,header=0)
	y_test = y_test_data.values.ravel()
	Truedata = np.concatenate((y_train, y_test), axis=None)
	
	data = pd.DataFrame(list(zip(Truedata,predictied)),columns=['Truedata','Predictied'])
	data['Type'] = np.array(train_pred.shape[0]*['Training set']+test_pred.shape[0]*['Test set']).T
	
	
	g = sns.jointplot(x=y_train, y=train_pred,
					kind="reg", truncate=False,color="m", height=6)
	g.set_axis_labels('Barrier', 'Predicted', fontsize=13)
	g.ax_joint.text(2.1,0.1,s=u"$R^2$ = {:.3f}".format(r2_score(y_train, train_pred)))
	g.figure.tight_layout()
	g.savefig('result/Traing_set_regression.png',dpi=800)
	
	
	h = sns.jointplot(x=y_test, y=test_pred,
	                  kind="reg", truncate=False, color="royalblue",height=6)
	h.set_axis_labels('Barrier', 'Predicted', fontsize=13)
	h.figure.tight_layout()
	h.ax_joint.text(2.1,0.15,s=u"$R^2$ = {:.3f}".format(r2_score(y_test, test_pred)))
	h.savefig('result/Test_set_regression.png',dpi=800)


def plot_box():
	sns.set_theme(style="ticks", palette=sns.color_palette("hls", 8))
	data = pd.read_csv('result/model_comparison_CV.csv',index_col=None,header=0)
	ax = sns.boxplot(x="models", y="R2", data=data, showfliers=False)
	ax = sns.swarmplot(x="models", y="R2", data=data)
	ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=15)
	# ax.set_yticks(np.arange(-0.5,1))
	# ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
	ax.set_xlabel('Machine learning models', fontsize=12)
	ax.set_ylabel(u"$R^2$", fontsize=12)
	sns.despine()
	ax.figure.tight_layout()
	plt.savefig('result/model_comparison_CV.png',dpi=800)
	plt.show()


def plot_bar():
	data = pd.read_csv('result/model_comparison_test.csv',index_col=None,header=0)
	sns.set_theme(style="ticks", palette=sns.color_palette("coolwarm", 4))
	fig, ax1 = plt.subplots()
	sns.barplot(x="models", y="data", data=data, hue="metric",ax=ax1)
	# Create a second y-axis with the scaled ticks
	ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=11,rotation=15)
	ax1.set_ylabel(u"$R^2$", fontsize=12)
	ax2 = ax1.twinx()
	ax2.set_ylabel('RMSE', fontsize=12)
	ax1.set_xlabel('Machine learning models', fontsize=13)
	ax1.figure.tight_layout()
	ax1.legend(loc=1, frameon=False)
	ax1.figure.tight_layout()
	plt.savefig('result/model_comparison_test.png',dpi=800)
	plt.show() # shows the plot. 
	
	
plot_regression()
plot_box()
plot_bar()