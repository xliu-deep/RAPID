# Heterogeneous-Reaction-Energy-Prediction

The heterogeneous transformation of organic pollutants into more toxic chemicals poses substantial risks to humans and the environment, e.g., the formation of environmental persistent free radicals (EPFRs). The activation energy a system is an important measure of whether the reactions may occur. In this work, we aim to develop an efficient machine learning (ML) model for predicting activation energy for the formation of EPFRs via the transformation reactions of phenolic organics over montmorillonite surfaces using easily available properties of the metal cations and organic precursors, such as redox potential, softness, hardness, etc Six supervised ML algorithms including the decision tree (DT), eXtreme Gradient Boosting (XGBoost), random forest (RF), support vector machine (SVM), multilayer perceptron (MLP), and linear regression (LR), were adopted to choose the optimal model. 

---
Machine learning workflow for predicting reaction activation energy.
![image](https://user-images.githubusercontent.com/1555415/185529955-b018cdf5-1774-49f3-a578-19fa2d123049.png)


#### data folder
  > the data used in the study
#### result folder
  > all the prediction result files
#### modeling files
- modeling.py
  > hyperparameter tunning and model predicting
- plot.py 
  > drawing figures showed in our research article
