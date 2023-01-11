# Heterogeneous-Reaction-Energy-Prediction

Heterogeneous transformation of organic pollutants into more toxic chemicals poses substantial health risks to humans. Activation energy is an important indicator that help us to understand transformation efficacy of environmental interfacial reactions. However, the determination of activation energies for large numbers of pollutants using either the experimental or high-accuracy theoretical methods is expensive and time-consuming. Alternatively, the machine learning (ML) method show the strength in predictive performance. In this study, using the formation of a typical montmorillonite-bound phenoxy radical as an example, a generalized ML framework RAPID was proposed for activation energy prediction of environmental interfacial reactions. Accordingly, an explainable ML model was developed to predict the activation energy via easily accessible properties of the cations and organics. The model developed by decision tree (DT) performed best with the lowest root-mean-squared error (RMSE=0.22) and the highest coefficient of determination values (R2 score=0.928), the underlying logic of which was well understood by combining model visualization and SHapley Additive exPlanations (SHAP) analysis. The performance and interpretability of the established model suggest that activation energies can be predicted by the well-designed ML strategy, and this would allow us to predict more heterogeneous transformation reactions in the environmental field.  

---
Machine learning workflow for predicting reaction activation energy.
![image](https://user-images.githubusercontent.com/1555415/211749631-8e0e1780-14dc-4784-9f4f-1f47c3287e9e.png)


#### data folder
  > training set and test set
#### result folder
  > all the prediction result files
#### modeling files
- modeling.py
  > hyperparameter tunning and model predicting
- plot.py 
  > drawing figures showed in our research article
