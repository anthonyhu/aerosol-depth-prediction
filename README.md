# Multiple Instance Regression: Aerosol optical depth prediction

Among the biggest challenges of current climate research is the impact of Aerosols on the Earth’s climate and our health. Recent satellite spectral images and measurements enable us to tackle the problem of predicting Aerosol Optical Depth (AOD). Several approaches are considered. Our final model uses a probabilistic approach: observations are assumed to be sampled from a distribution and each distribution summarised by a kernel mean embedding.  AOD is predicted for each instance with Kernel Ridge regression and the results combined through a Support Vector Regression.  This stacking model gave a leaderboard score of __0.7356__ (RMSE metric), the best performance among our peers on the test data provided.

## Presentation

The Multiple Instance Regression problem arises when the label of every data point is not known uniquely but is instead shared among grouping of instances denoted _bag_. The goal is to train a regression model that can accurately predict the label of an unlabelled bag of instances.

In our case, each bag is made of 100 satellite measurements of reflectances (the fraction of light that is reflected from the atmosphere and the surface hit) in a given geographic area. The label of the bag is the Aerosol Optical Depth directly measured by an instrument on the field. The data is available here: https://inclass.kaggle.com/c/aerosol-prediction-practical

## Pruning approach

The AOD in a geographic area does not vary much within a reasonable range (up to 100km) however the satellite measurements is greatly dependent on the surface and on the presence of clouds. More precisely, the radiance measured by the satellite is a combination of the light reflected by the surface and by the atmosphere. If the surface is dark, all the light will be absorbed and only the reflected light from the atmosphere will be measured by the satellite, meaning that the AOD can be calculated with a deterministic function.
However, not all the instances in the bag come from a dark surface and that is why in our first approach, in the "__Instances pruning with tensorflow__" notebook, we will try to get rid of those noisy instances (instances associated with light surfaces) in each bag by recursively pruning them.

## Distribution regression

Instead  of  fitting  a  model  to  the  instances,  the  idea  of  distribution  regression is  to  find  a regression on the underlying probability distributions the instances come from. This approached proved to give more satisfactory results and is detailed in the notebook: "__Distribution regression__".


