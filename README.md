# ECG-based-Biometric-Identification

## Project Overview
This project was proposed following a consideration raised by Lugovaya [1] about the potential use of the electrocardiogram (ECG) as a biometric human identification method.
The project analyzes feature selection and parameter extraction from an ECG database, and proposed a model to identify the subjects based on these features, delivering insights on the impact of feature engineering and the potential use of ECG as a biometric human identification signal. 
The ECG database, created and contributed by Lugovaya [1] and available at Physionet [2](https://physionet.org/content/ecgiddb/1.0.0/), consists of 310 ECG recordings from 90 volunteers (44 men and 46 women aged from 13 to 75 years), with a duration of 20 seconds, digitized at 500 Hz with 12-bit resolution over a nominal ±10 mV range. Additionally, complementary files contains information about age, gender and recording date (.hea file). The number of records for each person varies from 2 (collected during one day) to 20 (collected periodically over 6 months).

## Background
The ECG is a record of electrical currents generated by the beating heart and depends on the anatomic features of the human heart and body. Thus, the ECG has the potential to be a distinctive human characteristic. In machine learning, a supervised classification algorithm can be proposed to accomplish the biometric identification of a subject based on the ECG signal. But, how can the ECG signal be used as input for the proposed model? Applying feature engineering, analizing the obtained results, and then compare the selected parameters and the raw signal as input are an integrative part for the optimization of the model and the use of the ECG signal as a biometric human identification method.

## Project Highlight
The development of this project involved:
* Preparation of data, including parameter extraction, feature engineering and selection 
* Perform exploratory data analysis (EDA) on biomedical data to inform model training and explain model performance
* Propose a supervised classification model (SVM linear) for biometric human identification based on basic demographic and ECG features
* Train and test an SVM linear model selecting different features
* Assess and compare the performance of each trained model through proposed metrics

## The Project at a glance

Part 1: [Data preparation](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Part1_DataPreparation.ipynb)

Part 2: Exporatory Data Analysis [(EDA)](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Part2_EDA.ipynb)
![EDA_heatmap](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Images/HeatMap.png)

Part 3: Data Modeling - ECG classification [algorithm](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Part3_DataModeling.ipynb)

![Feature_weights](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Images/Feature_weigth.png)
![Feature_impact](https://github.com/franciscoj-londonoh/ECG-based-Biometric-Identification/blob/main/Images/Feature_TrainImpact.png)


### References
[1] Lugovaya T.S. Biometric human identification based on electrocardiogram. [Master's thesis] Faculty of Computing Technologies and Informatics, Electrotechnical University "LETI", Saint-Petersburg, Russian Federation; June 2005.

[2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
