# Wine-Quality-Prediction  

![image](https://user-images.githubusercontent.com/68503114/132919974-ecbeac72-ec16-402e-ba43-148049573917.png)


Predicting the quality of wine on a scale of 0–10 when given a set of features as inputs.  

## About  

[This project](https://github.com/KhushiiKumar/Wine-Quality-Prediction/blob/main/Wine%20Quality%20Prediction.ipynb) is a case study for modelling taste preferences based on the [analytical dataset](https://github.com/KhushiiKumar/Wine-Quality-Prediction/blob/main/Wine-Quality-Dataset.csv) that we have from physiochemical tests of wines. Such a model can be used not only by the certification bodies but also by the wine producers to improve quality based on the physiochemical properties and by the consumers to predict the quality of wines.

Input variables:  

- **fixed acidity:** No. of grams of tartaric acid per dm3
- **volatile acidity:**  No. of grams of acetic acid per dm3 of wine
- **citric acid:** No. of grams of citric acid per dm3 of wine
- **residual sugar:** Remaining sugar after fermentation stops
- **chlorides:** No. of grams of sodium chloride per dm3 of wine
- **free sulphur dioxide:** No. of grams of free sulphites per dm3 of wine
- **total sulphur dioxide:** No. of grams of total sulphite (free sulphite+ bound)
- **density:** Density in gram per cm3 
- **pH:** To measure ripeness 
- **sulphates:** No. of grams of potassium sulphate per dm3 of wine
- **alcohol:** Volume of alcohol in %
- **quality (target):** 1-10 value

### Classification criteria
The scores for quality(x) are set between 0-10 as follows:  
**Bad:** 0<=x<=4   
**Medium:** 5,6  
**Good:** 7<=x<=10 

## Conclusion

<img src="https://github.com/KhushiiKumar/Wine-Quality-Prediction/blob/main/Visualizing_results.png" width="750" height="300">

For making automated decisions on model selection we need to quantify the performance of our model and give it a score. Therefore, accuracy rate of the model is checked on each prediction. Here, **Random Forest** has the maximum accuracy rate.

## Modification
- Add your desktop directory in the code where I have used dataset path, like: C:/Users/HP/Desktop/dataset.csv (this is my desktop directory). 
- If you don't know your desktop directory then just open terminal or command prompt and paste the below code. 

    `%userprofile%\Desktop`  

## Usage  
python3 Wine-Quality-Prediction.py

