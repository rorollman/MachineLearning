#  Predicting On-Time Flights
**Rowan Rollman** 

## Project Summary

<p>This project explored different ML models to predict if a flight was on-time, delayed, or cancelled. The data used for this project came from the U.S. Department of Transportation, Bureau of Transportation Statistics and covers the month of January 2022. There were over 500,000 records in the dataset, each record representing a flight. There were ~300,000 on-time flights, ~170,000 delayed flights, and ~30,000 cancelled flights. The baseline model, Logistic Regression, produced a score of 92%. There was concern of overfitting, so that was addressed in the finetuning stage of the final model. The final model was a Light GBM model with a score of 95%. Overall, this project was able to determine the optimal model for this data set, future projects should entail training on one month's data, and testing on another month's data.</p>

## Problem Statement 

1. Based on the data, I want to predict if a flight will be on-time, delayed, or cancelled based on the airline and the airport flying into.
2. I'll be using the Logistic Regression model as a benchmark for the other models, trying to improve the score.
3. This data comes from the U.S. Department of Transportaion Bureau of Transportation Statistics, with 537,902 records and 11 columns. There are no missing values. There are a few empty cells in the depart time column, but those rows are cancelled flights which are covered by the indicator variable. I'm planning on doing K-Fold cross validation on the data.
4. I hope to be able to create a model that predicts if a flight is on-time, delayed, or cancelled with a score of at least 70%.

## Dataset 
 
1. The dataset was created to better see the details of of flight in a given time period. The data is available to the public, and the website allows the user to build their own dataset of a certain year from the list of variables.
2. The data was collected by the United State Department of Transportion's Bureau of Transportation Statistics (DOTBTS). A summary of the information on the number of on-time, delayed, canceled, and diverted flights appears in the Department of Transportation's monthly Air Travel Consumer's Report.
3. The data includes many variables, the names and descriptions have been provided in the table below. The data is from January 2022 and includes records from the 17 airlines that report to the DOTBTS. I will be creating a new column to use as an indicator for on-time, delayed, or cancelled.   
 
 
| **Variable** | **Description** |
| ---- | -------- |
| OP_UNIQUE_CARRIER | Unique carrier code, specific to a certain airline |
| ORIGIN | Code of the origin airport, where the flight departed from |
| DEST | Code of the destination airport |
| CRS_DEP_TIME | CRS Departure Time (local time: hhmm) |
| DEP_TIME | Actual Departure Time (local time: hhmm) |
| DEP_DELAY_NEW | Difference in minutes between scheduled and actual departure time. Early departures set to 0. |
| CANCELLED | Cancelled Flight Indicator (1=Yes) |
| DAY_OF_THE_MONTH | Numerical representation of the day of the month |
| DAY_OF_THE_WEEK | Numerical representation of the day of the week |

 
 ![download](https://media.git.txstate.edu/user/2190/files/4895512b-139a-4845-99ed-ed06886a3e7e)

 Because the data included string objects, I knew that I was going to have to use a LabelEncoder to turn those strings into integers. 
 ```
 le = preprocessing.LabelEncoder()
 df['OP_UNIQUE_CARRIER'] = le.fit_transform(df['OP_UNIQUE_CARRIER'])
 df['ORIGIN'] = le.fit_transform(df['ORIGIN'])
 df['DEST'] = le.fit_transform(df['DEST'])
 ```
 Luckily, the website where I got the data has the option of removing missing values from the dataset before downloading. My data does still have some "missing" values (about 30,000), but these values were due to flights being cancelled, so the DEP_DELAY_NEW column has no data for that row. Since the indicator variable already accounts for the flight being cancelled, I filled the DEP_DELAY_NEW as 0. 
 ```
 for index, row in df.iterrows():
    if row['INDICATOR']==2:
        df.at[index,'DEP_DELAY_NEW']=0
 ```
 Even though there is no longer missing values, I filtered the dataframe to only include rows that had an indicator variable present.


## Exploratory Data Analysis 

1. I will be using barplots, histograms, and possibly pie charts. 
2. I already have created two barplots to show the count of records for each airline (see above charts).

I began by looking at the distribution of classes in the data, which is the first pie chart aboce. After looking at the class distribution, I wanted to know what the distribution of records was among the different airlines.
 <img src="https://media.git.txstate.edu/user/2190/files/91527ad1-dc82-4969-94ca-1e2ad7339c51" width="400"> 
 Then, out of pure curiousity, I looked at the class distribution of each airline, hence the 17 pie charts.
 
<img src="https://media.git.txstate.edu/user/2190/files/556ac3c0-0925-45c6-8d7a-3d9e8ab18110" width="300">  <img src="https://media.git.txstate.edu/user/2190/files/f59ef47f-7652-461f-ba9a-325f708e26af" width="300"> 
<img src="https://media.git.txstate.edu/user/2190/files/4534d1f0-d5cd-4cee-b2f8-e7fe7f275c47" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/78ea539e-a932-48b5-a417-3aef76d7ae8d" width="300"> 
<img src="https://media.git.txstate.edu/user/2190/files/374ef977-6ebf-4ad6-969d-578e02a6102a" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/cd9aa932-387c-4401-bb56-32f5e65e48b6" width="300">
<img src="https://media.git.txstate.edu/user/2190/files/faa5733e-ed96-452c-8e73-1a5c0cf19989" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/4f174efb-f886-4703-b548-9a84df545e1e" width="300"> 
<img src="https://media.git.txstate.edu/user/2190/files/a9ef9836-2cef-446f-9158-f5ca4e080b6f" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/13c965ac-1354-4e12-9cb8-872c1cfebd80" width="300">
<img src="https://media.git.txstate.edu/user/2190/files/987cd571-1281-4b45-909f-cc6566bb4b74" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/85e872ce-8354-4716-966e-74e4e1d88ff3" width="300">
<img src="https://media.git.txstate.edu/user/2190/files/56597d39-7699-48ba-9237-207228d01e75" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/14549d8b-29a0-46d7-9b10-7047fe04c1fc" width="300">
<img src="https://media.git.txstate.edu/user/2190/files/2921db90-50b4-4d98-a593-19845fe86c3a" width="300"> <img src="https://media.git.txstate.edu/user/2190/files/348ce3c6-deb1-45f5-ab3e-08390922ae01" width="300">
<img src="https://media.git.txstate.edu/user/2190/files/aca4770f-77f3-4654-bf14-f9e3bf029bbf" width="300">

 I used StandardScaler() before splitting the data. Because of my problem statement, I am focusing on classification models, specifically models that are able to handle large datasets.

## Data Preprocessing 

 I began the project without scaling the data, but each model was taking 15+ minutes to run. Even with changing parameters to better handle the data, it was taking longer than it should. I then applied StandardScaler() and immediately had better performance times, around 7-10 minutes.  
 ```
 x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.25, random_state=42)

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
 ```

I began the project without scaling the data, but each model was taking 15+ minutes to run. Even with changing parameters to better handle the data, it was taking longer than it should. I then applied StandardScaler() and immediately had better performance times, around 7-10 minutes.


## Machine Learning Approaches

1. Baseline evaluation is going to be using Multinomial Logistic Regression, because I want to classify the data as on-time, delayed, or cancelled. Knowing that LogReg is used as a binary classification, I would need to use a multi-class classification technique.
2. I will be focusing on classification methods (Logistic Regression, Multiple Naive Bayes, and Decision Tree).  
 Logistic Regression: I chose this one as a baseline for calssification, gives me a good idea of what tuning needs to be done to the data.  
 Multiple Naive Bayes: This one might not work very well, it assumes that all features are independent, and I am trying to prove that these features are dependent, or at least are heavily correlated.  
 Decision Tree: since the training data will have a large set of categorical values, this also seemed like a good model to use.

 
 My first two models used Logistic Regression. The reason for two different models is because I wanted to see the difference in the solvers. Logistic Regression with the default settings wouldn't run fast, and ended with a score of 100%. The solvers 'sag' and 'saga' are specific for large datasets. After applying those, the model was no longer overfitting and was performing at a better time.
 
 <h4 align="center">The baseline model</h4>
 <p align="center">
  <img src="https://media.git.txstate.edu/user/2190/files/0b9b797f-1a4a-4075-ac5c-dcaef0a3981f" width="500">
 </p>
 Performed with a score of 92.6%  
 Cross Validated Scores Array: [0.92170885 0.92170788 0.92146    0.92146    0.92067918]
 
 ### Other models  
 <h4 align="center">Logistic Regression with solver='SAG'</h4>
 <p align="center">
  <img src="https://media.git.txstate.edu/user/2190/files/8cd49fa3-85a3-4a26-a6e7-4dcc31ec7c6b" width="500"> 
 </p>
 
 ```
 from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=10000, solver='sag',n_jobs=-1)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

scores_log=cross_val_score(clf,x_train,y_train,cv=5)
print(scores_log)
 ```
  <h4 align="center">Logistic Regression with solver='SAGA'</h4>
 <p align="center">
  <img src="https://media.git.txstate.edu/user/2190/files/a83eead1-df90-4cdb-848a-d1ecbf55c765" width="500"> 
 </p>
 
 ```
 from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
clf=LogisticRegression(max_iter=10000, solver='saga', n_jobs=-1)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

scores_log=cross_val_score(clf,x_train,y_train,cv=5)
 print(scores_log)
 ```
 <h4 align="center">KNN Classifier using the best K value</h4>
 <p align="center">
  <img src="https://media.git.txstate.edu/user/2190/files/b4cceee6-ab6d-4d4f-b8a2-4809771b22ee" width="500">
 </p>
 
 ```
 knn_best = KNeighborsClassifier(n_neighbors=scores.index(max(scores))+1)
knn_best.fit(x_train,y_train)
knn_best.score(x_test,y_test)
...
 knn_best.score(x_test,y_test)
 >> 0.8635295517415748
 k_scores = []
for k in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
 >> [0.8564594217462498, 0.8560603395076043, 0.8529147836781534, 0.845498301159704, 0.8465542596698821, 0.8389320428721128, 0.8402978445699109, 0.8342223797424044, 0.835134566693872, 0.8299465022918342, 0.8306380800376992, 0.8262209123230466, 0.8268777868434677, 0.8229538948051658, 0.8235388851437371, 0.8200810015448632, 0.8206387235098248, 0.8175700076006391, 0.8179195137458823, 0.8154333076793925, 0.8153787742500693, 0.8133858450273384, 0.8133833648693967, 0.8113309453729615, 0.8112045269937571, 0.8093479300847621, 0.8092785244326273, 0.8075880034227968, 0.8072979866603409]
 print(max(k_scores))
 >> 0.8564594217462498
 ```

 Then began the more robust ML Models, XGBoost, Light GBM, and HistGradientBoostingClassifier.  
 <h4>XGBClassifier</h4>
 
 ```
 import xgboost as xgb
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
xgbscores = []
for lr in learning_rates:
    xgb_clf = xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=lr)
    xgb_clf.fit(x_train, y_train)
    y_pred = xgb_clf.predict(x_test)
    xgbscores.append(xgb_clf.score(x_test,y_test))
    print(xgb_clf.score(x_test,y_test))
print("Best learning rate is ",learning_rates[xgbscores.index(max(xgbscores))]," with a score of ",max(xgbscores))
 ```
 
 | **Learning Rate** | **Score** |
 | ----- | ----- |
 | 0.05 | 0.9521327225675957 |
 | 0.1 | 0.9535233052738035 |
 | 0.25 | 0.9536720306969273 |
 | 0.5 | 0.952184776465689 |
 | 0.75 | 0.9497977334245515 |
 | 1 | 0.9473809452987894 |
 
**Best learning rate is  0.25  with a score of  0.9536720306969273**  
 
  <h4>Light GBM</h4>
 
 ```
 import lightgbm as lgbm

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
lgbmscores = []
for lr in learning_rates:
    lgbm_clf=lgbm.LGBMClassifier(num_leaves=100,n_estimators=100,max_depth=10,learning_rate=lr)
    lgbm_clf.fit(x_train,y_train)
    print(lgbm_clf.score(x_test,y_test))
    lgbmscores.append(lgbm_clf.score(x_test,y_test))
print("Best learning rate is ",learning_rates[lgbmscores.index(max(lgbmscores))]," with a score of ",max(lgbmscores))
 ```
 
 | **Learning Rate** | **Score** |
 | ----- | ----- |
 | 0.05 | 0.9517237276540051 |
 | 0.1 | 0.9532630357833368 |
 | 0.25 | 0.9537612659508017 |
 | 0.5 | 0.9429043100627621 |
 | 0.75 | 0.6865537344953746 |
 | 1 | 0.6036541836461524 |
 
 **Best learning rate is  0.25  with a score of  0.9537612659508017**
 
  <h4>HistGradientBoostingClassifier</h4>
 
```
 from sklearn.ensemble import HistGradientBoostingClassifier
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
histscores = []
for lr in learning_rates:
    hist_clf=HistGradientBoostingClassifier(learning_rate=lr)
    hist_clf.fit(x_train,y_train)
    print(hist_clf.score(x_test,y_test))
    histscores.append(hist_clf.score(x_test,y_test))
print("Best learning rate is ",learning_rates[histscores.index(max(histscores))]," with a score of ",max(histscores))
 ```
 
  | **Learning Rate** | **Score** |
 | ----- | ----- |
 | 0.05 | 0.9495820815610221 |
 | 0.1 | 0.95159731104435 |
 | 0.25 | 0.940569320919718 |
 | 0.5 | 0.9464960290312026 |
 | 0.75 | 0.9300172521490824 |
 | 1 | 0.9208855111692793 |
 
 **Best learning rate is  0.1  with a score of  0.95159731104435**
 

## Experiments 


 My best performing model was Light GBM. The reason for choosing this model is because it has a faster training speed and has better accuracy than other boosting algorithms, but MAINLY because it is able to handle large datasets with relative ease.  
 
 My basline Light GBM gave me a score of 0.95159731104435 with a precision-recall matrix as follows:
 
 <img width="477" alt="Screenshot 2022-12-04 at 8 01 34 PM" src="https://media.git.txstate.edu/user/2190/files/a9b5ee4a-7c64-4cfb-a7af-809b78d949cb">

 After finetuning my LGBM Model by changing learning rates, bagging_fraction, bagging frequency, and the number of leaves, I got the maximum score from the following values:  
        
        Bagging Frequency: 5  
        Bagging Fraction: 0.9  
        Learning Rate: 0.25  
        Leaves: 100  
        Score: 0.953805884  
 
  <img width="463" alt="Screenshot 2022-12-04 at 8 03 48 PM" src="https://media.git.txstate.edu/user/2190/files/79af615a-b08a-4dff-99c7-320a35724237">
 
 The finetuning code was not very efficient, but it got the job done.
 
 ```
import lightgbm as lgbm
outputFile = open("LGBM_output.txt","w")
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
bagging_fractions = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
bagging_frequency = [5,10,15,20]
numleaves = [10,25,50,100]
lgbmscores = []

for lr in learning_rates:
    for bfrac in bagging_fractions:
        for bfre in bagging_frequency:
            for leaves in numleaves:
                lgbm_clf=lgbm.LGBMClassifier(num_leaves=leaves,n_estimators=100,max_depth=10,learning_rate=lr, bagging_fraction=bfrac, bagging_freq=bfre)
                lgbm_clf.fit(x_train,y_train)
                lgbmscores.append(lgbm_clf.score(x_test,y_test))
                outputString = "Frequency: "+str(bfre)+"\tFraction: "+str(bfrac)+"\tLearning Rate: "+str(lr)+"\tLeaves: "+str(leaves)+"\tScore: "+str(lgbm_clf.score(x_test,y_test))
                outputFile.write(outputString)
                outputFile.write('\n')
outputFile.close()
 ```


## Conclusion

 I am worried that, even though the data may be very predictable, that my model is still overfitting the dataset. My scores are no longer 99.99% or 100%, but I am wary that they are still high (90%-95%).  
 In terms of tuning parameters, I researched what parameters for each model would work with handling large datasets and would help with overfitting. For Light GBM, it was suggested to use small ```num_leaves```, ```bagging_freq```, and ```bagging_fraction``` to combat overfitting. That is what my 4 nested for loop testeed, to try to find the optimal combination of the three above, as well as the learning rate.  
 Using the score method that models have, I was able to tell that my data was overfitting. Once I decided to use a precision-recall matrix display, it was easier to see what was going on within each of the predicted classes.

 
 **Submission Format** 
 
1. Python code with markdown documentation, images saved in .jpg or .png format, and README.md as a project report
2. Jupyter notebook (.ipynb) that contains full markdown sections as listed above 
