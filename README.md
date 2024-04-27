# Healthcare Dataset

## Introduction 
This synthetic healthcare dataset has been created to serve as a valuable resource for data science, machine learning, and data analysis enthusiasts. It is designed to mimic real-world healthcare data, enabling users to practice, develop, and showcase their data manipulation and analysis skills in the context of the healthcare industry.

## Dataset


-	**Name**               -  This column represents the name of the patient associated with the healthcare record.  
-	**Age**                -  The age of the patient at the time of admission, expressed in years.
-	**Gender**             -  Indicates the gender of the patient, either "Male" or "Female."
-	**Blood Type**         -  The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
-	**Medical Condition**  - This column specifies the primary medical condition or diagnosis associated with the <br>
 patient, such as "Diabetes," "Hypertension," "Asthma," and more.
-	**Date of Admission**  -  The date on which the patient was admitted to the healthcare facility.
-	**Doctor**             -  The name of the doctor responsible for the patient's care during their admission.
-	**Hospital**           -  Identifies the healthcare facility or hospital where the patient was admitted.
-	**Insurance Provider** -  This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
-   **Billing Amount**     -  The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number
-	**Room Number**        -  The room number where the patient was accommodated during their admission.
-	**Admission Type**     -  Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
-	**Discharge Date**     -  The date on which the patient was discharged from the healthcare facility, based on the <br>
 admission date and a random number of days within a realistic range.
-	**Medication**         -  Identifies a medication prescribed or administered to the patient during their admission. <br>
 Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
-	**Test results**       -   Describes the results of a medical test conducted during the patient's admission. Possible <br>
values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.

## Outline

#### This dataset can be utilized for a wide range of purposes, including:

- Developing and testing healthcare predictive models.
- Practicing data cleaning, transformation, and analysis techniques.
- Creating data visualizations to gain insights into healthcare trends.
- Learning and teaching data science and machine learning concepts in a healthcare context.
- You can treat it as a Multi-Class Classification Problem and solve it for Test Results which contains 3 categories(Normal,     
 Abnormal, and Inconclusive).

## Process overview
-Throughout the project, I followed a structured process to develop a predictive model for patient discharge outcomes. Here's an   overview of the process, along with some insights into my experience.
1.	Data Collection and Exploration
2.  Data Preprocessing
3.  Model Selection
4.  Model Training and Evaluation
5.  Iterative Improvement
6.	Validation and Testing
7.	Deployment and Monitoring


## EDA



### Pair plot of numerical variables

```
    numeric_features = ['Age', 'Billing Amount']<br>
    sns.pairplot(df[numeric_features])<br>
    plt.show()
```
![alt text](<average billing per doctor.PNG>)

- creates a scatter plot (pair plot) that visually represents the relationship between 'Age' and 'Billing Amount' variables <br>
 from the dataset.The graph shows a positive correlation between age and billing amount. <br> This means that as people get older, their billing amount tends to increase.

### Histogram of Age

```
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Age'], bins=20, kde=True) 
    plt.xlabel('Age') 
    plt.ylabel('Count') 
    plt.title('Distribution of Age') 
    plt.show()
```
<img width="318" alt="Numeric features" src="https://github.com/Srihari2811/Ml/assets/103255536/fd2ea285-2269-4d2e-82c7-7c0437d56622">


- The histogram plot to visualize the distribution of 'Age' in the dataset. <br>It follows principles of information visualization 
 by  selecting appropriate plot types, labeling axes, and providing a title for context.<br> This line graph is a simple and effective way to visualize the distribution of age in a population.


### Distribution of Billing Amount by Admission Type


```    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
    plt.xlabel('Admission Type')
    plt.ylabel('Billing Amount')
    plt.title('Billing Amount by Admission Type')
    plt.show()
```

- The box plot to visualize the distribution of billing amounts across different admission types.<br> It follows principles of information visualization by selecting appropriate plot types, encoding data attributes, labeling axes, and providing a title for context.
-  Analyzing the distribution of 'Billing Amount' across different 'Admission Types' can provide insights into the importance of  'Admission Type' as a predictor for 'Billing Amount'
- The box plot allows identify potential differences in billing amounts across different admission types and detect any outliers










# Feature importance


``` 
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df[[ 'Length of Stay']]
```


- The 'Length of Stay' feature is derived from the difference in between the 'Date of Admission' and 'Discharge Date' columns.<br>
- In machine learning, feature engineering involves creating new features from existing ones to improve model performance.<br>  
- Adding the length of stay as a feature may provide valuable information to predictive models, especially in healthcare-related applications where the duration of hospitalization can be indicative of various outcomes or conditions.





```
    bins = [0, 30, 50, 70, 100]  
    labels = ['Under 30', '31-50', '51-70', 'Over 70']  

    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    df[['Age', 'Age Group']]
```

- The 'Age Group' feature transforms the continuous 'Age' feature into a categorical variable.<br> 
- This transformation can capture non-linear relationships between age and the target variable, which may improve the performance of machine learning models that work better with categorical data.


``` 
    average_billing_per_doctor = df.groupby('Doctor')['Billing Amount'].mean()
    print(average_billing_per_doctor)
```


- By grouping observations based on the 'Doctor' column. This creates distinct groups for each doctor in the dataset.
- Computing the mean of the 'Billing Amount' within each group,we make a new feature that represents the average billing amount associated with each doctor.
- Understanding the average billing amount per doctor can provide insights into the behavior or performance of individual doctors.



```
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df
```


- The continuous 'Age' feature into five age groups: '0-20', '21-40', '41-60', '61-80', and '81+'.<br> This creates a categorical
  variable 'Age Group' that represents different age ranges.
- converting the continuous 'Age' feature into discrete age groups, transforms the data in a way that may be more suitable for certain types of models, such as decision trees or logistic regression, which handle categorical variables more effectively.
- Discretizing age into meaningful categories can potentially improve model performance.



```
    variables_to_remove = ['Name','Date of Admission','Hospital','Room Number','Discharge Date','Doctor','Age']

    df = df.drop(variables_to_remove, axis=1)
    df
```

- The name, hospital, room number dosent affect the target variable.
- Then the difference between the date of admission and discharge date  created new variable, so were  removed these variables.
- Age variable is created into groups of ages, it was also removed.





# Feature engineering 

### Distribution of Age Groups in the Dataset


``` 
    from sklearn.preprocessing import LabelEncoder
        lc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col]=lc.fit_transform(df[col])

    df['Age Group'] = lc.fit_transform(df['Age Group'])
    df.head()
```



Here we are performing label encoding,
- Label Encoding is a technique used to convert categorical data into numerical form, which can be more easily processed by machine learning algorithms.
- We perform label encoding on Gender, Blood Type,	Medical Condition, Insurance Provider, Admission Type,	Medication,	Test Result
these variables.



### Correlation Heatmap of Features

```
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
```

- Correlation measures the strength and direction of a linear relationship between two variables. 
- There's a strong positive correlation between "Billing Amount" and "Length of Stay" and a weak negative correlation between "Age Group" and "Length of Stay".
- Features in a model that predicts hospital costs, while excluding "Age Group" if it has a weak correlation.



# Model Fitting


## Train test spliting 
```
    X,y=df.drop(['Test Results'],axis=1), df['Test Results']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

```
- Here we perform oparation on  data by splitting it into training and testing sets, which are essential for building and evaluating machine learning models.


```
    X_train.shape, y_test.shape
```
- The size and shape of the training and testing datasets, which is essential for proper data splitting and model training

## Random forest classifier

```
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)   

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
```

- Accuracy of the model on the testing data is approximately 0.499, which means the model correctly predicts the target variable for about 49.9% of the instances in the testing set.
-  The average of metrics across all classes, weighted equally.
- Model's accuracy is slightly below 50%, indicating that its predictive performance is not significantly better than random guessing.
-  Random Forest model has been trained and evaluated, its performance is modest.



 
## XGBoost Classifier model


```
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
```




- Evaluation metrics include accuracy score, classification report, and confusion matrix.
- The model achieved an accuracy of approximately on the testing data.
- The accuracy score indicates the overall predictive performance of the model on the testing data.
- Precision, recall, and F1-score offer a detailed assessment of the model's ability to classify instances for each class.
- The confusion matrix provides additional context on the distribution of correct and incorrect predictions.
- The achived accuracy was 0.50 for XGBoost Classifier


## Logistic regression model

```
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report


    logistic_regression = LogisticRegression()

    logistic_regression.fit(X_train, y_train)

    y_pred = logistic_regression.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
```







- Data splitting is conducted using train_test_split to partition the dataset into training and testing subsets.

- Precision, recall, F1-score, and support are provided for each class (e.g., positive and negative).
- Precision represents the proportion of true positive predictions among all positive predictions.
- Recall indicates the proportion of true positive predictions among all actual positives.
- F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
- Support denotes the number of actual occurrences of each class in the testing set.
- The Accuracy of logistic regression was  0.5003.


## Support vector machine model

```
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report


    svc_model = SVC()

    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)   
```





- The accuracy score furnishes an overall assessment of the model's predictive performance on the testing data.
- The classification report offers detailed insights into the model's precision, recall, and F1-score for each class, facilitating a comprehensive understanding of its classification capabilities.
- Support denotes the number of actual occurrences of each class in the testing set.
- The classification report offers detailed insights into the model's precision, recall, and F1-score for each class, facilitating a comprehensive understanding of its classification capabilities.
- Accuracy of support vector machine was 0.5003



