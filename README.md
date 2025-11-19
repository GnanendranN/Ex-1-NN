<H3>NAME: Gnanendran N</H3>
<H3>REGISTER NO.: 212223240037</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 13.08.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.fillna(0)
df.isnull().sum()

df.duplicated()

df['EstimatedSalary'].describe()

scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x

print("Y Values")
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X Training data")
x_train

print("X Testing data")
x_test
```
## OUTPUT:
### Read the dataset from drive
<img width="1125" height="361" alt="image" src="https://github.com/user-attachments/assets/d56ebca6-7331-41ac-bce9-900d540b08b0" />

### Finding Missing Values
<img width="678" height="291" alt="image" src="https://github.com/user-attachments/assets/0cf9e2c7-b6a4-44b3-86b7-bfec23f3334f" />

### Handling Missing values
<img width="871" height="292" alt="image" src="https://github.com/user-attachments/assets/e1ab3690-ee1f-463a-9f49-13e9660a752a" />

### Check for Duplicates
<img width="681" height="237" alt="image" src="https://github.com/user-attachments/assets/54118b4a-74ff-4791-8141-cc285e4308c7" />

### Detect Outliers
<img width="1037" height="173" alt="image" src="https://github.com/user-attachments/assets/962b0666-d950-4495-8480-40f5753d17b4" />

### Normalize the dataset
<img width="1122" height="360" alt="image" src="https://github.com/user-attachments/assets/d949f83e-1ac5-41a5-bee7-e60eabb5502f" />

### Split the dataset into input and output
<img width="614" height="284" alt="image" src="https://github.com/user-attachments/assets/27d22368-f77a-48b3-8f2e-badc85a6f434" />

### Print the training data and testing data
<img width="1120" height="351" alt="image" src="https://github.com/user-attachments/assets/e0719209-b2b4-4685-b33e-f25c863dc2c9" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


