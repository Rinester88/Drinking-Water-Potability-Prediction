# Drinking-Water-Potability-Prediction
The acceptability of water for human eating and other domestic needs without producing harm or health hazards is referred to as water potability. Potable water, also known as drinking water, must meet certain quality criteria in order to be safe for consumption and pose no health risks
Water potability, the acceptability of water for human consumption and domestic use without posing harm or health hazards, is a critical aspect of public health and safety. Potable water, commonly known as drinking water, must adhere to specific quality standards to ensure it is safe for consumption. This GitHub repository hosts a case study on water potability prediction, exploring various aspects of water quality and utilizing machine learning techniques.

Understanding Water Potability
1. Potable Water's Importance: The availability of safe and potable water is paramount for human health, sanitation, and overall well-being. It is essential for various daily activities such as drinking, cooking, bathing, and more.

2. Parameters of Water Quality: Water potability depends on a range of water quality indicators and pollutants, including pH levels, turbidity, temperature, dissolved oxygen, total dissolved solids (TDS), and the presence of certain chemical compounds and microbes.

3. Regulatory Requirements: Governments and regulatory agencies establish standards and guidelines for drinking water quality. These criteria, although varying by country, typically specify maximum allowable levels for contaminants like heavy metals, pathogens, organic compounds, and more.

4. Water Purification: Municipal water treatment facilities employ various techniques such as filtration, chlorination, ozonation, and ultraviolet (UV) treatment to make water potable. These methods aim to eliminate or neutralize pollutants and pathogens.

5. Typical Contaminants: Contaminants that can affect water potability include bacteria, viruses, protozoa, heavy metals (e.g., lead, arsenic), pesticides, industrial chemicals, and organic matter.

6. Testing and Supervision: Regular water quality testing and monitoring are crucial to ensuring that water remains safe for consumption throughout the distribution system. Homeowners can also assess the potability of their tap water using DIY kits or professional testing services.

7. Health Consequences: Consuming non-potable water can lead to various health issues, including waterborne infections (e.g., cholera, dysentery) and long-term health problems due to exposure to pollutants.

8. Challenges: Providing safe drinking water to all can be challenging, especially in areas with inadequate infrastructure or natural sources of pollution. Factors like climate change, population growth, and pollution can further strain water resources and compromise potability.

Project Goals
The primary objective of this project is to create a predictive model that can determine whether a water sample is potable (safe for consumption) or non-potable using various water quality factors. The project workflow includes the following steps:

Data Collection
Data Preprocessing
Data Splitting
Model Selection
Model Training
Model Evaluation
Reporting and Visualization
We will leverage both traditional Machine Learning (ML) algorithms and H2O AutoML for prediction.

About H2O AutoML
H2O AutoML is an automatic machine learning (AutoML) platform provided by H2O.ai, a leading player in the field of artificial intelligence and machine learning. H2O AutoML simplifies and expedites the process of creating and deploying machine learning models by automating various components of the ML workflow. Here are some key features and aspects of H2O AutoML:

Model Selection Automation: H2O AutoML automates the selection of machine learning methods and hyperparameters, saving valuable time for data scientists and ML developers.

A Diverse Set of Algorithms: It supports a wide range of ML algorithms, including linear models, tree-based models, gradient boosting, deep learning, and more, allowing users to benefit from cutting-edge techniques without in-depth algorithm expertise.

Flexibility: H2O AutoML is designed to handle large datasets and can work in distributed computing environments, making it suitable for organizations dealing with extensive data.

Simple User Interface: The platform offers a user-friendly web-based interface for importing data, configuring AutoML settings, and tracking the automatic model building process.

Model Explanation: H2O AutoML provides tools for model explanation and interpretation, allowing users to understand why a model makes specific predictions, which is crucial for regulatory compliance and model transparency.

Deployment of Models: It simplifies model deployment in various environments, whether on-premises or in the cloud, facilitating the transition from model creation to real-world deployment.

Integration: H2O AutoML seamlessly integrates with existing data science and ML workflows, offering APIs and connectors for popular programming languages like Python and R.

Time-Series Support: The platform includes capabilities for time-series data, such as automated lagging and differencing to address temporal dependencies.

Hyperparameter Adjustment: While H2O AutoML automates some hyperparameter tuning, users can also fine-tune hyperparameters manually if necessary.

Model Stacking: H2O AutoML supports model stacking, allowing users to combine multiple ML models to enhance predictive performance.

H2O AutoML is a valuable tool for organizations and data science teams looking to harness the power of machine learning without extensive expertise in model selection and tuning. It is suitable for a wide range of applications, including predictive analytics, classification, and regression, and its automation features enable rapid development and deployment of robust ML models.

Why Use H2O AutoML for Water Potability Prediction
Using H2O AutoML for water potability prediction offers several advantages, particularly when dealing with large datasets or when you have limited machine learning expertise. Here's why and how H2O AutoML is beneficial for this specific application:

Model Selection Automation: H2O AutoML automates the selection of the best machine learning algorithms and hyperparameters for your dataset. If you're uncertain about which algorithms are suitable for predicting water potability, you can simply upload your water quality dataset to H2O AutoML, which will experiment with multiple algorithms and configurations to discover the most accurate model.

Efficient Use of Time and Resources: Manually selecting, training, and fine-tuning machine learning models can be time-consuming and resource-intensive. H2O AutoML streamlines this process, allowing you to focus on interpreting results rather than getting bogged down in technical details.

Complex Data Handling: Water quality datasets can be complex, with numerous features and potential interactions. H2O AutoML can effectively handle such complexities and identify models that generalize well to your data. It supports a wide range of machine learning algorithms, including ensemble methods and deep learning, making it suitable for diverse data sources.

Flexibility: Some water quality datasets can be large. H2O AutoML is built for scalability, capable of analyzing and modeling substantial datasets effectively. It can leverage distributed computing systems to handle extensive data.

Model Explanation: In public health applications, understanding why a model makes specific predictions is critical. H2O AutoML provides model explainability tools, including SHAP (SHapley Additive exPlanations) values and variable importance analysis, helping you interpret and justify model predictions.

Deployment Readiness: After developing a model with H2O AutoML, it's straightforward to deploy it in production environments, which is essential for real-time water potability prediction. H2O AutoML simplifies deployment through APIs and connectors for seamless integration into production systems.

Model Stacking (Ensemble Learning): H2O AutoML supports model stacking, allowing you to combine different models generated by the platform. This can further improve prediction accuracy, which is valuable when aiming to enhance potability predictions.

Getting Started
To get started with the project, you can follow the steps outlined in the repository. We have provided code snippets and explanations to guide you through data preprocessing, model development, and evaluation.

```
import numpy as np # This line imports the NumPy library and assigns it the alias np.
import pandas as pd # pandas library and assigns it the alias pd. 
import matplotlib.pyplot as plt # imports the matplotlib.pyplot module and assigns it the alias plt. 
import seaborn as sns #  imports the Seaborn library, often aliased as sns. Seaborn is built on top of Matplotlib and provides a higher-level interface for creating attractive and informative statistical graphics
from sklearn.model_selection import train_test_split # imports the train_test_split function from scikit-learn (sklearn).
%matplotlib inline
```
```
df = pd.read_csv(r"C:\Users\HP\Downloads\Water Potability\train_dataset.csv")
Import pandas as pd: This line imports the pandas library and assigns it the alias pd. pandas is a powerful data manipulation and analysis library in Python.

file_path = r"C:\Users\HP\Downloads\Water Potability\train_dataset.csv": This line defines a variable file_path that stores the absolute file path to the CSV file you want to read. The 'r' prefix before the string indicates that it's a raw string, which is used to prevent escape character interpretation in the file path. pd.read_csv(file_path): This line uses the read_csv function from pandas to read the CSV file located at file_path. It loads the data from the CSV file into a pandas DataFrame named df. The resulting DataFrame contains the data from the CSV file, and you can perform various data analysis and manipulation tasks using pandas methods on this DataFrame.
```
```
df.head()
```
```
ph	Hardness	Solids	Chloramines	Sulfate	Conductivity	Organic_carbon	Trihalomethanes	Turbidity	Potability
0	7.080795	219.674262	22210.613083	5.875041	333.775777	398.517703	11.502316	112.412210	2.994259	0
1	6.783888	193.653581	13677.106441	5.171454	323.728663	477.854687	15.056064	66.396293	3.250022	0
2	6.010618	184.558582	15940.573271	8.165222	421.486089	314.529813	20.314617	83.707938	4.867287	1
3	8.097454	218.992436	18112.284447	6.196947	333.775777	376.569803	17.746264	59.909941	4.279082	1
4	8.072612	210.269780	16843.363927	8.793459	359.516169	559.167574	17.263576	68.738989	5.082206	0
```
```
data_shape = df.shape
```
# Data Analysis and Visualization
```
sns.countplot(data=df,x=df.Potability)
df.Potability.value_counts()
```
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/c2ed572a-76c4-4390-9313-b00b459fc374)

```
sns.pairplot(data=df,hue='Potability')
```
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/6c7487e2-e60f-47d1-b7ea-4c24b11448da)
```
# Define the lower and upper bounds for acceptable values of 'PH'
lower_bound = 2  # Set your desired lower bound
upper_bound = 11  # Set your desired upper bound
filtered_df = df[(df['ph'] >= lower_bound) & (df['ph'] <= upper_bound)]
```
```
plt.figure(figsize=(6, 4))
plt.boxplot([filtered_df['ph']], labels=['ph'], vert=False)
plt.title('Box Plot of PH (Outliers Removed)')
plt.xlabel('ph')
plt.show()
```
```
# Define the lower and upper bounds for acceptable values of 'Hardness'
lower_bound = 100  # Set your desired lower bound
upper_bound = 250  # Set your desired upper bound

# Filter the DataFrame to remove outliers in the 'Hardness' column
filtered_df = df[(df['Hardness'] >= lower_bound) & (df['Hardness'] <= upper_bound)]

# Display some statistics to see the impact of removing outliers
print("Original DataFrame shape:", df.shape)
print("Filtered DataFrame shape:", filtered_df.shape)

# Now 'filtered_df' contains the data with outliers removed in the 'Hardness' column

```
```
# Create a histogram to visualize the distribution of 'Hardness' values
plt.figure(figsize=(8, 6))
plt.hist(filtered_df['Hardness'], bins=10, edgecolor='k', alpha=0.7)
plt.title('Histogram of Hardness (Outliers Removed)')
plt.xlabel('Hardness')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/845a94fc-e197-4e01-a56f-bdd374c93d7e)

```
df.corr()
```
```
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')  # Change 'coolwarm' to your desired color map
plt.show()
```
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/fce7c575-f219-4152-babc-58dd93830645)

# Feature Engineering
```
from sklearn.ensemble import ExtraTreesClassifier

x = df.drop(['Potability'],axis =1)
y= df.Potability
```
```
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
```
```
feature =pd.Series(Ext.feature_importances_,index=x.columns)
feature.sort_values(ascending =True).nlargest(10).plot(kind='barh')
```
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/2bb4d48a-e908-4196-b5e6-3c749e4d3f30)
# Lets Standardize the data

```
from sklearn.preprocessing import StandardScaler
```
```
scaled =scale.fit_transform(x)
```
```
scaled_df =pd.DataFrame(scaled,columns =x.columns)
scaled_df.head()
```
# Model Development
We will use the following models:
- logistic Regression
- SVM
- Random Forest
```
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```
# Logistic Regression
```
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_train_hat = lr.predict(X_train)
y_test_hat = lr.predict(X_test)


print('Test performance')
print('-----------------------------------------------')
print(classification_report(y_test, y_test_hat))

print('Roc_auc score')
print('-----------------------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')


print('Confusion matrix')
print('-----------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))
print('')

print('accuracy score')
print('-----------------------------------------------')
print("test data accuracy score:",accuracy_score(y_test, y_test_hat)*100)
print("train data accuracy score:",accuracy_score(y_train, y_train_hat)*100)

```
# Support Vector Machines
```
svm = SVC()
svm.fit(X_train, y_train)
y_train_hat = svm.predict(X_train)
y_test_hat = svm.predict(X_test)

print('Test performance')
print('-----------------------------------------------')
print(classification_report(y_test, y_test_hat))

print('Roc_auc score')
print('-----------------------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')


print('Confusion matrix')
print('-----------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))
print('')

print('accuracy score')
print('-----------------------------------------------')
print("test data accuracy score:",accuracy_score(y_test, y_test_hat)*100)
print("train data accuracy score:",accuracy_score(y_train, y_train_hat)*100)
```
# Random Forest
```
rf = RandomForestClassifier(n_jobs=-1, random_state=123)
rf.fit(X_train, y_train)
y_train_hat =rf.predict(X_train)
y_test_hat = rf.predict(X_test)

print('Test performance')
print('-----------------------------------------------')
print(classification_report(y_test, y_test_hat))

print('Roc_auc score')
print('-----------------------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')


print('Confusion matrix')
print('-----------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))
print('')

print('accuracy score')
print('-----------------------------------------------')
print("test data accuracy score:",accuracy_score(y_test, y_test_hat)*100)
print("train data accuracy score:",accuracy_score(y_train, y_train_hat)*100)

```
# Using AutoML
![image](https://github.com/Rinester88/Drinking-Water-Potability-Prediction/assets/111410933/5e0db0dc-0014-4176-bb6f-d14c274211a6)
```
!pip install requests
!pip install tabulate
!pip install "colorama>=0.3.8"
!pip install future
```
```
!pip install h2o
```
# Importing the h2o python module and H2O AutoML Class
```
import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')
```
# Loading Data
```
import h2o
from h2o.frame import H2OFrame

# Initialize H2O
h2o.init()

# Assuming 'df' is your pandas DataFrame
# Convert it to an H2OFrame
h2o_df = H2OFrame(df)

# Now you can use 'h2o_df' for H2O-related operations
```
```
df =h2o.import_file(r"C:\Users\HP\Downloads\Water Potability\train_dataset.csv")
```
```
df.head()
```
# H2O auto ml can do all the data processing techniques
```
y ="Potability" 
x =df.columns
x.remove(y)
```
# Spliiting the Data
```
df_train,df_test = df.split_frame(ratios=[.8]
```
# Defining the model
```
aml = H2OAutoML(max_runtime_secs=300,max_models=10, verbosity="info", nfolds=2)
```
# Fitting the Model
```
aml.train(x=x, y=y, training_frame=df_train)
```
# Seeing the Leaderboard
```
lb = aml.leaderboard
lb
```
# Getting all the Model IDs
```
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0]
```
```
model_ids
```
```
aml. leader.model_performance(df_test)

```
# Getting the models details for best performing model
```
h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
```
```
output = h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in m
```
```
aml.leader
```
```
y_pred=aml.leader.predict(df_test)
```
```
y_pred
```
