from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

###########################################
## First model:                          ##
## -  predictors = lifestyle interest    ##
## -  AUC ROC = .57                      ##
###########################################

# Define the "Voters_Turnout" column based on "General_2020"
model_data = young_voters.withColumn(
    "Voters_Turnout",
    F.when(F.col("General_2020") == 'Y', 1).otherwise(0)
)

# Define interest columns and convert Yes/Null to 1 or 0
interest_columns = [
    'CommercialDataLL_Interest_in_Boating_Sailing_In_Household',
    'CommercialDataLL_Interest_in_Camping_Hiking_In_Household',
    'CommercialDataLL_Interest_in_Cooking_General_In_Household',
    'CommercialDataLL_Interest_in_Cooking_Gourmet_In_Household',
    'CommercialDataLL_Interest_in_Crafts_In_Household',
    'CommercialDataLL_Interest_in_Current_Affairs_Politics_In_Household',
    'CommercialDataLL_Interest_in_Education_Online_In_Household',
    'CommercialDataLL_Interest_in_Electronic_Gaming_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Aerobic_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Health_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Running_Jogging_In_Household',
]

for column in interest_columns:
    model_data = model_data.withColumn(
        column,
        F.when(F.col(column) == 'Yes', 1).otherwise(0)
    )

# Select columns related to interests/lifestyle and voter turnout
selected_columns = interest_columns + ['Voters_Turnout']

# Filter the dataframe to select only the relevant columns
selected_df = model_data.select(selected_columns)

# Drop any rows with null values in the selected columns
selected_df = selected_df.dropna()

# Define the feature vector assembler
assembler = VectorAssembler(inputCols=interest_columns, outputCol='features')

# Define the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='Voters_Turnout')

# Create a pipeline for data preprocessing and model training
pipeline = Pipeline(stages=[assembler, lr])

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = selected_df.randomSplit([0.8, 0.2], seed=42)

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='Voters_Turnout')
auc = evaluator.evaluate(predictions)

# Print the Area Under ROC Curve (AUC) score
print(f"AUC: {auc}")

# Show sample predictions and actual values
predictions.select('Voters_Turnout', 'prediction', 'probability').show(10, truncate=False)

# Calculate confusion matrix
conf_matrix = predictions.groupBy('Voters_Turnout', 'prediction').count()
conf_matrix.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = selected_df.toPandas().corr()

# Plot the correlation matrix using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()




from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

########################################################
## Second model:                                      ##
## -  predictors = age + gender + household income    ##
## -  AUC ROC = .584                                  ##
########################################################
# Define the "Voters_Turnout" column based on "General_2020"
model_data = young_voters.withColumn(
    "Voters_Turnout",
    F.when(F.col("General_2020") == 'Y', 1).otherwise(0)
)


# Remove the dollar sign and cast the column to integer
model_data = model_data.withColumn(
    "CommercialData_EstimatedHHIncomeAmount",
    F.col("CommercialData_EstimatedHHIncomeAmount").substr(2, 100).cast("integer")
)

# Define interest columns and convert Yes/Null to 1 or 0
interest_columns = [
    "Voters_Age",
    "Voters_Gender",
    "CommercialData_EstimatedHHIncomeAmount"
]

for column in interest_columns:
    if column == "Voters_Gender":
        model_data = model_data.withColumn(
            column,
            F.when(F.col(column) == 'M', 1).otherwise(0)
        )

# Select columns related to interests/lifestyle and voter turnout
selected_columns = interest_columns + ['Voters_Turnout']

# Filter the dataframe to select only the relevant columns
selected_df = model_data.select(selected_columns)

# Drop any rows with null values in the selected columns
selected_df = selected_df.dropna()

# Define the feature vector assembler
assembler = VectorAssembler(inputCols=interest_columns, outputCol='features')

# Define the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='Voters_Turnout')

# Create a pipeline for data preprocessing and model training
pipeline = Pipeline(stages=[assembler, lr])

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = selected_df.randomSplit([0.8, 0.2], seed=42)

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='Voters_Turnout')
auc = evaluator.evaluate(predictions)

# Print the Area Under ROC Curve (AUC) score
print(f"AUC: {auc}")

# Show sample predictions and actual values
predictions.select('Voters_Turnout', 'prediction', 'probability').show(10, truncate=False)

# Calculate confusion matrix
conf_matrix = predictions.groupBy('Voters_Turnout', 'prediction').count()
conf_matrix.show()



from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

##############################################################################
## Third model:                                                             ##
## -  predictors = lifestyle interests + age + gender + household income    ##
## -  AUC ROC = .612                                                        ##
##############################################################################

# Define the "Voters_Turnout" column based on "General_2020"
model_data = young_voters.withColumn(
    "Voters_Turnout",
    F.when(F.col("General_2020") == 'Y', 1).otherwise(0)
)

# Remove the dollar sign and cast the column to integer for CommercialData_EstimatedHHIncomeAmount
model_data = model_data.withColumn(
    "CommercialData_EstimatedHHIncomeAmount",
    F.col("CommercialData_EstimatedHHIncomeAmount").substr(2, 100).cast("integer")
)

# Define interest columns and convert Yes/Null to 1 or 0
interest_columns_1 = [
    'CommercialDataLL_Interest_in_Boating_Sailing_In_Household',
    'CommercialDataLL_Interest_in_Camping_Hiking_In_Household',
    'CommercialDataLL_Interest_in_Cooking_General_In_Household',
    'CommercialDataLL_Interest_in_Cooking_Gourmet_In_Household',
    'CommercialDataLL_Interest_in_Crafts_In_Household',
    'CommercialDataLL_Interest_in_Current_Affairs_Politics_In_Household',
    'CommercialDataLL_Interest_in_Education_Online_In_Household',
    'CommercialDataLL_Interest_in_Electronic_Gaming_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Aerobic_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Health_In_Household',
    'CommercialDataLL_Interest_in_Exercise_Running_Jogging_In_Household',
]

for column in interest_columns_1:
    model_data = model_data.withColumn(
        column,
        F.when(F.col(column) == 'Yes', 1).otherwise(0)
    )

# Define interest columns for the second model
interest_columns_2 = [
    "Voters_Age",
    "Voters_Gender",
    "CommercialData_EstimatedHHIncomeAmount"
]

# Transform Voters_Gender in interest_columns_2
for column in interest_columns_2:
    if column == "Voters_Gender":
        model_data = model_data.withColumn(
            column,
            F.when(F.col(column) == 'M', 1).otherwise(0)
        )

# Select all relevant interest columns
all_interest_columns = interest_columns_1 + interest_columns_2

# Select columns related to interests/lifestyle and voter turnout
selected_columns = all_interest_columns + ['Voters_Turnout']

# Filter the dataframe to select only the relevant columns
selected_df = model_data.select(selected_columns)

# Drop any rows with null values in the selected columns
selected_df = selected_df.dropna()

# Define the feature vector assembler
assembler = VectorAssembler(inputCols=all_interest_columns, outputCol='features')

# Define the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='Voters_Turnout')

# Create a pipeline for data preprocessing and model training
pipeline = Pipeline(stages=[assembler, lr])

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = selected_df.randomSplit([0.8, 0.2], seed=42)

# Fit the pipeline to the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='Voters_Turnout')
auc = evaluator.evaluate(predictions)

# Print the Area Under ROC Curve (AUC) score
print(f"AUC: {auc}")

# Show sample predictions and actual values
predictions.select('Voters_Turnout', 'prediction', 'probability').show(10, truncate=False)

# Calculate confusion matrix
conf_matrix = predictions.groupBy('Voters_Turnout', 'prediction').count()
conf_matrix.show()



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#############################
## Third model - ROC curve ##
#############################

# Get the ROC curve data from the BinaryClassificationEvaluator
results = predictions.select(['Voters_Turnout', 'probability']).rdd.map(lambda row: (float(row['probability'][1]), float(row['Voters_Turnout'])))

# Extract the false positive rate (FPR), true positive rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(results.map(lambda x: x[1]).collect(), results.map(lambda x: x[0]).collect())

# Calculate the Area Under Curve (AUC) score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


##############################################
## Third model - confusion matrix heaat map ##
##############################################

# Convert the confusion matrix to a Pandas DataFrame
conf_matrix_df = conf_matrix.toPandas()

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df.pivot("Voters_Turnout", "prediction", "count"), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
