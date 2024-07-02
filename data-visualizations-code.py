import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as fn
import seaborn as sns
from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as fn
from pyspark.sql import SparkSession
from pyspark import SparkConf

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Read Voter File Data") \
    .getOrCreate()

# pull Georgia data from GCP bucket storage
ga_data = spark.read.format("parquet").option('nullValue','null').load("gs://voter-files-16/VM2Uniform--GA--2021-04-16/*.parquet")

# random sample to downsize data
ga_samp = ga_data.sample(withReplacement=False, fraction=0.01, seed=42)


###########################################################
## PLOT 1 - ethnicity count v. num of registrated voters ##
###########################################################
murray_county = ga_data[fn.col('County') == 'MURRAY'].cache()
murray_county.count()
young_voters = murray_county.withColumn('Voters_Age',fn.col('Voters_Age').cast('int')) \
            .filter((fn.col('Voters_Age') >= 18) & (fn.col('Voters_Age') <= 30)).cache()

df_GA_ethnicgroup = young_voters.select(['EthnicGroups_EthnicGroup1Desc'])

df_GA_ethnicgroup = df_GA_ethnicgroup.groupBy('EthnicGroups_EthnicGroup1Desc')\
    .agg(count('EthnicGroups_EthnicGroup1Desc').alias('ethnicgroup_count')) \
    .orderBy(col('ethnicgroup_count').desc()).dropna()

df_GA_ethnicgroup_plt = df_GA_ethnicgroup.toPandas()
plt.rcParams.update({'font.size': 10})
df_GA_ethnicgroup_plt.plot.bar(x = 'EthnicGroups_EthnicGroup1Desc', y = 'ethnicgroup_count', 
                               xlabel = 'Ethnicity', ylabel = '# of Registered Voters', 
                               title = 'Count of Ethnicities in Dataset', rot = 15, legend = False, color = '#2A7DBD')


##############################################
## PLOT 2 - ethnicity v. 2020 voter turnout ##
##############################################
df_GA_ethnicgroup = young_voters.select(['EthnicGroups_EthnicGroup1Desc', 'General_2020'])

df_GA_ethnicgroup = df_GA_ethnicgroup.na.fill(value='N',subset=['General_2020'])
df_GA_ethnicgroup = df_GA_ethnicgroup.withColumn('General_2020', translate('General_2020', 'Y', '1'))
df_GA_ethnicgroup = df_GA_ethnicgroup.withColumn('General_2020', translate('General_2020', 'N', '0'))
df_GA_ethnicgroup = df_GA_ethnicgroup.withColumn('General_2020', col('General_2020').cast('int'))

df_GA_ethnicgroup_turnout = df_GA_ethnicgroup.groupBy('EthnicGroups_EthnicGroup1Desc')\
    .agg(count('EthnicGroups_EthnicGroup1Desc').alias('ethnicgroup_count'),
         avg('General_2020').alias('general_2020_turnout'))\
    .orderBy(col('general_2020_turnout').desc()).drop('ethnicgroup_count').dropna()

df_GA_ethnicgroup_turnout = df_GA_ethnicgroup_turnout.toPandas()
plt.rcParams.update({'font.size': 10})
df_GA_ethnicgroup_turnout.plot.bar(x = 'EthnicGroups_EthnicGroup1Desc', y = 'general_2020_turnout', 
                               xlabel = 'Ethnicity', ylabel = 'General 2020 Election Turnout', 
                               title = 'General 2020 Election Turnout by Ethnicity', rot = 15, legend = False, color = '#2A7DBD')



#################################################
## PLOT 3 - ethnicity distr. of Georgia voters ##
#################################################
ga_group = young_voters.select('EthnicGroups_EthnicGroup1Desc').groupby('EthnicGroups_EthnicGroup1Desc').count()
ethga = [row[0] for row in ga_group.select('EthnicGroups_EthnicGroup1Desc').collect()]
numga = [row[0] for row in ga_group.select('count').collect()]

fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(numga, labels=ethga,
                                           autopct='%.2f%%',
                                           textprops={'size': 'smaller'})
plt.setp(autotexts, size='x-small')
autotexts[0].set_color('white')
autotexts[1].set_color('white')
autotexts[2].set_color('white')
autotexts[3].set_color('white')
autotexts[4].set_color('white')
plt.title('Ethnic Distribution of Georgia Voters who Specified Ethnicity')



##############################################################################
## PLOT 4 - counts of voters with recreational/personal interests specified ##
##############################################################################
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt
import seaborn as sns

# Rename the actual dataframe columns to match the odd interest column names
rename_map = {
    'CommercialDataLL_Interest_in_Camping_Hiking_In_Household': 'Camping_Hiking',
    'CommercialDataLL_Interest_in_Cooking_General_In_Household': 'Cooking_General',
    'CommercialDataLL_Interest_in_Cooking_Gourmet_In_Household': 'Cooking_Gourmet',
    'CommercialDataLL_Interest_in_Crafts_In_Household': 'Crafts',
    'CommercialDataLL_Interest_in_Current_Affairs_Politics_In_Household': 'Current_Affairs_Politics',
    'CommercialDataLL_Interest_in_Education_Online_In_Household': 'Education_Online',
    'CommercialDataLL_Interest_in_Electronic_Gaming_In_Household': 'Electronic_Gaming',
    'CommercialDataLL_Interest_in_Exercise_Health_In_Household': 'Exercise_Health'
}

# Apply the column renaming to the dataframe
renamed_df = young_voters
for old_name, new_name in rename_map.items():
    renamed_df = renamed_df.withColumnRenamed(old_name, new_name)

# List of interest columns (renamed)
interest_columns = list(rename_map.values())

# Initialize subplots
fig, axs = plt.subplots(1, len(interest_columns), figsize=(15, 6))

# Convert Yes/Null to 1 or 0 for each interest column and plot
for idx, column in enumerate(interest_columns):
    selected_df = renamed_df.withColumn(column, 
                                        when(col(column) == 'Yes', 1)
                                        .when(col(column).isNull(), 0)
                                        .otherwise(0))

    # Remove rows with all NaN values in interest columns
    #selected_df = selected_df.dropna(subset=[column])

    # Convert to Pandas for visualization
    pandas_df = selected_df.toPandas()

    # Plot in the corresponding subplot
    sns.countplot(x=column, data=pandas_df, ax=axs[idx])
    #axs[idx].set_title(f"{column}")
    axs[idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

# Adjust layout and add legend
plt.tight_layout()
plt.show()



#############################################################
## PLOT 5 - plot 4 information but w/ proportion not count ##
#############################################################
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt
import seaborn as sns

# Rename the actual dataframe columns to match the odd interest column names
rename_map = {
    'CommercialDataLL_Interest_in_Camping_Hiking_In_Household': 'Camping_Hiking',
    'CommercialDataLL_Interest_in_Cooking_General_In_Household': 'Cooking_General',
    'CommercialDataLL_Interest_in_Cooking_Gourmet_In_Household': 'Cooking_Gourmet',
    'CommercialDataLL_Interest_in_Crafts_In_Household': 'Crafts',
    'CommercialDataLL_Interest_in_Current_Affairs_Politics_In_Household': 'Current_Affairs_Politics',
    'CommercialDataLL_Interest_in_Education_Online_In_Household': 'Education_Online',
    'CommercialDataLL_Interest_in_Electronic_Gaming_In_Household': 'Electronic_Gaming',
    'CommercialDataLL_Interest_in_Exercise_Health_In_Household': 'Exercise_Health'
}

# Apply the column renaming to the dataframe
renamed_df = young_voters
for old_name, new_name in rename_map.items():
    renamed_df = renamed_df.withColumnRenamed(old_name, new_name)

# List of interest columns (renamed)
interest_columns = list(rename_map.values())

# Initialize subplots
fig, axs = plt.subplots(1, len(interest_columns), figsize=(15, 6))

# Convert Yes/Null to 1 or 0 for each interest column and plot the ratio
for idx, column in enumerate(interest_columns):
    selected_df = renamed_df.withColumn(column, 
                                        when(col(column) == 'Yes', 1)
                                        .when(col(column).isNull(), 0)
                                        .otherwise(0))

    # Calculate the ratio of 0 and 1 for the interest column
    ratio_values = selected_df.groupBy(column).count()
    ratio_values = ratio_values.withColumn('Ratio', col('count') / selected_df.count())

    # Convert to Pandas for visualization
    pandas_df = ratio_values.toPandas()

    # Plot the ratio in the corresponding subplot
    sns.barplot(x=column, y='Ratio', data=pandas_df, ax=axs[idx])
    axs[idx].set_title(f"{column}")
    axs[idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

# Adjust layout
plt.tight_layout()
plt.show()



###############################################
## TABLE - count and ratio for each interest ##
###############################################
from pyspark.sql.functions import when, col
import pandas as pd

# Rename the actual dataframe columns to match the odd interest column names
rename_map = {
    'CommercialDataLL_Interest_in_Camping_Hiking_In_Household': 'Camping_Hiking',
    'CommercialDataLL_Interest_in_Cooking_General_In_Household': 'Cooking_General',
    'CommercialDataLL_Interest_in_Cooking_Gourmet_In_Household': 'Cooking_Gourmet',
    'CommercialDataLL_Interest_in_Crafts_In_Household': 'Crafts',
    'CommercialDataLL_Interest_in_Current_Affairs_Politics_In_Household': 'Current_Affairs_Politics',
    'CommercialDataLL_Interest_in_Education_Online_In_Household': 'Education_Online',
    'CommercialDataLL_Interest_in_Electronic_Gaming_In_Household': 'Electronic_Gaming',
    'CommercialDataLL_Interest_in_Exercise_Health_In_Household': 'Exercise_Health'
}

# Apply the column renaming to the dataframe
renamed_df = young_voters
for old_name, new_name in rename_map.items():
    renamed_df = renamed_df.withColumnRenamed(old_name, new_name)

# List of interest columns (renamed)
interest_columns = list(rename_map.values())

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Interest_Column', 'Count_Yes', 'Count_Null', 'Ratio_Yes'])

# Convert Yes/Null to 1 or 0 for each interest column and calculate the count and ratio
for idx, column in enumerate(interest_columns):
    selected_df = renamed_df.withColumn(column, 
                                        when(col(column) == 'Yes', 1)
                                        .when(col(column).isNull(), 0)
                                        .otherwise(0))

    # Calculate the count of Yes and Null values for the interest column
    count_values = selected_df.groupBy(column).count().collect()
    count_yes = next((row['count'] for row in count_values if row[column] == 1), 0)
    count_null = next((row['count'] for row in count_values if row[column] == 0), 0)

    # Calculate the ratio of Yes to Null values for the interest column
    ratio_yes = count_yes / (count_yes + count_null) if count_yes + count_null != 0 else 0

    # Add the results to the DataFrame
    result_df = pd.concat([result_df, pd.DataFrame({'Interest_Column': [column],
                                                    'Count_Yes': [count_yes],
                                                    'Count_Null': [count_null],
                                                    'Ratio_Yes': [ratio_yes]})])

# Print the resulting DataFrame
print("Count and Ratio for each Interest Column:")
print(result_df)



###########################################################
## PLOT 6 - 2020 voter turnout v. Georgia regional areas ##
###########################################################
area_columns = ['General_2020','County','Residence_Addresses_Latitude',
                  'Residence_Addresses_Longitude','CommercialData_EstimatedHHIncomeAmount','CommercialData_EstimatedAreaMedianHHIncome']
ga_area = ga_samp.select(area_columns)
ga_area = ga_area.withColumnRenamed('General_2020', 'General 2020') \
       .withColumnRenamed('Residence_Addresses_Latitude', "Latitude") \
       .withColumnRenamed('Residence_Addresses_Longitude', "Longitude")\
       .withColumnRenamed("CommercialData_EstimatedHHIncomeAmount", "Household Income") \
       .withColumnRenamed("CommercialData_EstimatedAreaMedianHHIncome", "Median Household Income")

# cast proper column types, fix format
ga_area = ga_area.withColumn("Household Income", regexp_replace(col("Household Income"), "\\$", "").cast("int"))
ga_area = ga_area.withColumn("Median Household Income", regexp_replace(col("Median Household Income"), "\\$", "").cast("int"))
ga_area = ga_area.withColumn('Latitude', col('Latitude').cast('float'))
ga_area = ga_area.withColumn('Longitude', col('Longitude').cast('float'))
null_counts = ga_area.select([count(when(isnull(c), c)).alias(c) for c in ga_area.columns])

# percentage of null values in each column
for column_name in null_counts.columns:
    null_counts = null_counts.withColumn(column_name, round(col(column_name) / ga_area.count(), 3))
    
print("Ratio of Null Values for each Area and Income Column:\n")
print(null_counts.show())

ga_area_clean = ga_area.dropna()
ga_area_clean.count()

ga_area_vote = ga_area.na.fill(value='N',subset=['General 2020'])
ga_area_vote = ga_area_vote.withColumn('General 2020', translate('General 2020', 'Y', '1'))
ga_area_vote = ga_area_vote.withColumn('General 2020', translate('General 2020', 'N', '0'))
ga_area_vote = ga_area_vote.withColumn('General 2020', col('General 2020').cast('int'))

# coarsen gridsize for plotting
from pyspark.ml.feature import Bucketizer

min_long = ga_area_vote.agg({"Longitude": "min"}).collect()[0][0]
max_long = ga_area_vote.agg({"Longitude": "max"}).collect()[0][0]
min_lat = ga_area_vote.agg({"Latitude": "min"}).collect()[0][0]
max_lat = ga_area_vote.agg({"Latitude": "max"}).collect()[0][0]

# Generate splits
step = .05
splits_long = list(np.arange(min_long, max_long + step, step))
splits_lat = list(np.arange(min_lat, max_lat + step, step))

bucketizer_long = Bucketizer(splits=splits_long, inputCol="Longitude", outputCol="long_bucket")
bucketizer_lat = Bucketizer(splits=splits_lat, inputCol="Latitude", outputCol="lat_bucket")

ga_area_vote_buck = bucketizer_long.transform(ga_area_vote)
ga_area_vote_buck = bucketizer_lat.transform(ga_area_vote_buck)
                                                                                
from pyspark.ml.feature import OneHotEncoder

encoder_long = OneHotEncoder(inputCols=["long_bucket"], outputCols=["long_bucket_vec"])
ga_area_vote_encoded = encoder_long.fit(ga_area_vote_buck).transform(ga_area_vote_buck)
encoder_lat = OneHotEncoder(inputCols=["lat_bucket"], outputCols=["lat_bucket_vec"])
ga_area_vote_encoded = encoder_lat.fit(ga_area_vote_encoded).transform(ga_area_vote_encoded)

ga_area_vote_encoded.show()

ga_area_vote_pandas = ga_area_vote_encoded.toPandas()
                                                                                
ga_area_turnout = ga_area_vote_pandas.groupby(['long_bucket', 'lat_bucket']).agg({'General 2020': 'mean'}).reset_index()
plt.figure(figsize=(10, 6))
plt.scatter(data=ga_area_turnout, x='long_bucket', y='lat_bucket', c='General 2020', alpha=0.8)
plt.colorbar(label='General 2020 Turnout')
plt.title('General 2020 Voter Turnout Across Regions within Georgia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()




#########################################################
## PLOT 7 - household income v. Georgia regional areas ##
#########################################################
ga_area_pandas = ga_area_clean.toPandas()
                                                                                
# Plot longitude and latitude with income data
plt.figure(figsize=(10, 6))
plt.scatter(data=ga_area_pandas, x='Longitude', y='Latitude', c='Household Income', alpha=0.8)
plt.colorbar(label='Household Income')
plt.title('Household Income Across Regions within Georgia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

#############################################################################################################
## PLOT 7.5 - median household income v. Georgia regional areas (smoother, more interpretable than Plot 7) ##
#############################################################################################################
plt.figure(figsize=(10, 6))
plt.scatter(data=ga_area_pandas, x='Longitude', y='Latitude', c='Median Household Income', alpha=0.8)
plt.colorbar(label='Median Household Income')
plt.title('Median Household Income Across Regions within Georgia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


############################################
## PLOT 8 - avg voter turnout rate v. age ##
############################################
relevant_data = ga_data.select(
    "Voters_Age",
    "Voters_Gender",
    "Ethnic_Description",
    "CommercialData_EstimatedHHIncome",
    "General_2020", "General_2016", "General_2012", # Add or remove years as needed
)

turnout_by_age = relevant_data.groupBy("Voters_Age").agg(
    avg("General_2020").alias("Average_Turnout_2020")
)

converted_data = relevant_data.withColumn(
    "Voted_2020",
    when(col("General_2020") == 'Y', 1).otherwise(0)
)

filtered_data = converted_data.filter(col("Voters_Age") != 100)

turnout_by_age = filtered_data.groupBy("Voters_Age").agg(
    avg("Voted_2020").alias("Average_Turnout_2020")
).orderBy("Voters_Age")

turnout_by_age_pd = turnout_by_age.toPandas()

plt.figure(figsize=(14, 8))
sns.barplot(x="Voters_Age", y="Average_Turnout_2020", data=turnout_by_age_pd, palette="viridis")
plt.title("Average Voter Turnout by Age for General 2020")
plt.xlabel("Age")
plt.ylabel("Average Turnout Rate")
plt.xticks(rotation=90)  
plt.tight_layout()
plt.show()


###################################################################
## PLOT 9 - 2020 voter turnout v. total registered voters v. age ##
###################################################################
filtered_data = converted_data.filter(col("Voters_Age") != 100)

total_and_voted_by_age = filtered_data.groupBy("Voters_Age").agg(
    count("*").alias("Total_Voters"),
    sum("Voted_2020").alias("Voters_Who_Voted")
).orderBy("Voters_Age")

total_and_voted_by_age_pd = total_and_voted_by_age.toPandas()

plt.figure(figsize=(14, 8))
sns.barplot(x="Voters_Age", y="Total_Voters", data=total_and_voted_by_age_pd, color='lightgrey', label='Total Voters')
sns.barplot(x="Voters_Age", y="Voters_Who_Voted", data=total_and_voted_by_age_pd, color='blue', label='Voters Who Voted')
plt.legend()
plt.title("Total Voters and Voters Who Voted by Age for General 2020")
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



##########################################################
## PLOT 10 - 2020 voter turnout v. est household income ##
##########################################################
turnout_by_gender = converted_data.groupBy("Voters_Gender").agg(
    avg("Voted_2020").alias("Average_Turnout_By_Gender")
)

turnout_by_gender_pd = turnout_by_gender.toPandas()

income_turnout_relationship = converted_data.groupBy("CommercialData_EstimatedHHIncome").agg(
    avg("Voted_2020").alias("Average_Turnout_By_Income")
).orderBy("CommercialData_EstimatedHHIncome")

income_turnout_relationship_pd = income_turnout_relationship.toPandas()

plt.figure(figsize=(14, 8))
sns.lineplot(x="CommercialData_EstimatedHHIncome", y="Average_Turnout_By_Income", data=income_turnout_relationship_pd)
plt.title("Voter Turnout by Estimated Household Income for General 2020")
plt.xlabel("Estimated Household Income")
plt.ylabel("Average Turnout Rate")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


####################################################################
## PLOT 11 - 2020 avg voter turnout v. household income v. gender ##
####################################################################
turnout_by_income_gender = converted_data.groupBy("CommercialData_EstimatedHHIncome", "Voters_Gender").agg(
    avg("Voted_2020").alias("Average_Turnout_By_Income_Gender")
).orderBy("CommercialData_EstimatedHHIncome", "Voters_Gender")

turnout_by_income_gender_pd = turnout_by_income_gender.toPandas()

# Plotting the relationship between income, gender, and voter turnout
plt.figure(figsize=(12, 6))
sns.barplot(
    x="CommercialData_EstimatedHHIncome",
    y="Average_Turnout_By_Income_Gender",
    hue="Voters_Gender",
    data=turnout_by_income_gender_pd,
    palette="muted"
)
plt.title("Average Voter Turnout by Income and Gender for General 2020")
plt.xlabel("Estimated Household Income")
plt.ylabel("Average Turnout Rate")
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust layout to make room for the x-axis labels
plt.legend(title='Gender')
plt.show()



############################################################
## PLOT 12 - avg age v. ethnicity among registered voters ##
############################################################
avg_age_by_ethnicity = ga_data.groupBy("Ethnic_Description").agg(
    avg("Voters_Age").alias("Average_Age")
).orderBy("Average_Age")

avg_age_by_ethnicity_pd = avg_age_by_ethnicity.toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(x="Ethnic_Description", y="Average_Age", data=avg_age_by_ethnicity_pd, palette="coolwarm")
plt.title("Average Age by Ethnic Group")
plt.xlabel("Ethnic Group")
plt.ylabel("Average Age")
plt.xticks(rotation=90)  
plt.tight_layout() 
plt.show()



#########################################################################
## PLOT 13 - avg household income v. ethnicity among registered voters ##
#########################################################################
ga_data = ga_data.withColumn(
    "Cleaned_Income",
    regexp_replace(col("CommercialData_EstimatedHHIncomeAmount"), "[\\$,]", "").cast("integer")
)
income_by_ethnicity = ga_data.groupBy("Ethnic_Description").agg(
    avg("Cleaned_Income").alias("Average_Income")
).orderBy("Ethnic_Description")
income_by_ethnicity_pd = income_by_ethnicity.toPandas()
income_by_ethnicity_pd = income_by_ethnicity_pd.dropna(subset=["Average_Income"])
income_by_ethnicity_pd_sorted = income_by_ethnicity_pd.sort_values(by="Average_Income")
plt.figure(figsize=(14, 8))
barplot = sns.barplot(
    x="Ethnic_Description",
    y="Average_Income",
    data=income_by_ethnicity_pd_sorted,
    palette='Spectral'
)
plt.xticks(rotation=90)
plt.xlabel("Ethnic Group")
plt.ylabel("Average Household Income")
plt.title("Average Household Income by Ethnic Group")
plt.tight_layout()
plt.show()










