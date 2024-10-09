import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import Row

# Create a SparkSession
spark = SparkSession.builder.appName("Combined_Flights_SVM").getOrCreate()

# Load the CSV file into a DataFrame
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(sys.argv[1])

# Define the columns to select
columns_for_analysis = [
    'FlightDate', 'Airline', 'CRSDepTime', 'DepDelayMinutes', 'Month', 'DayOfWeek', 
    'OriginState', 'DestState', 'AirTime', 'Distance', 'TaxiOut', 'TaxiIn', 
    'CRSArrTime', 'ArrDelayMinutes', 'DepartureDelayGroups', 'ArrivalDelayGroups', 
    'Cancelled', 'Diverted', 'ArrDel15'
]

# Select the specified columns
df = df.select(*columns_for_analysis)

# Convert 'ArrDel15' to 1 or 0
df = df.withColumn("ArrDel15", when(col("ArrDel15") >= 1, 1).otherwise(0))

# Drop all the rows with null values
df = df.na.drop()

# Count the occurrences of each class in the target variable
target_counts = df.groupBy("ArrDel15").count()
target_counts.show()

# Select relevant columns for modeling
df1 = df.select('Airline', 'CRSDepTime', 'DepDelayMinutes', 'Month', 'DayOfWeek', 'Distance', 'CRSArrTime', 'ArrDelayMinutes','ArrDel15','OriginState', 'DestState')

# Split the data into training and test sets (80% train, 20% test)
train_df, test_df = df1.randomSplit([0.8, 0.2], seed=42)

# Indexing and One-Hot Encoding for categorical columns
airline_indexer = StringIndexer(inputCol="Airline", outputCol="AirlineIndex")
airline_encoder = OneHotEncoder(inputCol="AirlineIndex", outputCol="AirlineVec")

origin_indexer = StringIndexer(inputCol="OriginState", outputCol="OriginStateIndex")
origin_encoder = OneHotEncoder(inputCol="OriginStateIndex", outputCol="OriginStateVec")

dest_indexer = StringIndexer(inputCol="DestState", outputCol="DestStateIndex")
dest_encoder = OneHotEncoder(inputCol="DestStateIndex", outputCol="DestStateVec")

# Assemble Features
assembler = VectorAssembler(
    inputCols=['AirlineVec', 'OriginStateVec', 'DestStateVec', 'CRSDepTime', 'DepDelayMinutes', 'Month', 'DayOfWeek', 'Distance', 'CRSArrTime'],
    outputCol='assembled_features'
)

# Standardization
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")

# Linear SVM Model
svm = LinearSVC(featuresCol='features', labelCol='ArrDel15')

# Pipeline
pipeline = Pipeline(stages=[airline_indexer, airline_encoder, origin_indexer, origin_encoder, dest_indexer, dest_encoder, assembler, scaler, svm])

# Create ParamGrid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(svm.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(svm.maxIter, [10, 50, 100]) \
    .build()

# Create CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="ArrDel15"),
                          numFolds=3)

# Fit CrossValidator
cv_model = crossval.fit(train_df)

# Get the best model
best_model = cv_model.bestModel

# Make predictions on test data
svm_predictions = best_model.transform(test_df)

# Define a UDF to make the correct prediction
@udf(returnType=IntegerType())
def make_prediction(pred):
    return 1 if pred > 0 else 0

# Apply the UDF to create a new column
svm_predictions = svm_predictions.withColumn("corrected_prediction_int", make_prediction(svm_predictions.prediction))
svm_predictions = svm_predictions.withColumn("corrected_prediction", col("corrected_prediction_int").cast(DoubleType()))

# Show the results
svm_predictions.select("Airline", "ArrDel15", "prediction", "corrected_prediction_int", "corrected_prediction").show(10)

# Use the integer corrected prediction for the confusion matrix
confusion_matrix = svm_predictions.groupBy('ArrDel15').pivot('corrected_prediction_int').count().fillna(0).orderBy('ArrDel15')
confusion_matrix.show()

# Evaluate the model using AUC
evaluator = BinaryClassificationEvaluator(labelCol='ArrDel15')
svm_auc = evaluator.evaluate(svm_predictions, {evaluator.metricName: "areaUnderROC"})
print(f"Linear SVM AUC: {svm_auc}")

# Calculate precision, recall, F1 score, and accuracy using MulticlassClassificationEvaluator
evaluator_precision = MulticlassClassificationEvaluator(labelCol='ArrDel15', predictionCol='corrected_prediction', metricName='precisionByLabel')
evaluator_recall = MulticlassClassificationEvaluator(labelCol='ArrDel15', predictionCol='corrected_prediction', metricName='recallByLabel')
evaluator_f1 = MulticlassClassificationEvaluator(labelCol='ArrDel15', predictionCol='corrected_prediction', metricName='f1')
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='ArrDel15', predictionCol='corrected_prediction', metricName='accuracy')

precision = evaluator_precision.evaluate(svm_predictions)
recall = evaluator_recall.evaluate(svm_predictions)
f1 = evaluator_f1.evaluate(svm_predictions)
accuracy = evaluator_accuracy.evaluate(svm_predictions)

# Display the metrics in a tabular format
metrics = [
    Row(metric="AUC", value=svm_auc),
    Row(metric="Accuracy", value=accuracy),
    Row(metric="Precision", value=precision),
    Row(metric="Recall", value=recall),
    Row(metric="F1 Score", value=f1)
]

# Convert the list of metrics to a DataFrame
metrics_df = spark.createDataFrame(metrics)

# Show the metrics in a tabular form
metrics_df.show()

# Print the best parameters
print("Best Model Parameters:")
best_svm = best_model.stages[-1]
print(f"Regularization Parameter: {best_svm.getRegParam()}")
print(f"Max Iterations: {best_svm.getMaxIter()}")

# Extract and display feature coefficients
print("Feature Coefficients:")
# Extract the feature coefficients
coefficients = best_svm.coefficients
# Get the input column names from the VectorAssembler stage
input_cols = best_model.stages[6].getInputCols()  # Assumes assembler is the 7th stage in the pipeline

# Print feature coefficients
for feature, coef in zip(input_cols, coefficients):
    print(f"{feature}: {coef}")

