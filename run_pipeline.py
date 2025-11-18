from pipeline import MLProjectPipeline
pipeline = MLProjectPipeline(None)
"""
# Run regression
regression_results = pipeline.run_regression_pipeline(
    "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/train_data.csv",
    "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/test_data.csv"
)
print(regression_results.head())

# Run classification
classification_results = pipeline.run_classification_pipeline(
     "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/train_data.csv",
    "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/test_data.csv"
)
"""
# Run clustering
clustering_results = pipeline.run_clustering_pipeline(
     "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/train_data.csv",
    "C:/Users/HP/Desktop/kani/Data_science/guvi/Project/Clickstream_data/test_data.csv"
)