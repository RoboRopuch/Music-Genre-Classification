# Music-Genre-Classification

## Description:
The Music Genre Classification System aims to predict the genre of songs based on their audio features and metadata. It involves preprocessing audio data, training machine learning models, and deploying an API for making predictions. The project also includes conducting experiments to analyze user engagement and behavior.

## Key Features:
- **Data Preparation**: Load and preprocess artist and track data, group similar genres, and reduce the number of genres.
- **Modeling**: Train baseline and advanced models using Random Forest Classifier and neural networks respectively.
- **API Development**: Implement FastAPI endpoints for making predictions and conducting experiments.
- **Experiments**: Conduct A/B experiments to compare user engagement with and without genre prediction, analyze session data, and understand user behavior.

## Technologies Used:
- **Machine Learning**: Scikit-learn, TensorFlow/Keras.
- **API Development**: FastAPI.
- **Data Handling**: Pandas.
- **Model Serialization**: Joblib, Pickle.
- **Experiment Analysis**: Pandas.

## Endpoints:
- **Base Model Prediction Endpoint**: Predicts the genre of a song using the baseline model.
- **Advanced Model Prediction Endpoint**: Predicts multiple genres for a song using the advanced neural network model.
- **Comparison Endpoint**: Compares predictions from the baseline and advanced models.
- **Accuracy Calculation Endpoint**: Calculates the accuracy of the baseline model.
- **A/B Experiment Endpoint**: Conducts A/B experiments to compare user engagement.


## How to Run:
To run the game, ensure you have Python installed along with the required libraries mentioned above. Then, execute the provided Python script (main.py) in your preferred Python environment.

## Future Enhancements: 
- bImplement more sophisticated neural network architectures for improved genre prediction.
- Incorporate user feedback and preferences into the prediction process.
- Enhance experiment analysis by including additional user metrics and A/B testing methodologies.
