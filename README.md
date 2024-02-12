# Music-Genre-Classification

![image](https://github.com/RoboRopuch/Music-Genre-Classification/assets/128647614/c9fbb776-e47d-4ede-8a44-d40ca0903312)


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
To execute the program, ensure that you have Python installed, along with the necessary libraries listed in the requirements.txt file. The next step is to follow the instructions in przgotowanie_danych.ipynb and modele.ipynb notebooks. Upon completion of these steps, serialized models should be generated in the artifacts folder. You can then run the microservice by executing the command "uvicorn main:app --reload" from the services folder.

## Future Enhancements: 
- bImplement more sophisticated neural network architectures for improved genre prediction.
- Incorporate user feedback and preferences into the prediction process.
- Enhance experiment analysis by including additional user metrics and A/B testing methodologies.
