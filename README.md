# Stock Market Prediction using Machine Learning

## Overview
This project focuses on predicting stock prices using three different models:
1. **Linear Regression** - A basic statistical method for modeling relationships between variables.
2. **LSTM (Long Short-Term Memory)** - A type of recurrent neural network (RNN) designed for sequential data.
3. **Improved LSTM** - An optimized version of LSTM with additional techniques to enhance prediction accuracy.

The implementation is structured in a Jupyter Notebook (`Stock_market.ipynb`), covering data preprocessing, model training, evaluation, and visualization of results.

## Dataset
The dataset used for this project consists of historical stock market prices, including features such as:
- Open price
- Close price
- High and low prices
- Volume traded
- Date/time information

## Models Implemented
### 1. Linear Regression
- Used as a baseline model.
- Predicts stock prices based on a linear relationship between features.
- Evaluated using Mean Squared Error (MSE) and R-squared score.

### 2. LSTM Model
- A deep learning model capable of learning patterns in time series data.
- Uses past stock prices as input to predict future prices.
- Implemented with TensorFlow/Keras.
- Evaluated using RMSE (Root Mean Squared Error).

### 3. Improved LSTM Model
- Enhances LSTM performance using additional techniques like:
  - More layers and units.
  - Dropout regularization to prevent overfitting.
  - Batch normalization.
- Provides better generalization and accuracy.

## Results
The three models are compared based on prediction accuracy and error metrics.

| Model | MSE | RMSE | 
|--------|--------|--------|
| Linear Regression | 08.13 | 00.28 |
| LSTM | 00.27 | 00.05 |
| Improved LSTM | 00.13 | 00.03 |

## Visualizations
The output graphs comparing actual vs. predicted prices for each model are stored in the `images/` folder. To add them, place the images in:
```
images/linear_regression_result.png
images/lstm_result.png
images/improved_lstm_result.png
```
Ensure these files exist before running the notebook to view the results.

## Requirements
To run the notebook, install the required dependencies:
```
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

## Usage
1. Open `Stock_market.ipynb` in Jupyter Notebook.
2. Run all cells to preprocess data, train models, and visualize predictions.
3. Review the saved output graphs in the `images/` folder.

## Future Improvements
- Experiment with different LSTM architectures.
- Use attention mechanisms for better time series forecasting.
- Integrate real-time stock market data for live predictions.

## Author
Atharv Jagtap

