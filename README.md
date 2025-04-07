```markdown
# Time Series Forecasting with TensorFlow

## Overview

This project demonstrates a comprehensive approach to time series forecasting using TensorFlow and Python. The code generates a synthetic time series with trend, seasonality, and noise, and then applies various forecasting methods to predict future values. The techniques include:

- **Naive Forecasting:** Uses the last observed value as the prediction.
- **Moving Average Forecasting:** Uses the average of the previous observations.
- **Differencing:** Removes trend and seasonality by differencing the time series.
- **Combination Approaches:** Incorporates smoothing and reintroduces trend/seasonality for improved forecasts.

The results are visualized using matplotlib, and forecast accuracy is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## License

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:  
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.

## Setup

- **TensorFlow Version:**  
  This project requires TensorFlow 2.x. The code includes a check to enforce TensorFlow 2.x, especially when running in a Google Colab environment.
  
- **Required Libraries:**  
  Ensure that the following Python libraries are installed:
  - `numpy`
  - `matplotlib`
  - `tensorflow` (which includes Keras)

You can install these dependencies using pip:

```bash
pip install numpy matplotlib tensorflow
```

## Data Description

The data in this project is synthetically generated to mimic a realistic time series with the following characteristics:

- **Time Series Length:**  
  The time series is generated over a span of 4 years plus one day (i.e., `4 * 365 + 1` days).

- **Components of the Series:**
  - **Trend:** A linear trend is added to the series using a slope of `0.09`.
  - **Seasonality:** A periodic seasonal component with a period of 365 days is incorporated. The seasonal pattern is defined by a custom function that uses a cosine function for the initial part of the cycle and an exponential decay for the remainder.
  - **Baseline:** A constant baseline value of `10` is added to the series.
  - **Noise:** Random Gaussian noise is added to simulate real-world data variability, using a noise level of `6` and a fixed seed (`42`) to ensure reproducibility.

- **Data Splitting:**  
  The synthetic time series is split into:
  - **Training Set:** The first 1000 data points.
  - **Validation Set:** The remaining data points beyond the 1000th time step.

This synthetic data allows us to test different forecasting strategies under controlled conditions where the underlying trend, seasonality, and noise are known.

## Code Breakdown

### 1. TensorFlow Version Check and Imports

The following code ensures that TensorFlow 2.x is used (especially important in Colab) and imports the required libraries.

```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
print(tf.__version__)
```

### 2. Generating the Time Series Data

This block defines functions for creating trends, seasonality, and noise. It then generates a synthetic time series that mimics real-world data patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# Create the series with trend, seasonality, and noise
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

### 3. Splitting the Data

The time series is split into a training set and a validation set. The training set is used to develop the forecasting model, and the validation set is used for evaluation.

```python
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()
```

### 4. Naive Forecasting

The simplest forecast uses the last observed value as the prediction for the next time step. This approach serves as a baseline.

```python
naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
```

### 5. Moving Average Forecast

A moving average forecast computes the mean of a window of past observations.

```python
def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
```

### 6. Differencing to Remove Trend and Seasonality

Differencing the series with a lag equal to the seasonality period helps remove trend and seasonal effects, resulting in a more stationary series.

```python
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()
```

### 7. Forecasting with the Differenced Series

A moving average forecast is applied to the differenced series. Then, the trend and seasonality are reintroduced by adding past values from the original series.

```python
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

# Adding back the trend and seasonality
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
```

### 8. Smoothing the Forecasts

To further reduce the noise in the forecast, a moving average is applied to the past values before reintroducing them.

```python
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
```

## Results and Evaluation

Each forecasting method is evaluated using two metrics:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

These metrics help compare the performance of the naive forecast against the improved methods (moving average and differencing-based approaches).

## How to Run the Project

1. **Clone or Download the Repository:**  
   Get the project files onto your local machine.

2. **Install Dependencies:**  
   Ensure you have Python 3.x installed along with TensorFlow, NumPy, and matplotlib. Use the following command to install the necessary packages:
   ```bash
   pip install numpy matplotlib tensorflow
   ```

3. **Run the Code:**  
   - **Jupyter Notebook:** Open the notebook file and run each cell sequentially.
   - **Python Script:** If you prefer, run the script from the command line using:
     ```bash
     python your_script_name.py
     ```
   Ensure that your environment supports graphical display for matplotlib plots, or adjust the code to save plots to files.

## Conclusion

This project serves as a hands-on introduction to time series forecasting using TensorFlow. By generating synthetic data and applying various forecasting techniques—from simple naive forecasts to more advanced methods involving differencing and smoothing—this example provides a solid foundation for understanding and building more complex forecasting models.

## Acknowledgements

This project is inspired by common time series analysis techniques and serves as an educational resource to help understand the fundamentals of forecasting with Python and TensorFlow.
```
