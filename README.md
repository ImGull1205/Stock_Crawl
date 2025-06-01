# Stock Data Collection and Prediction App

This application serves as a comprehensive tool for stock market data collection and prediction. It utilizes advanced machine learning algorithms to predict future stock prices based on historical data.

## Features
1. **Data Collection:** 
   The application is capable of collecting real-time stock market data, providing users with up-to-date information about their preferred stocks.

2.  **Stock Price Prediction:** 
   Using machine learning algorithms, the application can predict future stock prices, aiding users in making informed investment decisions.

3. **User-friendly Interface:** 
   The application features a user-friendly interface, making it easy for users to navigate and use the application's features.

4. **Company News Extension:** 
   This feature allows users to stay informed about the latest news related to their preferred companies. Users can install this as a Chrome extension for easy access.
   
## Installation

1. Navigate to the project directory:

    ```bash
    cd AIML-G1
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Stock price predict application

1. Run the application:

    ```bash
    streamlit run app.py
    ```

    This will start the web application on your local machine.

2. Access the application:

    Open your web browser and go to `http://localhost:8501` to access the stock price prediction app.

### Company news extension:

1. Installation:
    - Visit chrome://extensions
    - Enable Developer mode
    - Choose Load unpacked, navigate to the project folder and select the chrome-extension folder.
2. Usage
    - Click on the extension icon ![alt text](./chrome-extension/icons/icon16.png)
    - Then, a menu will display a bar to select the company code. Click on the company code you want to view.
