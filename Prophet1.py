import future
from prophet import Prophet
from sklearn.model_selection import train_test_split

def get_prophet_predict(data, num):
    model = Prophet()
    
    data = data.rename_axis('Time').reset_index()
    df = data[['Time','Close']]
    
    rename_dict = {
        "Time": "ds",
        "Close": "y"
    }
    db = df.rename(columns=rename_dict)
    x_train, x_test= train_test_split(db, shuffle = False, test_size = .02)
    
    model.fit(x_train)
    future = model.make_future_dataframe(periods=(num+len(x_test)))
    forecast = model.predict(future)
    return x_test['y'].array, forecast['yhat'].iloc[len(x_train):(len(x_train)+len(x_test))].array, model, forecast['yhat'].iloc[(len(x_train)+len(x_test)):(len(db)+5)].array
    