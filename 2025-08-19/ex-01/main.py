import pandas as pd

def main():
    
    data_path = "datasets\hourly_energy_consumption\AEP_hourly.csv"
    
    data = pd.read_csv(data_path)

    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

    daily_mean = data['AEP_MW'].resample('D').mean()
    weekly_mean = data['AEP_MW'].resample('W').mean()
    global_mean = data['AEP_MW'].mean()

    data['Daily_mean'] = daily_mean.reindex(data.index, method='ffill')
    data['Weekly_mean'] = weekly_mean.reindex(data.index, method='ffill')
    data['Global_mean'] = global_mean

    data['consumption_day'] = (data['AEP_MW'] > data['Daily_mean']).map({True: 'high', False: 'low'})
    data['consumption_week'] = (data['AEP_MW'] > data['Weekly_mean']).map({True: 'high', False: 'low'})
    data['consumption_global'] = (data['AEP_MW'] > data['Global_mean']).map({True: 'high', False: 'low'})
    
    print(data.head(10))

if __name__ == "__main__":
    main()