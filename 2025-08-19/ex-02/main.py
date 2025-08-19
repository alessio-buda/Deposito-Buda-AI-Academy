import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():

    # load data
    data_path = "C:\\Users\\XZ374JM\\OneDrive - EY\\Desktop\\AI Academy\\Deposito-Buda-AI-Academy\\datasets\\air_quality\\AirQualityUCI.csv"
    data = pd.read_csv(data_path, sep=';')
    
    # data cleaning
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    for col in data.columns:
        data[col] = data[col].replace(-200, None)  # replace -200 with NaN
        if col not in ['Date', 'Time']:
            data[col] = data[col].astype(float)
    data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
    data.dropna()
    
    # compute daily mean
    data.set_index('Date', inplace=True)
    daily_mean = data['CO(GT)'].resample('D').mean()
    data['Daily_mean'] = daily_mean.reindex(data.index, method='ffill')
    
    data['CO_target'] = (data['CO(GT)'] > data['Daily_mean'])
    
    data.reset_index(inplace=True, drop=True)
    X = data.drop(['CO_target', 'Time'], axis=1)
    y = data['CO_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    
    y_pred = decision_tree.predict(X_test)
    
    print(classification_report(y_test, y_pred, target_names=['Low CO', 'High CO']))
    
    plt.figure(figsize=(18, 10))
    plot_tree(decision_tree, feature_names=X.columns, class_names=['Low CO', 'High CO'], filled=True)
    plt.title("Decision Tree - Dataset AEP")
    plt.show()
    
    
if __name__ == "__main__":
    main()
