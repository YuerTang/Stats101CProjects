import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def splitting_data(df):
    train_data_list = []
    test_data_list = []

    df_model = df.sort_values(by='Team')  
    for team, group in df_model.groupby('Team'):
        train_size = int(len(group) * 0.7)
        train_data_list.append(group.iloc[:train_size])
        test_data_list.append(group.iloc[train_size:])

    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)

    X_train = train_data.drop(columns=['W/L', 'Team'])
    y_train = train_data['W/L']
    X_test = test_data.drop(columns=['W/L', 'Team'])
    y_test= test_data['W/L']
    return (X_train, y_train, X_test, y_test)


def lasso_regression(X_train, y_train,X_test):
    lasso_model = LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=10000)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(lasso_model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_lasso_model = grid.best_estimator_
    y_pred = best_lasso_model.predict(X_test)

    return (best_lasso_model,y_pred)
