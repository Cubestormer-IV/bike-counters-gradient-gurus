# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-12-10T17:02:53.974988Z","iopub.execute_input":"2023-12-10T17:02:53.975465Z","iopub.status.idle":"2023-12-10T17:03:14.208377Z","shell.execute_reply.started":"2023-12-10T17:02:53.975423Z","shell.execute_reply":"2023-12-10T17:03:14.207352Z"}}
import subprocess
subprocess.run(["pip", "install", "workalendar"])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from workalendar.europe import France

df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")


def split_date(data):
    
    # Get integer values of hours, day, month and year from the date column
    data = data.copy()
    
    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.day
    data["weekday"] = data["date"].dt.weekday
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    
    # Mark the days which are bank holidays in France
    cal = France()
    data['bank_holiday'] = data['date'].apply(lambda x: cal.is_holiday(x))

    # Drop date
    return data.drop(columns=["date"])


def merge_weather_data(data):
    file_path = "../input/mdsb-2023/external_data.csv"
    weather_data = pd.read_csv(file_path, parse_dates=["date"])

    data['date'] = pd.to_datetime(data['date']).astype('datetime64[ns]')
    weather_data['date'] = pd.to_datetime(weather_data['date']).astype('datetime64[ns]')

    data = data.copy()

    # Merge weather data and save information on original order of tuples
    data["sorting_index"] = np.arange(data.shape[0])
    data = pd.merge_asof(
        data.sort_values("date"), 
        weather_data[["date", "t", 'u', 'ff', 'td', 'raf10', 'rafper']].sort_values("date"), 
        on="date"
    )

    # Sort the dataframe to the original order
    data = data.sort_values("sorting_index")
    del data["sorting_index"]
    return data


# Get training data
df_train = merge_weather_data(df_train)
date_values = split_date(df_train[["date"]])
df_train = pd.concat([df_train, date_values], axis=1)

# Get testing data
df_test = merge_weather_data(df_test)
test_date_values = split_date(df_test[["date"]])
df_test = pd.concat([df_test, test_date_values], axis=1)

# Split X and y from training data
y_train = df_train['log_bike_count']
X_train = df_train.drop('log_bike_count', axis=1)

# Specify column to encode and passthrough
categorical_encoder = OrdinalEncoder()
categorical_cols = ["counter_name", "bank_holiday"]
passthrough_cols = ['latitude', 'longitude', 't', 'u', 'ff', 'td', 'raf10', 'rafper', 'month', 'weekday', 'hour', 'day']

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_encoder, categorical_cols),
        ('passthrough', 'passthrough', passthrough_cols)
    ]
)

# Use parameters obtained from the GridSearch to initialise XGBRegressor
regressor = XGBRegressor(
    learning_rate=0.1, 
    max_depth=8, 
    n_estimators=500,
    min_child_weight=5, 
    reg_alpha=0.1, 
    colsample_bytree=0.8
)


# param_grid = {
#     'xgbregressor__learning_rate': [0.1],
#     'xgbregressor__max_depth': [8],
#     'xgbregressor__n_estimators': [100],
#     'xgbregressor__gamma': [0],
#     'xgbregressor__min_child_weight': [5],
#     'xgbregressor__reg_alpha': [0.1, 0.15, 0.2],
#     'xgbregressor__colsample_bytree': [0.8, 0.9, 1],
# }

# # Create the GridSearchCV object
# grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best model from the grid search
# best_model = grid_search.best_estimator_

# # Print the best hyperparameters
# print("Best Hyperparameters:", grid_search.best_params_)

# # Make predictions with the best model
# y_pred = best_model.predict(df_test)


# Create pipeline and train
pipeline = make_pipeline(preprocessor, regressor)
pipeline.fit(X_train, y_train)

# Predict and save result
y_pred = pipeline.predict(df_test)


final_pred = pd.DataFrame(dict(Id=np.arange(y_pred.shape[0]), log_bike_count=y_pred))
final_pred.to_csv("submission.csv", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-10T16:59:21.749333Z","iopub.execute_input":"2023-12-10T16:59:21.749737Z","iopub.status.idle":"2023-12-10T16:59:35.095914Z","shell.execute_reply.started":"2023-12-10T16:59:21.749698Z","shell.execute_reply":"2023-12-10T16:59:35.094682Z"},"jupyter":{"outputs_hidden":false}}
# !pip install workalendar

# %% [code] {"jupyter":{"outputs_hidden":false}}
