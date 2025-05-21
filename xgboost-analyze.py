import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from datetime import datetime

# ---- 1. データの読み込み ----
flag = 'mezamashi'  # "sazae"に変えると別データ
if flag == "mezamashi":
    file_path = "./result_mezamashi.csv"
elif flag == "sazae":
    file_path = "./result_sazae.csv"

df = pd.read_csv(file_path)

# ---- 2. データの前処理 ----
weekday_encoder = LabelEncoder()
df["weekday_code"] = weekday_encoder.fit_transform(df["weekdays"])
result_encoder = LabelEncoder()
df["result_code"] = result_encoder.fit_transform(df["result"])

# ---- 3. 特徴量・ラベルの定義 ----
X = df[["year", "month", "day", "weekday_code", "times"]]
y = df["result_code"]

# ---- 4. クロスバリデーションの準備 ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- 5. RandomForestのグリッドサーチ ----
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy')
grid_rf.fit(X, y)
print("RandomForest最適パラメータ:", grid_rf.best_params_)
print("RandomForestグリッドサーチ後の最高精度: {:.3f}".format(grid_rf.best_score_))

# ---- 6. XGBoostのグリッドサーチ ----
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring='accuracy')
grid_xgb.fit(X, y)
print("XGBoost最適パラメータ:", grid_xgb.best_params_)
print("XGBoostグリッドサーチ後の最高精度: {:.3f}".format(grid_xgb.best_score_))

# ---- 7. 予測関数（例: モデルA）----
def predict_by_date_with_model(model, year, month, day, times):
    date_obj = datetime(year, month, day)
    weekday_str = date_obj.strftime("%a")
    weekday_code = weekday_encoder.transform([weekday_str])[0]
    X_input = pd.DataFrame([[year, month, day, weekday_code, times]],
                            columns=["year", "month", "day", "weekday_code", "times"])
    pred_code = model.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

# ---- 8. 予測例 ----
print("RandomForestでの予測:", predict_by_date_with_model(grid_rf.best_estimator_, 2025, 5, 15, 1))
print("XGBoostでの予測:", predict_by_date_with_model(grid_xgb.best_estimator_, 2025, 5, 15, 1))
