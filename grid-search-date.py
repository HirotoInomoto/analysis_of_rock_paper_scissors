import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from datetime import datetime

# ---- 1. データの読み込み ----
flag = input('flag入力')  # "sazae"に変えると別データ
file_path = f"{flag}_janken.csv"

df = pd.read_csv(file_path)

# ---- 2. データの前処理 ----
weekday_encoder = LabelEncoder()
df["weekday_code"] = weekday_encoder.fit_transform(df["weekdays"])
result_encoder = LabelEncoder()
df["result_code"] = result_encoder.fit_transform(df["result"])

# ---- 3. 特徴量・ラベルの定義 ----
X = df[["year", "month", "day", "weekday_code", "times"]]
y = df["result_code"]

input_train, input_test, answer_train,answer_test = train_test_split(X,y,test_size = 0.2,random_state=42)
date_model = RandomForestClassifier(random_state=42)
date_model.fit(input_train, answer_train)

# ---- 4. クロスバリデーションの準備 ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- 5. RandomForestのグリッドサーチ ----
param_grid_rf = {
    # 'n_estimators': [300,375,400,500],
    'n_estimators': [375,400,500,525],
    # 'max_depth': [12,15,18],
    'max_depth': [14,15,16],
}
rf = RandomForestClassifier(random_state=42,min_samples_leaf=4,min_samples_split=4)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy')
grid_rf.fit(X, y)
print("RandomForest最適パラメータ:", grid_rf.best_params_)
print("RandomForestグリッドサーチ後の最高精度: {:.3f}".format(grid_rf.best_score_))

# # ---- 6. XGBoostのグリッドサーチ ----
# param_grid_xgb = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
# }
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring='accuracy')
# grid_xgb.fit(X, y)
# print("XGBoost最適パラメータ:", grid_xgb.best_params_)
# print("XGBoostグリッドサーチ後の最高精度: {:.3f}".format(grid_xgb.best_score_))

# ---- 7. 予測関数（例: モデルA）----
def predict_by_date_with_model(model, year, month, day, times):
    date_obj = datetime(year, month, day)
    weekday_str = date_obj.strftime("%a")
    weekday_code = weekday_encoder.transform([weekday_str])[0]
    X_input = pd.DataFrame([[year, month, day, weekday_code, times]],
                            columns=["year", "month", "day", "weekday_code", "times"])
    pred_code = model.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

# ---- 4. 精度評価 ----
test_pred = date_model.predict(input_test)
accuracy = accuracy_score(answer_test, test_pred)
print("dateモデルのテスト精度:", accuracy)


# ---- 8. 予測例 ----
# print("RandomForestでの予測:", predict_by_date_with_model(grid_rf.best_estimator_, 2025, 5, 18, 1))
# print("XGBoostでの予測:", predict_by_date_with_model(grid_xgb.best_estimator_, 2025, 5, 18, 1))