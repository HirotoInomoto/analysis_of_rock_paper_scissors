import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. データ処理 ----

# どちらのデータを使うか選ぶ
while True:
    flag = input("どちらのジャンケンを予測するかを入力してください（sunday/daily）: ")
    if flag == 'sunday' or flag == 'daily':
        break  
# flag = 'sunday' →元コード
file_path = f"./{flag}_janken.csv"
# file_path = "daily_janken_gender.csv"

# 特徴量としてgenderを足したとき
# RandomForest最適パラメータ: {'max_depth': 5, 'n_estimators': 200}
# RandomForestグリッドサーチ後の最高精度: 0.312
# dateモデルのテスト精度: 0.2236842105263158

data = pd.read_csv(file_path)

# 曜日を数字に変換
weekday_encoder = LabelEncoder()  # 例: ["Mon", "Tue", ...] → [0, 1, ...]
data["weekday_code"] = weekday_encoder.fit_transform(data["weekdays"])

# ジャンケンの手を数字に変換
result_encoder = LabelEncoder()   # 例: ["グー", "チョキ", "パー"] → [0, 1, 2]
data["answer_code"] = result_encoder.fit_transform(data["result"])

# gender_encoder = LabelEncoder()
# data['gender'] = gender_encoder.fit_transform(data['gender'])

# ---- 2. モデル学習 ----

input_data = data[["year", "month", "day", "weekday_code", "times"]]  # 入力データ
answer_data = data["answer_code"]  # 答え（ジャンケンの手）

# データを学習用とテスト用に分ける（8:2の割合）
input_train, input_test, answer_train, answer_test = train_test_split(input_data, answer_data, test_size=0.2, random_state=42)
date_model = RandomForestClassifier(random_state=42)
date_model.fit(input_train, answer_train)


# ---- 4. クロスバリデーションの準備 ----
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
cv = 5
# ---- 5. RandomForestのランダムサーチ ----
depth = list(range(1,20,2))
depth.insert(0,None)

param_random_rf = {
    'n_estimators': list(range(20,1000,5)),
    'max_depth': depth,
    'min_samples_split':list(range(2,11)),
    'min_samples_leaf':list(range(1,11)),
}
rf = RandomForestClassifier(random_state=42)
random_rf = RandomizedSearchCV(rf, param_random_rf,n_iter=10, cv=cv, scoring='accuracy',random_state=42)
random_rf.fit(input_data, answer_data)
print("RandomForest最適パラメータ:", random_rf.best_params_)
print("RandomForestランダムサーチ後の最高精度: {:.3f}".format(random_rf.best_score_))



# ---- 3. 予測関数 ----

# dateモデル：日付・曜日・回数を入れると、ジャンケンの手（グー/チョキ/パー）を返す。
def date_model_predict(year, month, day, times):
    # 日付から曜日を取得し、数字に変換
    date_obj = datetime(year, month, day)
    weekday_str = date_obj.strftime("%a")  # "Mon"などの形式に変換
    weekday_code = weekday_encoder.transform([weekday_str])[0]
    # 入力データを作り、予測
    model_input = pd.DataFrame([[year, month, day, weekday_code, times]],
                            columns=["year", "month", "day", "weekday_code", "times"])
    prediction_code = date_model.predict(model_input)[0]
    # 数字からジャンケンの手に戻す
    return result_encoder.inverse_transform([prediction_code])[0]

# ---- 4. 精度評価 ----
test_pred = date_model.predict(input_test)
accuracy = accuracy_score(answer_test, test_pred)
print("dateモデルのテスト精度:", accuracy)
