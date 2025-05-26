import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# ---- 1. データ処理 ----

# どちらのデータを使うか選ぶ
flag = 'sazae'

if flag == "mezamashi":
    file_path = "./result_mezamashi.csv"
elif flag == "sazae":
    file_path = "./result_sazae.csv"

data = pd.read_csv(file_path)

# 曜日を数字に変換
weekday_encoder = LabelEncoder()  # 例: ["Mon", "Tue", ...] → [0, 1, ...]
data["weekday_code"] = weekday_encoder.fit_transform(data["weekdays"])

# ジャンケンの手を数字に変換
result_encoder = LabelEncoder()   # 例: ["グー", "チョキ", "パー"] → [0, 1, 2]
data["answer_code"] = result_encoder.fit_transform(data["result"])

# ---- 2. モデル学習 ----

# 【モデル】日付・曜日・回数からジャンケンの手を予測
input = data[["year", "month", "day", "weekday_code", "times"]]  # 入力データ
answer = data["answer_code"]  # 答え（じゃんけんの手）
# データを学習用とテスト用に分ける（8:2の割合）
input_train, input_test, answer_train, answer_test = train_test_split(input, answer, test_size=0.2, random_state=20)
date_model = RandomForestClassifier(random_state=20)
date_model.fit(input_train, answer_train)

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
print("モデルのテスト精度:", accuracy)

#---- 5.実行 ---
print("dateモデルによる予測:", date_model_predict(2025, 6, 1, 1))