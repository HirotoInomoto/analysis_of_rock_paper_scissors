import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime

# flag = "mezamashi"
flag = "sazae"

# CSVファイルを読み込み
if flag == "mezamashi":
    file_path = "./result_mezamashi.csv"
elif flag == "sazae":
    file_path = "./result_sazae.csv"
df = pd.read_csv(file_path)

# 曜日を数値に変換
weekday_encoder = LabelEncoder()
df["weekday_code"] = weekday_encoder.fit_transform(df["weekdays"])

# 手（グー, チョキ, パー）を数値に変換
result_encoder = LabelEncoder()
df["result_code"] = result_encoder.fit_transform(df["result"])

# --- モデルA（日付・曜日・回数から予測） ---
X_A = df[["year", "month", "day", "weekday_code", "times"]]
y_A = df["result_code"]

# 訓練・テスト分割と学習
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
model_A = RandomForestClassifier(random_state=42)
model_A.fit(X_A_train, y_A_train)

# --- モデルB（直近3手から次の手を予測） ---
# 連続した4手のセットを作成
sequences = []
for i in range(len(df) - 3):
    prev_moves = df["result_code"].iloc[i:i+3].tolist()
    next_move = df["result_code"].iloc[i+3]
    sequences.append(prev_moves + [next_move])

df_B = pd.DataFrame(sequences, columns=["prev1", "prev2", "prev3", "next"])
X_B = df_B[["prev1", "prev2", "prev3"]]
y_B = df_B["next"]

# 学習
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)
model_B = RandomForestClassifier(random_state=42)
model_B.fit(X_B_train, y_B_train)

# 両モデルの予測関数を定義
def predict_by_date(year, month, day, times):
    date_obj = datetime(year, month, day)
    weekday_str = date_obj.strftime("%a")
    weekday_code = weekday_encoder.transform([weekday_str])[0]

    # 列名を明示的に指定
    X_input = pd.DataFrame(
        [
            [year, month, day, weekday_code, times]
        ],
        columns=["year", "month", "day", "weekday_code", "times"]
    )
    pred_code = model_A.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

def predict_by_prev_moves(prev_moves):
    move_codes = result_encoder.transform(prev_moves)
    
    # DataFrameで列名を明示
    X_input = pd.DataFrame(
        [move_codes],
        columns=["prev1", "prev2", "prev3"]
    )
    pred_code = model_B.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

if flag == "mezamashi":
    print(predict_by_date(2025, 5, 15, 1))
    print(predict_by_prev_moves(["グー", "チョキ", "パー"]))
elif flag == "sazae":
    print(predict_by_date(2025, 5, 11, 1))
    print(predict_by_prev_moves(["パー", "パー", "チョキ"]))