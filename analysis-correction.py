import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# ---- 1. データの読み込み ----
# どちらのデータを使うか選びます
# flag = "sazae"  # "mezamashi"にすると別のデータ
flag = 'sazae'

if flag == "mezamashi":
    file_path = "./result_mezamashi.csv"
elif flag == "sazae":
    file_path = "./result_sazae.csv"

df = pd.read_csv(file_path)

# ---- 2. データの前処理 ----
# 機械学習では文字（例:「グー」や「月曜日」）を数字に直す必要があります

# 曜日を数字に変換します
weekday_encoder = LabelEncoder()  # 例: ["Mon", "Tue", ...] → [0, 1, ...]
df["weekday_code"] = weekday_encoder.fit_transform(df["weekdays"])

# ジャンケンの手も数字に変換します
result_encoder = LabelEncoder()   # 例: ["グー", "チョキ", "パー"] → [0, 1, 2]
df["result_code"] = result_encoder.fit_transform(df["result"])

# ---- 3. モデルの学習 ----

# 【モデルA】日付・曜日・回数からジャンケンの手を予測
X_A = df[["year", "month", "day", "weekday_code", "times"]]  # 入力データ
y_A = df["result_code"]                                      # 答え
# データを学習用とテスト用に分けます（8:2の割合）
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
model_A = RandomForestClassifier(random_state=42)
model_A.fit(X_A_train, y_A_train)

# 【モデルB】直近3回の手から次の手を予測
# 4つ連続した手を使い、「最初の3つ」を入力に、「4つ目」を答えにします
sequences = []
for i in range(len(df) - 3):
    row = df["result_code"].iloc[i:i+4].tolist()
    sequences.append(row)

df_B = pd.DataFrame(sequences, columns=["prev1", "prev2", "prev3", "next"])
X_B = df_B[["prev1", "prev2", "prev3"]]
y_B = df_B["next"]
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

model_B = RandomForestClassifier(random_state=42)
model_B.fit(X_B_train, y_B_train)

# ---- 4. 予測のための関数 ----

# モデルA：日付・曜日・回数を入れると、ジャンケンの手（グー/チョキ/パー）を返します
def predict_by_date(year, month, day, times):
    # 日付から曜日を取得し、数字に変換
    date_obj = datetime(year, month, day)
    weekday_str = date_obj.strftime("%a")  # "Mon"など
    weekday_code = weekday_encoder.transform([weekday_str])[0]
    # 入力データを作り、予測
    X_input = pd.DataFrame([[year, month, day, weekday_code, times]],
                            columns=["year", "month", "day", "weekday_code", "times"])
    pred_code = model_A.predict(X_input)[0]
    # 数字からジャンケンの手に戻す
    return result_encoder.inverse_transform([pred_code])[0]

# モデルB：直近3回の手を入れると、次の手を予測します
def predict_by_prev_moves(prev_moves):
    # 文字（例："グー"）を数字に変換
    move_codes = result_encoder.transform(prev_moves)
    X_input = pd.DataFrame([move_codes], columns=["prev1", "prev2", "prev3"])
    pred_code = model_B.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

# ---- 5. 予測を使ってみる＋テスト精度の確認 ----
if flag == "mezamashi":
    print("モデルAによる予測:", predict_by_date(2025, 5, 15, 1))
    # モデルAのテスト精度
    y_A_pred = model_A.predict(X_A_test)
    accuracy_A = accuracy_score(y_A_test, y_A_pred)
    print("モデルAのテスト精度:", accuracy_A)

    print("モデルBによる予測:", predict_by_prev_moves(["グー", "チョキ", "パー"]))
    y_B_pred = model_B.predict(X_B_test)
    accuracy_B = accuracy_score(y_B_test, y_B_pred)
    print("モデルBのテスト精度:", accuracy_B)

elif flag == "sazae":
    print("モデルAによる予測:", predict_by_date(2025, 5, 11, 1))
    y_A_pred = model_A.predict(X_A_test)
    accuracy_A = accuracy_score(y_A_test, y_A_pred)
    print("モデルAのテスト精度:", accuracy_A)

    # print("モデルBによる予測:", predict_by_prev_moves(["パー", "パー", "チョキ"]))
    y_B_pred = model_B.predict(X_B_test)
    accuracy_B = accuracy_score(y_B_test, y_B_pred)
    print("モデルBのテスト精度:", accuracy_B)