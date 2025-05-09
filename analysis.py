import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CSVファイルの読み込み
file_path = "./result.csv"
df = pd.read_csv(file_path)

# # データの先頭を表示して構造を確認
# print(df.head())

# 日付を datetime に変換
df["date"] = pd.to_datetime(df[["year", "month", "day"]])

# 手を数値にエンコード（グー=0, チョキ=1, パー=2）
# 授業で実際に使う場合、ここは自力実装の方が望ましいと思う
label_encoder = LabelEncoder()
df = df[df["hand"].isin(["グー", "チョキ", "パー"])].copy()
df["hand_encoded"] = label_encoder.fit_transform(df["hand"])

# 直近n回の履歴を特徴量にするためのデータ整形関数
def create_features(data, n_lag=3):
    features = []
    targets = []
    dates = []
    for i in range(n_lag, len(data)):
        features.append(data["hand_encoded"].values[i - n_lag:i])
        targets.append(data["hand_encoded"].values[i])
        dates.append(data["date"].values[i])
    return np.array(features), np.array(targets), np.array(dates)

# 特徴量とターゲットの作成
X, y, date_list = create_features(df, n_lag=3)

# 学習用・検証用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの学習
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 精度評価
y_pred = model.predict(X_test)
# report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
report = classification_report(y_test, y_pred, labels=[0,1,2], target_names=label_encoder.classes_)

print("=== report ===")
print(report)
print("==============")
print()

# 指定日付までのデータから予測する関数
def predict_move_on_date(target_date_str, n=3):
    try:
        target_date = pd.to_datetime(target_date_str)
    except:
        return "日付の形式が正しくありません（例：2000-01-01）"
    
    subset = df[df["date"] < target_date]
    if len(subset) < n:
        return "指定日までの履歴が不足しています"
    
    recent_moves = subset["hand_encoded"].values[-n:]
    prediction = model.predict([recent_moves])[0]
    return label_encoder.inverse_transform([prediction])[0]

example = predict_move_on_date("2025-05-11")
print("=== predict_move_on_date ===")
print(example)
print("============================")
print()

# 入力として手のリストを受け取り、その次の手を予測する関数に修正
def predict_next_move_from_input(hand_list):
    """
    hand_list: 直近の手のリスト（例: ["グー", "チョキ", "パー"]）
    """
    if len(hand_list) == 0:
        return "手の入力が空です"

    try:
        encoded = label_encoder.transform(hand_list)
    except ValueError as e:
        return f"無効な手が含まれています: {e}"

    if len(encoded) != model.n_features_in_:
        return f"{model.n_features_in_} 回分の手を入力してください"

    prediction = model.predict([encoded])[0]
    return label_encoder.inverse_transform([prediction])[0]

# テスト例
result2 = predict_next_move_from_input(["パー", "パー", "チョキ"])

print("=== predict_next_move_from_input ===")
print(result2)
print("====================================")
