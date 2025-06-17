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
file_path = "daily_janken_gender.csv"

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

gender_encoder = LabelEncoder()
data['gender'] = gender_encoder.fit_transform(data['gender'])

# ---- 2. モデル学習 ----

input_data = data[["year", "month", "day", "weekday_code", "times","gender"]]  # 入力データ
answer_data = data["answer_code"]  # 答え（ジャンケンの手）

# データを学習用とテスト用に分ける（8:2の割合）
input_train, input_test, answer_train, answer_test = train_test_split(input_data, answer_data, test_size=0.2, random_state=42)
date_model = RandomForestClassifier(random_state=42)
date_model.fit(input_train, answer_train)

# ---- 4. クロスバリデーションの準備 ----
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
cv = 5
# ---- 5. RandomForestのランダム・グリッドサーチ ----
depth = list(range(1,20,2))

param_random_rf = {
    'n_estimators': list(range(20,1000,5)),
    'max_depth': depth.append(None),
    'min_samples_split':list(range(2,11)),
    'min_samples_leaf':list(range(1,11)),
}
# param_grid_rf = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None,5, 10, 20],
#     'min_samples_split':[2,5,10],
#     'min_samples_leaf':[1,2,4],
# }
rf = RandomForestClassifier(random_state=42)
random_rf = RandomizedSearchCV(rf, param_random_rf,n_iter=100, cv=cv, scoring='accuracy',random_state=42)
random_rf.fit(input_data, answer_data)
print("RandomForest最適パラメータ:", random_rf.best_params_)
print("RandomForestグリッドサーチ後の最高精度: {:.3f}".format(random_rf.best_score_))
# grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy')
# grid_rf.fit(input_data, answer_data)
# print("RandomForest最適パラメータ:", grid_rf.best_params_)
# print("RandomForestグリッドサーチ後の最高精度: {:.3f}".format(grid_rf.best_score_))

#特徴量の重要度
feature = date_model.feature_importances_

#特徴量の名前
label = data.columns[0:]

#特徴量の重要度順（降順）
# indices = np.argsort(feature)[::-1]

# for i in range(len(feature)):
#     print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

# plt.title('Feature Importance')
# plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
# plt.xticks(range(len(feature)), label[indices], rotation=90)
# plt.xlim([-1, len(feature)])
# plt.tight_layout()
# plt.show()


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

#---- 5.実行 ---

# # ここのエラーハンドリングをすべきか
# year = int(input("年を入力してください: "))
# month = int(input("月を入力してください: "))
# day = int(input("日を入力してください: "))
# # ここの日付エラーハンドリングは注釈で載せて置き、それを授業では高度であるので触れないことを教師用メモとして書いておく


# if flag == "sunday":
#     print(f"日曜日のジャンケンの{year}-{month}-{day} の予測:", date_model_predict(year, month, day, 1))
# elif flag == "daily":
#     times = int(input("何回目のジャンケンかを入力して下さい: "))
#     print(f"毎週のジャンケンの{year}-{month}-{day} の{times}回目の予測:", date_model_predict(year, month, day, times))
    