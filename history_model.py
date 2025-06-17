import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---- 1. データの読み込み ----
# どちらのデータを使うか選ぶ
while True:
    flag = input("どちらのジャンケンを予測するかを入力してください（sunday/daily）: ")
    if flag == 'sunday' or flag == 'daily':
        break  
file_path = f'{flag}_janken.csv'

data = pd.read_csv(file_path)

# ジャンケンの手を数字に変換
result_encoder = LabelEncoder()   # 例: ["グー", "チョキ", "パー"] → [0, 1, 2]
data["answer_code"] = result_encoder.fit_transform(data["result"])

# 【historyモデル】直近3回の手から次の手を予測
# 4つ連続した手を使い、「最初の3つ」を入力に、「4つ目」を答えにします
four_hands = []
for i in range(len(data) - 3):
    row = data["answer_code"].iloc[i:i+4].tolist()
    four_hands.append(row)

hands_data = pd.DataFrame(four_hands, columns=["prev1", "prev2", "prev3", "answer"])
# ---- 2. モデル学習 ----
input_data = hands_data[["prev1", "prev2", "prev3"]]
answer_data = hands_data["answer"]
input_train, input_test, answer_train, answer_test = train_test_split(input_data, answer_data, test_size=0.2, random_state=42)

history_model = RandomForestClassifier(random_state=42)
history_model.fit(input_train, answer_train)

# ---- 3 予測のための関数 ----

# historyモデル：直近3回の手を入れると、次の手を予測します
def history_model_predict(prev_moves):
    # 文字（例："グー"）を数字に変換
    move_codes = result_encoder.transform(prev_moves)
    # 入力データを作り、予測
    model_input = pd.DataFrame([move_codes], columns=["prev1", "prev2", "prev3"])
    prediction_code = history_model.predict(model_input)[0]
    # 数字からジャンケンの手に戻す
    return result_encoder.inverse_transform([prediction_code])[0]

# ---- 4. 精度評価 ----
test_pred = history_model.predict(input_test)
accuracy = accuracy_score(answer_test, test_pred)
print("historyモデルのテスト精度:", accuracy)

#---- 5.実行 ---
while True:
    print('グー/チョキ/パー で入力して下さい')
    prev1 = input('1回前のジャンケンの手')
    prev2 = input('2回前のジャンケンの手')
    prev3 = input('3回前のジャンケンの手')

    valid_hands = ["グー","チョキ","パー"]

    prev1_ok = prev1 in valid_hands
    prev2_ok = prev2 in valid_hands
    prev3_ok = prev3 in valid_hands

    if prev1_ok and prev2_ok and prev3_ok:
        break

print("historyモデルによる次回のジャンケンの手の予測:", history_model_predict([prev1,prev2,prev3]))