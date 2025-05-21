import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---- 1. データの読み込み ----
flag = 'mezamashi'  # または 'sazae'
if flag == "mezamashi":
    file_path = "./result_mezamashi.csv"
elif flag == "sazae":
    file_path = "./result_sazae.csv"

df = pd.read_csv(file_path, encoding='utf-8-sig')
df.columns = df.columns.str.strip()  # カラム名の前後空白除去

# ---- 2. データの前処理 ----
# 文字列カラムの値もすべてstrip
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# ジャンケンの手を数字に変換
result_encoder = LabelEncoder()
df["result_code"] = result_encoder.fit_transform(df["result"])

# ---- 3. モデルB用データ作成 ----
# 直近3回の手から次の手を予測
sequences = []
for i in range(len(df) - 3):
    row = df["result_code"].iloc[i:i+4].tolist()  # 4つ連続の手をリスト化
    sequences.append(row)

df_B = pd.DataFrame(sequences, columns=["prev1", "prev2", "prev3", "next"])
X_B = df_B[["prev1", "prev2", "prev3"]]
y_B = df_B["next"]

# ---- 4. クロスバリデーションとグリッドサーチ ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- RandomForest ---
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy')
grid_rf.fit(X_B, y_B)
print("RandomForest最適パラメータ:", grid_rf.best_params_)
print("RandomForestグリッドサーチ後の最高精度: {:.3f}".format(grid_rf.best_score_))

# --- XGBoost ---
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring='accuracy')
grid_xgb.fit(X_B, y_B)
print("XGBoost最適パラメータ:", grid_xgb.best_params_)
print("XGBoostグリッドサーチ後の最高精度: {:.3f}".format(grid_xgb.best_score_))

# ---- 5. 予測関数 ----
def predict_by_prev_moves(model, prev_moves):
    # prev_movesは 例: ["グー", "チョキ", "パー"]
    move_codes = result_encoder.transform(prev_moves)
    X_input = pd.DataFrame([move_codes], columns=["prev1", "prev2", "prev3"])
    pred_code = model.predict(X_input)[0]
    return result_encoder.inverse_transform([pred_code])[0]

# ---- 6. 予測例 ----
test_moves = ["グー", "チョキ", "パー"]
print(f"RandomForestでの予測: {predict_by_prev_moves(grid_rf.best_estimator_, test_moves)}")
print(f"XGBoostでの予測: {predict_by_prev_moves(grid_xgb.best_estimator_, test_moves)}")
