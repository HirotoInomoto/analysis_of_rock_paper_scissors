# 環境構築

```shell-session
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# 実行

```shell-session
python3 analysis.py
```

# 実行結果

## report

| | precision | recall | f1-score | support |
| ---- | ---- | ---- | ---- | ---- |
| グー | 0.51 | 0.59 | 0.55 | 86 |
| チョキ | 0.37 | 0.36 | 0.36 | 108 |
| パー | 0.52 | 0.46 | 0.49 | 100 |
|  |  |  |  |  |
| accuracy |  |  | 0.46 | 294 |
| macro avg | 0.47 | 0.47 | 0.47 | 294 |
| weighted avg | 0.46 | 0.46 | 0.46 | 294 |

## predict_move_on_date

2025年5月11日を指定

```
グー
```

## predict_next_move_from_input

2025年4月20日、4月27日、5月4日の3日分のデータを入力した状態で実行

```
グー
```
