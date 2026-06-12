import pandas as pd

txn = pd.read_csv("data/dataset/train_transaction.csv")
identity = pd.read_csv("data/dataset/train_identity.csv")

print("train_transaction")
missing = txn.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    print(f"{col:<25s}  {count:>8,}  ({count / len(txn):>6.1%})")

print("train_identity")
missing_id = identity.isnull().sum()
missing_id = missing_id[missing_id > 0].sort_values(ascending=False)
for col, count in missing_id.items():
    print(f"  {col:<25s}  {count:>8,}  ({count / len(identity):>6.1%})")
