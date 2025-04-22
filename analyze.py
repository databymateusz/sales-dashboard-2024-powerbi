import pandas as pd

df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

# 👁️‍🗨️ Podgląd danych
print(df.head())
print(df.info())
print(df.describe())

# 🔍 Sprawdzenie braków
print(df.isnull().sum())

# 🔁 Duplikaty
print(f"Duplikaty: {df.duplicated().sum()}")

# ❓ Dziwne wartości - zwroty
print(df['InvoiceNo'].str.contains('C').value_counts())

# 💥 Top 10 krajów, top produkty, dni tygodnia
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()

print(df['Country'].value_counts().head(10))
print(df['Description'].value_counts().head(10))
print(df['DayOfWeek'].value_counts())
