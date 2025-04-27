import pandas as pd

# 1. Wczytanie danych
df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

# 2. Usunięcie zwrotów (InvoiceNo zaczyna się na 'C') i anomalii (ujemne wartości)
df = df[~df['InvoiceNo'].str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 3. Uzupełnienie brakujących CustomerID
df['CustomerID'] = df['CustomerID'].fillna(0).astype(int).astype(str).replace('0', 'Unknown')

# 4. Dodanie kolumn obliczeniowych
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour

# 5. Zapis oczyszczonych danych (opcjonalnie)
df.to_csv('data/clean_data.csv', index=False)

# 6. Podgląd wyników
print("Rekordy po czyszczeniu:", len(df))
print(df.head())
