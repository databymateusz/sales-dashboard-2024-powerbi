import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Wczytanie i oczyszczenie danych
    df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = (
        df['CustomerID']
        .fillna(0).astype(int).astype(str)
        .replace('0', 'Unknown')
    )

    # 2. Inżynieria cech
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour

    # 3. Zapis oczyszczonych danych
    df.to_csv('data/clean_data.csv', index=False)
    print(f'Records after cleaning: {len(df)}')

    # 4. Statystyki opisowe
    print('\nRevenue stats:')
    print(df['Revenue'].agg(['min','max','mean','median','std']))

    # 5. Trendy czasowe
    monthly = df.resample('M', on='InvoiceDate')['Revenue'].sum()
    plt.figure(figsize=(10,4))
    monthly.plot(title='Miesięczny przychód')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig('images/monthly_revenue.png')
    plt.close()

    # 6. Przychód wg dnia tygodnia
    by_day = (
        df.groupby('DayOfWeek')['Revenue']
          .sum()
          .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    )
    plt.figure(figsize=(8,4))
    by_day.plot.bar(title='Przychód wg dnia tygodnia')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig('images/revenue_by_day.png')
    plt.close()

    # 7. Top listy
    top_prod = df.groupby('Description')['Revenue'].sum().nlargest(10)
    print('\nTop 10 produktów wg przychodu:')
    print(top_prod)

    top_cntry = df.groupby('Country')['Revenue'].sum().nlargest(10)
    print('\nTop 10 krajów wg przychodu:')
    print(top_cntry)

    # 8. Korelacje i wykres price vs quantity
    corr = df[['UnitPrice','Quantity']].corr().iloc[0,1]
    print(f'\nKorelacja UnitPrice vs Quantity: {corr:.2f}')

    plt.figure(figsize=(6,6))
    plt.scatter(df['UnitPrice'], df['Quantity'], alpha=0.1)
    plt.title('Cena jednostkowa vs ilość')
    plt.xlabel('UnitPrice')
    plt.ylabel('Quantity')
    plt.tight_layout()
    plt.savefig('images/price_vs_quantity.png')
    plt.close()

    # 9. Segmentacja RFM
    now = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    }).rename(columns={
        'InvoiceDate':'Recency',
        'InvoiceNo':'Frequency',
        'Revenue':'Monetary'
    })
    rfm.to_csv('data/rfm.csv', index=True)
    print('\nRFM (pierwsze 5 wierszy):')
    print(rfm.head())

if __name__ == '__main__':
    main()
