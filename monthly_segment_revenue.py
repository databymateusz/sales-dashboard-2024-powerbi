import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Wczytanie oczyszczonych danych i ocen RFM
    clean = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'], low_memory=False)
    rfm   = pd.read_csv('data/rfm_scored.csv',    index_col='CustomerID')

    # 2. Połączenie clean_data z segmentami
    df = clean.merge(
        rfm['Segment'],
        left_on='CustomerID',
        right_index=True,
        how='left'
    )
    df['Segment'] = df['Segment'].fillna('Unknown')

    # 3. Ustawienie indeksu czasowego
    df.set_index('InvoiceDate', inplace=True)

    # 4. Obliczenie miesięcznego przychodu per segment
    monthly = (
        df
        .groupby([pd.Grouper(freq='M'), 'Segment'])['Revenue']
        .sum()
        .unstack(fill_value=0)
    )

    # 5. Zapis do CSV
    monthly.to_csv('data/monthly_segment_revenue.csv')

    # 6. Wygenerowanie i zapis wykresu
    plt.figure(figsize=(12,6))
    monthly.plot(title='Miesięczny przychód wg segmentów')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig('images/monthly_revenue_by_segment.png')
    plt.close()

    # 7. Podgląd wyników
    print("Pierwsze 5 wierszy miesięcznego przychodu per segment:")
    print(monthly.head())

if __name__ == '__main__':
    main()
