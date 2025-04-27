import pandas as pd

def main():
    # 1. Wczytanie ocenionej RFM (zawiera już R_quartile, F_quartile, M_quartile, RFM_Score, Segment)
    rfm = pd.read_csv('data/rfm_scored.csv', index_col='CustomerID')

    # 2. Obliczenie ARPU (Average Revenue Per User), jeśli nie było w pliku
    rfm['ARPU'] = rfm.apply(
        lambda row: row['Monetary'] / row['Frequency'] if row['Frequency'] > 0 else 0,
        axis=1
    )

    # 3. Wczytanie oczyszczonych danych z niską pamięcią
    clean = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'], low_memory=False)

    # 4. Obliczenie czasu (Tenure) dla każdego klienta
    cust_dates = (
        clean.groupby('CustomerID')['InvoiceDate']
             .agg(first_purchase='min', last_purchase='max')
    )
    cust_dates['TenureDays'] = (
        cust_dates['last_purchase'] - cust_dates['first_purchase']
    ).dt.days + 1

    # 5. Połączenie RFM + Tenure
    df = rfm.merge(
        cust_dates['TenureDays'],
        left_index=True, right_index=True, how='left'
    )

    # 6. Estymacja transakcji rocznych i CLV
    df['TransPerDay'] = df['Frequency'] / df['TenureDays']
    df['EstAnnualFreq'] = df['TransPerDay'] * 365
    df['CLV_Annual'] = df['ARPU'] * df['EstAnnualFreq']

    # 7. Podsumowanie CLV per segment
    seg_clv = df.groupby('Segment').agg(
        Count=('CLV_Annual', 'size'),
        Avg_CLV_Annual=('CLV_Annual', 'mean'),
        Median_CLV_Annual=('CLV_Annual', 'median')
    )
    print("CLV per segment:")
    print(seg_clv)

    # 8. Zapis wyników
    seg_clv.to_csv('data/segment_clv_summary.csv')
    df.to_csv('data/customer_estimated_clv.csv')

if __name__ == '__main__':
    main()
