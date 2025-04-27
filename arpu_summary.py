import pandas as pd

def main():
    # 1. Wczytanie ocenionej RFM
    rfm = pd.read_csv('data/rfm_scored.csv', index_col='CustomerID')

    # 2. Obliczenie ARPU
    # Uwaga: jeśli Frequency==0, wtedy ARPU ustawiamy na 0, by uniknąć dzielenia przez 0
    rfm['ARPU'] = rfm.apply(
        lambda row: row['Monetary'] / row['Frequency'] 
                    if row['Frequency'] > 0 else 0,
        axis=1
    )

    # 3. Grupowanie po segmencie i agregacja
    summary = rfm.groupby('Segment').agg(
        Count=('ARPU', 'size'),
        Avg_ARPU=('ARPU', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Avg_RFM_Score=('RFM_Score', 'mean')
    )

    # 4. Wypisanie i zapis
    print(summary)
    summary.to_csv('data/segment_summary.csv')
    print("\nZapisano podsumowanie do data/segment_summary.csv")

if __name__ == '__main__':
    main()
