import pandas as pd

def main():
    # 1. Wczytanie danych RFM
    rfm = pd.read_csv('data/rfm.csv', index_col='CustomerID')

    # 2. Obliczenie rankingu procentowego (pct=True)
    rfm['R_rank'] = rfm['Recency'].rank(method='max', pct=True)
    rfm['F_rank'] = rfm['Frequency'].rank(method='max', pct=True)
    rfm['M_rank'] = rfm['Monetary'].rank(method='max', pct=True)

    # 3. Przypisanie kwartyli na podstawie rankingu
    # Recency: im mniejszy pct, tym wyższy kwartyl (4→1)
    rfm['R_quartile'] = rfm['R_rank'].apply(
        lambda x: 4 if x <= 0.25 else 3 if x <= 0.50 else 2 if x <= 0.75 else 1
    ).astype(int)
    # Frequency i Monetary: im większy pct, tym wyższy kwartyl (1→4)
    rfm['F_quartile'] = rfm['F_rank'].apply(
        lambda x: 1 if x <= 0.25 else 2 if x <= 0.50 else 3 if x <= 0.75 else 4
    ).astype(int)
    rfm['M_quartile'] = rfm['M_rank'].apply(
        lambda x: 1 if x <= 0.25 else 2 if x <= 0.50 else 3 if x <= 0.75 else 4
    ).astype(int)

    # 4. RFM Score jako suma kwartyli
    rfm['RFM_Score'] = rfm['R_quartile'] + rfm['F_quartile'] + rfm['M_quartile']

    # 5. Przypisanie segmentów
    def assign_segment(row):
        if row.R_quartile == 4 and row.F_quartile == 4 and row.M_quartile == 4:
            return 'Champions'
        if row.F_quartile >= 3 and row.M_quartile >= 3:
            return 'Loyal'
        if row.R_quartile in (2, 3) and row.F_quartile <= 2 and row.M_quartile <= 2:
            return 'Need Attention'
        if row.R_quartile <= 2 and row.F_quartile >= 3 and row.M_quartile >= 3:
            return 'At Risk'
        if row.R_quartile == 1 and row.F_quartile == 1 and row.M_quartile == 1:
            return 'Hibernating'
        return 'Others'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    # 6. Zapis zbiorczy i plików per segment
    rfm.to_csv('data/rfm_scored.csv', index=True)
    for seg in rfm['Segment'].unique():
        safe = seg.replace(' ', '_')
        rfm[rfm['Segment'] == seg].to_csv(f'data/segment_{safe}.csv', index=True)

    # 7. Podsumowanie
    print("Liczebność segmentów:")
    print(rfm['Segment'].value_counts())

if __name__ == '__main__':
    main()
