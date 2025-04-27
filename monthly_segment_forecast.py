import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def main():
    # 0. Parametry
    horizon = 6  # liczba prognozowanych miesięcy

    # 1. Wczytanie macierzy przychodów
    df = pd.read_csv('data/monthly_segment_revenue.csv', index_col=0, parse_dates=True)
    
    # 2. Przygotowanie DataFrame na prognozy
    forecasts = pd.DataFrame(index=pd.date_range(
        start=df.index[-1] + pd.offsets.MonthBegin(),
        periods=horizon,
        freq='M'
    ))
    
    # 3. Pętla po segmentach
    for seg in df.columns:
        series = df[seg].astype(float)
        
        # dopasowanie modelu Holt–Winters (trend add, sezon 12 miesięcy)
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit(optimized=True)
        
        # prognoza
        fcast = model.forecast(horizon)
        forecasts[seg] = fcast
        
        # zapis wykresu
        plt.figure(figsize=(10, 4))
        series.plot(label='History')
        fcast.plot(label='Forecast')
        plt.title(f'Forecast przychodu: segment {seg}')
        plt.ylabel('Revenue')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'images/forecast_{seg.replace(" ", "_")}.png')
        plt.close()
    
    # 4. Zapis prognoz do CSV
    forecasts.to_csv('data/monthly_segment_revenue_forecast.csv')
    print(f'Zapisano prognozy do data/monthly_segment_revenue_forecast.csv')

if __name__ == '__main__':
    main()
