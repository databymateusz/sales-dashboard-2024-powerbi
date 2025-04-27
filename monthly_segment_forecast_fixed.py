import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def main():
    # Liczba miesięcy do prognozy
    horizon = 6

    # 1. Wczytaj miesięczne przychody per segment
    df = pd.read_csv('data/monthly_segment_revenue.csv',
                     index_col=0, parse_dates=True)

    # 2. Przygotuj DataFrame na wyniki prognozy
    forecasts = pd.DataFrame(
        index=pd.date_range(
            start=df.index[-1] + pd.offsets.MonthBegin(),
            periods=horizon,
            freq='M'
        )
    )

    # 3. Dla każdego segmentu dobierz model i policz prognozę
    for seg in df.columns:
        series = df[seg].astype(float)
        n_obs = len(series)

        # Jeśli mamy co najmniej 2 pełne cykle sezonowe (2*12=24 m-cy), dodaj sezonowość
        if n_obs >= 24:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=12
            )
        else:
            # Za mało danych na wykrycie sezonowości → tylko trend
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None
            )

        fit = model.fit(optimized=True)
        fcast = fit.forecast(horizon)
        forecasts[seg] = fcast

        # Zapis wykresu
        plt.figure(figsize=(10,4))
        series.plot(label='History')
        fcast.plot(label='Forecast', linestyle='--')
        plt.title(f'Forecast przychodu: segment {seg}')
        plt.ylabel('Revenue')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'images/forecast_{seg.replace(" ", "_")}.png')
        plt.close()

    # 4. Zapis prognoz do CSV
    forecasts.to_csv('data/monthly_segment_revenue_forecast.csv')
    print("Prognozy zapisane w data/monthly_segment_revenue_forecast.csv")

if __name__ == '__main__':
    main()
