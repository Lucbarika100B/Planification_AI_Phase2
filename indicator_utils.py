import pandas as pd
import ta

def add_indicators(df):
    # S'assurer qu'il y a bien une colonne 'Close'
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'close'}, inplace=True)
    else:
        raise ValueError("La colonne 'Close' est introuvable dans le CSV !")

    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=14).ema_indicator()

    df.dropna(inplace=True)

    df.rename(columns={'close': 'price'}, inplace=True)
    return df



# import pandas as pd
# import ta  # technical analysis library

# # === Pour DQN (lecture directe du fichier CSV) ===
# def add_indicators(csv_path):
#     df = pd.read_csv(csv_path)
    
#     # Standardiser les colonnes
#     df.columns = [col.lower().strip() for col in df.columns]
    
#     # Calcul des indicateurs
#     df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
#     df['macd'] = ta.trend.MACD(close=df['close']).macd()
#     df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=14).ema_indicator()

#     df.dropna(inplace=True)
#     return df




# def add_indicators(df):
#     """
#     Enrichit un DataFrame avec RSI, MACD et EMA.
#     """
#     df = df.copy()
#     df.columns = [col.lower().strip() for col in df.columns]

#     df['rsi'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
#     df['macd'] = ta.trend.MACD(close=df['price']).macd()
#     df['ema'] = ta.trend.EMAIndicator(close=df['price'], window=14).ema_indicator()

#     df.dropna(inplace=True)
#     return df

