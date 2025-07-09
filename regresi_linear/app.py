from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan data
model = joblib.load('model_regresi_tlkm.pkl')
df = pd.read_csv('TLKM_labeled.csv')
df['date'] = pd.to_datetime(df['date'])
df.rename(columns={'date': 'Date', 'close': 'Close'}, inplace=True)
df = df.sort_values('Date')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df.set_index('Date', inplace=True)

# Buat list tanggal unik untuk dropdown
available_dates = df.index.strftime('%Y-%m-%d').tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_date = request.form['date']
        try:
            search_date = pd.to_datetime(input_date)
            today_close = df.loc[search_date]['Close']
            day_index = df.index.get_loc(search_date)
            prev_close = df.iloc[day_index - 1]['Close'] if day_index > 0 else None
            change = None
            if prev_close is not None:
                change = "Naik" if today_close > prev_close else "Turun" if today_close < prev_close else "Stagnan"

            days_since_start = (search_date - df.index.min()).days
            pred = model.predict([[days_since_start]])[0]

            result = {
                'tanggal': search_date.strftime('%Y-%m-%d'),
                'harga_asli': today_close,
                'harga_prediksi': round(pred, 2),
                'status': change
            }

        except Exception as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result, dates=available_dates)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)