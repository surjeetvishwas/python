import os
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import uuid # To generate unique filenames

# Import functions from your existing modules
from market_data import MarketData #
from data_processing import extraer_datos, procesar_datos #
from portfolio_optimizer import PortfolioOptimizer #
from report_generator import generar_reporte_pdf #
# Assuming visualization.py has plot_portfolio_growth that saves the plot

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static' # Folder to save plots and potentially reports

# Ensure the static folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Adapted Visualization Function ---
def save_portfolio_growth_plot(growth, filename):
    """Genera y guarda un gráfico de crecimiento del portafolio."""
    plt.figure(figsize=(10, 5))
    plt.plot(growth, label="Crecimiento del Portafolio")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.title("Simulación del Crecimiento del Portafolio")
    plt.legend()
    # Save the plot to the static folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(filepath)
    plt.close() # Close the plot to free memory
    print(f"Plot saved to {filepath}") # Debug print

@app.route('/')
def index():
    """Renders the input form."""
    # Default values or fetch from somewhere if needed
    indices = ["S&P 500", "FTSE 100", "Nikkei 225", "S&P/BMV IPC"]
    horizontes = ["1 año", "5 años", "10 años"]
    objetivos = ["Índice Sharpe", "Retorno máximo", "Riesgo mínimo"]
    return render_template('index.html', indices=indices, horizontes=horizontes, objetivos=objetivos)

@app.route('/optimize', methods=['POST'])
def optimize():
    """Handles form submission, runs optimization, and shows results."""
    try:
        tickers_str = request.form['tickers']
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',')]
        monto_inversion = float(request.form['monto_inversion'])
        moneda_usuario = request.form['moneda_usuario'].upper()
        indice = request.form['indice']
        # horizonte = request.form['horizonte'] # Currently unused in main logic
        objetivo = request.form['objetivo']
        fecha_inicio = "2014-01-01" # Or calculate based on horizon

        print(f"Received tickers: {tickers}") # Debug print

        # --- Core Logic adapted from main_v1.py ---
        market_data = MarketData() #
        risk_free_rate = market_data.get_risk_free_rate() #

        data = extraer_datos(tickers, fecha_inicio) #
        if data.empty or data.isnull().all().all():
             raise ValueError("No data fetched. Check tickers or date range.")
        data = procesar_datos(data) #
        if data.empty:
             raise ValueError("Data became empty after processing NAs. Check input data.")


        optimizer = PortfolioOptimizer(data) #
        weights = optimizer.calculate_optimal_weights() #


        df_weights = pd.DataFrame(list(weights.items()), columns=["Activo", "Peso"])
        df_weights["Inversión"] = df_weights["Peso"] * monto_inversion

        # Generate unique filenames for this request
        request_id = str(uuid.uuid4())
        report_filename = f"reporte_{request_id}.pdf"
        plot_filename = f"growth_plot_{request_id}.png"
        report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)


        # Generate report
        generar_reporte_pdf(indice, moneda_usuario, monto_inversion, df_weights, risk_free_rate, objetivo, filename=report_filepath) #

        # Simulate and plot growth
        growth = optimizer.simulate_portfolio(weights) #
        save_portfolio_growth_plot(growth, plot_filename) # Use the adapted function


        # --- Prepare results for template ---
        weights_dict = df_weights.to_dict(orient='records')

        return render_template('results.html',
                               tickers=tickers_str,
                               monto=monto_inversion,
                               moneda=moneda_usuario,
                               indice=indice,
                               objetivo=objetivo,
                               weights=weights_dict,
                               report_file=report_filename,
                               plot_file=plot_filename)

    except Exception as e:
        # Basic error handling
        print(f"Error during optimization: {e}") # Log the error
        return render_template('error.html', error_message=str(e)), 500

# Route to serve generated files (reports, plots) from the static folder
@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Make sure debug=False for production environments
    app.run(debug=True)