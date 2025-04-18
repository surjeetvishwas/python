import os
import re
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import uuid # To generate unique filenames
import glob # Added for file pattern matching

# Import functions from your existing modules
# from market_data import MarketData # <-- No longer needed for risk-free rate
from data_processing import extraer_datos, procesar_datos
from portfolio_optimizer import PortfolioOptimizer
from report_generator import generar_reporte_pdf
# Assuming visualization.py has plot_portfolio_growth that saves the plot

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static' # Folder to save plots and potentially reports

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
log_file = 'portfolio_app.log' # Name of the log file
log_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*10, backupCount=5) # 10MB per file, keep 5 backups
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO) # Log INFO level and above to file
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO) # Set Flask logger level
app.logger.info('Portfolio Optimizer Application Starting Up...')
# --- End Logging Configuration ---


# Ensure the static folder exists
static_folder_path = app.config['UPLOAD_FOLDER']
if not os.path.exists(static_folder_path):
    os.makedirs(static_folder_path)
    app.logger.info(f"Created static folder at: {static_folder_path}")

# --- Cleanup Function ---
def cleanup_static_files(folder_path, patterns):
    """Removes files matching given patterns in the specified folder."""
    app.logger.info(f"Running cleanup for patterns {patterns} in folder {folder_path}")
    files_deleted_count = 0
    for pattern in patterns:
        try:
            # Create full path pattern
            full_pattern = os.path.join(folder_path, pattern)
            # Find files matching the pattern
            files_to_delete = glob.glob(full_pattern)
            if not files_to_delete:
                 app.logger.info(f"No files found matching pattern: {pattern}")
                 continue

            app.logger.info(f"Found {len(files_to_delete)} file(s) matching pattern {pattern}: {files_to_delete}")
            for f in files_to_delete:
                try:
                    os.remove(f)
                    app.logger.info(f"Successfully deleted old file: {f}")
                    files_deleted_count += 1
                except OSError as e:
                    app.logger.error(f"Error deleting file {f}: {e}", exc_info=True)
        except Exception as e:
             app.logger.error(f"Error during cleanup pattern {pattern}: {e}", exc_info=True)

    app.logger.info(f"Cleanup finished. Deleted {files_deleted_count} file(s).")


# --- Adapted Visualization Function ---
# (Keep the save_portfolio_growth_plot function as it was)
def save_portfolio_growth_plot(growth, filename):
    """Genera y guarda un gráfico de crecimiento del portafolio."""
    plt.figure(figsize=(10, 5))
    plt.plot(growth, label="Crecimiento del Portafolio")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.title("Simulación del Crecimiento del Portafolio")
    plt.legend()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        plt.savefig(filepath)
        app.logger.info(f"Plot saved successfully to {filepath}")
    except Exception as e:
        app.logger.error(f"Failed to save plot to {filepath}: {e}", exc_info=True)
    finally:
        plt.close()


@app.route('/')
def index():
    """Renders the input form."""
    app.logger.info(f"Rendering index page for request from {request.remote_addr}")
    indices = ["S&P 500", "FTSE 100", "Nikkei 225", "S&P/BMV IPC"]
    horizontes = ["1 año", "5 años", "10 años"]
    objetivos = ["Índice Sharpe", "Retorno máximo", "Riesgo mínimo"]
    return render_template('index.html', indices=indices, horizontes=horizontes, objetivos=objetivos)

@app.route('/optimize', methods=['POST'])
def optimize():
    """Handles form submission, runs optimization, cleans old files, and shows results."""
    app.logger.info(f"Optimize request received from {request.remote_addr}")

    # --- Clean up old files BEFORE processing new request ---
    cleanup_static_files(static_folder_path, ["reporte_*.pdf", "growth_plot_*.png"])
    # --- End Cleanup ---

    error_messages = []
    try:
        # --- Get data from form and perform Input Validation ---
        # (Keep the input validation logic as it was)
        tickers_str = request.form.get('tickers', '').strip()
        if not tickers_str:
            error_messages.append("Tickers field is required.")
        elif not re.match(r"^[A-Z,.^ ]+$", tickers_str.upper()):
             error_messages.append("Invalid characters found in Tickers field. Use only letters, commas, spaces, and ^.")
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]
        if not tickers and not error_messages:
            error_messages.append("Tickers field cannot be empty after processing.")
        app.logger.info(f"Tickers received: {tickers}")

        try:
             monto_inversion = float(request.form['monto_inversion'])
             if monto_inversion <= 0:
                 error_messages.append("Investment Amount must be positive.")
        except (ValueError, KeyError):
             error_messages.append("Invalid or missing Investment Amount.")

        moneda_usuario = request.form.get('moneda_usuario', '').strip().upper()
        if not moneda_usuario:
            error_messages.append("Investment Currency is required.")
        elif not re.match(r"^[A-Z]{3}$", moneda_usuario):
             error_messages.append("Invalid Investment Currency format (should be 3 letters like USD).")

        indice = request.form.get('indice')
        if not indice: error_messages.append("Reference Index selection is required.")

        objetivo = request.form.get('objetivo')
        if not objetivo: error_messages.append("Optimization Objective selection is required.")

        try:
            risk_free_rate = float(request.form['risk_free_rate'])
            if not (-0.1 <= risk_free_rate < 0.5):
                 error_messages.append("Risk-Free Rate seems outside a reasonable range (-10% to 50%). Input as decimal (e.g., 0.04).")
        except (ValueError, KeyError):
            error_messages.append("Invalid or missing Risk-Free Rate.")

        if error_messages:
             app.logger.warning(f"Input validation failed: {'; '.join(error_messages)}")
             return render_template('error.html', error_message="Input validation failed:<ul><li>" + "</li><li>".join(error_messages) + "</li></ul>"), 400

        # --- Proceed with core logic only if validation passed ---
        fecha_inicio = "2014-01-01"
        app.logger.info(f"Input validation passed. Starting optimization for tickers: {tickers}, risk rate: {risk_free_rate}")

        # Fetch and Process Data
        data = extraer_datos(tickers, fecha_inicio)
        if data.empty:
             raise ValueError(f"Failed to fetch any data for tickers: {tickers}")

        data = procesar_datos(data)
        if data.empty:
             raise ValueError("No valid data remaining after processing. Check input tickers/date range and logs.")

        # Optimize Portfolio
        optimizer = PortfolioOptimizer(data)
        weights = optimizer.calculate_optimal_weights()
        app.logger.info(f"Calculated portfolio weights: {weights}")

        df_weights = pd.DataFrame(list(weights.items()), columns=["Activo", "Peso"])
        df_weights["Inversión"] = df_weights["Peso"] * monto_inversion

        # Generate unique filenames
        request_id = str(uuid.uuid4())
        report_filename = f"reporte_{request_id}.pdf"
        plot_filename = f"growth_plot_{request_id}.png"
        report_filepath = os.path.join(static_folder_path, report_filename)

        # Generate report
        generar_reporte_pdf(indice, moneda_usuario, monto_inversion, df_weights, risk_free_rate, objetivo, filename=report_filepath)
        app.logger.info(f"Generated PDF report: {report_filename}")

        # Simulate and plot growth
        growth = optimizer.simulate_portfolio(weights)
        save_portfolio_growth_plot(growth, plot_filename)

        # Prepare results for template
        weights_dict = df_weights.to_dict(orient='records')

        app.logger.info(f"Optimization successful. Rendering results page.")
        return render_template('results.html',
                               tickers=tickers_str,
                               monto=monto_inversion,
                               moneda=moneda_usuario,
                               indice=indice,
                               objetivo=objetivo,
                               risk_free_rate_used=risk_free_rate,
                               weights=weights_dict,
                               report_file=report_filename,
                               plot_file=plot_filename)

    except ValueError as ve:
        app.logger.error(f"Data or processing error: {ve}", exc_info=True)
        return render_template('error.html', error_message=str(ve)), 400
    except Exception as e:
        app.logger.error(f"Unexpected error during optimization: {e}", exc_info=True)
        return render_template('error.html', error_message=f"An unexpected server error occurred. Please check the logs or contact support."), 500

# Route to serve generated files (reports, plots) from the static folder
@app.route('/static/<filename>')
def serve_static(filename):
    app.logger.debug(f"Serving static file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)