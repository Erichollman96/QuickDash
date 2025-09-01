import tkinter as tk
from tkinter import ttk, messagebox
import requests
import datetime
import json
import os
import webbrowser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Portfolio Tracker")
        self.root.geometry("1000x700")
        self.stocks = []
        self.stock_data = {}
        self.selected_period = "1mo"  # Default to 1 month
        self.sort_column = None
        self.sort_reverse = False
        self.annotation = None  # Store annotation for hover
        self.data_file = "stock_data.json"  # File to save data
        self.show_forecast = tk.BooleanVar()  # Checkbox variable for ARIMA forecast

        # Input frame
        input_frame = ttk.LabelFrame(root, text="Add Stock", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(input_frame, text="Stock Symbol (e.g. AAPL):").pack(side=tk.LEFT, padx=(0,5))
        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Entry(input_frame, textvariable=self.symbol_var, width=15)
        self.symbol_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(input_frame, text="Add", command=self.add_stock).pack(side=tk.LEFT)

        # Stock list frame
        self.list_frame = ttk.LabelFrame(root, text="Your Stocks", padding=10)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(self.list_frame, columns=("symbol", "company", "30d", "7d", "24h"), show="headings", height=8)
        self.tree.heading("symbol", text="Symbol", command=lambda: self.sort_tree("symbol"))
        self.tree.heading("company", text="Company Name", command=lambda: self.sort_tree("company"))
        self.tree.heading("30d", text="30d Change (%)", command=lambda: self.sort_tree("30d"))
        self.tree.heading("7d", text="7d Change (%)", command=lambda: self.sort_tree("7d"))
        self.tree.heading("24h", text="24h Change (%)", command=lambda: self.sort_tree("24h"))
        self.tree.column("symbol", width=80, anchor=tk.CENTER)
        self.tree.column("company", width=200, anchor=tk.W)
        self.tree.column("30d", width=120, anchor=tk.CENTER)
        self.tree.column("7d", width=120, anchor=tk.CENTER)
        self.tree.column("24h", width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=5)
        
        # Bind click event to tree
        self.tree.bind("<ButtonRelease-1>", self.on_stock_select)

        # Button frame for actions
        button_frame = ttk.Frame(self.list_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="View on Yahoo Finance", command=self.open_yahoo_finance).pack(side=tk.LEFT)

        # Graph frame
        self.graph_frame = ttk.LabelFrame(root, text="Stock Graph", padding=10)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Time period selection frame
        period_frame = ttk.Frame(self.graph_frame)
        period_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(period_frame, text="Time Period:").pack(side=tk.LEFT, padx=(0, 5))
        self.period_var = tk.StringVar(value="1mo")
        period_combo = ttk.Combobox(period_frame, textvariable=self.period_var, 
                                   values=["1wk", "1mo", "3mo", "1y"], 
                                   state="readonly", width=10)
        period_combo.pack(side=tk.LEFT, padx=(0, 10))
        period_combo.bind("<<ComboboxSelected>>", self.on_period_change)
        
        # Add ARIMA forecast checkbox
        ttk.Checkbutton(period_frame, text="Show ARIMA Forecast", 
                       variable=self.show_forecast, 
                       command=self.on_forecast_toggle).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(period_frame, text="Refresh Graph", command=self.refresh_current_graph).pack(side=tk.LEFT)

        self.graph_notebook = ttk.Notebook(self.graph_frame)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)

        # Load saved data after tree is created
        self.load_data()

        # Bind window close event to save data
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_data(self):
        """Load saved stock data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.stocks = data.get('stocks', [])
                    # Note: We don't load stock_data as it needs to be refreshed
                    print(f"Loaded {len(self.stocks)} stocks from saved data")
                    
                    # Fetch fresh data for all saved stocks
                    for symbol in self.stocks:
                        print(f"Fetching data for {symbol}...")
                        stock_data = self.fetch_stock_data(symbol)
                        if stock_data:
                            self.stock_data[symbol] = stock_data
                        else:
                            print(f"Failed to fetch data for {symbol}, removing from list")
                            self.stocks.remove(symbol)
                    
                    # Update the tree display with loaded data
                    if self.stocks:
                        self.update_tree()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.stocks = []

    def save_data(self):
        """Save stock data to file"""
        try:
            data = {
                'stocks': self.stocks,
                'last_updated': datetime.datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("Data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")

    def on_closing(self):
        """Handle window closing - save data before exit"""
        try:
            self.save_data()
        except Exception as e:
            print(f"Error saving data on exit: {e}")
        finally:
            self.root.destroy()

    def on_period_change(self, event=None):
        self.selected_period = self.period_var.get()
        # Refresh the current graph if a stock is selected
        selected = self.tree.selection()
        if selected:
            symbol = self.tree.item(selected[0], "values")[0]
            if symbol in self.stock_data:
                self.show_stock_graph(symbol)

    def refresh_current_graph(self):
        selected = self.tree.selection()
        if selected:
            symbol = self.tree.item(selected[0], "values")[0]
            if symbol in self.stock_data:
                # Fetch fresh data with new period
                data = self.fetch_stock_data(symbol)
                if data:
                    self.stock_data[symbol] = data
                    self.update_tree()
                    self.show_stock_graph(symbol)
        elif self.stocks:  # If no stock is selected but we have stocks, show the first one
            symbol = self.stocks[0]
            if symbol in self.stock_data:
                # Fetch fresh data with new period
                data = self.fetch_stock_data(symbol)
                if data:
                    self.stock_data[symbol] = data
                    self.update_tree()
                    self.show_stock_graph(symbol)

    def sort_tree(self, column):
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        
        # Get all items and their values
        items = []
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            items.append((item, values))
        
        # Sort items
        def sort_key(item_tuple):
            item, values = item_tuple
            if column == "symbol" or column == "company":
                return values[{"symbol": 0, "company": 1}[column]]  # Symbol and company are strings
            else:
                # Extract numeric value from percentage string
                try:
                    return float(values[{"30d": 2, "7d": 3, "24h": 4}[column]].replace("%", ""))
                except:
                    return 0.0
        
        items.sort(key=sort_key, reverse=self.sort_reverse)
        
        # Reorder items in tree
        for item, values in items:
            self.tree.move(item, "", "end")

    def on_stock_select(self, event):
        selected = self.tree.selection()
        if selected:
            symbol = self.tree.item(selected[0], "values")[0]
            if symbol in self.stock_data:
                self.show_stock_graph(symbol)

    def show_stock_graph(self, symbol):
        # Clear existing graph
        for widget in self.graph_frame.winfo_children():
            if isinstance(widget, ttk.Notebook):
                widget.destroy()
        
        # Create new notebook
        self.graph_notebook = ttk.Notebook(self.graph_frame)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add graph for selected stock
        data = self.stock_data[symbol]
        self.add_graph_tab(symbol, data)

    def add_stock(self):
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol.")
            return
        if symbol in self.stocks:
            messagebox.showinfo("Duplicate", f"{symbol} is already in your list.")
            return
        try:
            data = self.fetch_stock_data(symbol)
            if data is None:
                messagebox.showerror("Error", f"Could not fetch data for {symbol}.")
                return
            self.stocks.append(symbol)
            self.stock_data[symbol] = data
            self.update_tree()
            self.symbol_var.set("")
            # Save data after adding stock
            self.save_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add stock: {e}")

    def remove_selected(self):
        selected = self.tree.selection()
        for item in selected:
            symbol = self.tree.item(item, "values")[0]
            if symbol in self.stocks:
                self.stocks.remove(symbol)
                self.stock_data.pop(symbol, None)
        self.update_tree()
        # Save data after removing stocks
        self.save_data()
        # Clear graph if no stocks left
        if not self.stocks:
            for widget in self.graph_frame.winfo_children():
                if isinstance(widget, ttk.Notebook):
                    widget.destroy()
            self.graph_notebook = ttk.Notebook(self.graph_frame)
            self.graph_notebook.pack(fill=tk.BOTH, expand=True)

    def update_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for symbol in self.stocks:
            data = self.stock_data.get(symbol)
            if data:
                # Create item with values
                item = self.tree.insert("", tk.END, values=(
                    symbol,
                    data.get('company_name', symbol),
                    f"{data['30d_change']:.2f}%",
                    f"{data['7d_change']:.2f}%",
                    f"{data['24h_change']:.2f}%"
                ))
                
                # Configure symbol column to always be black
                self.tree.tag_configure("symbol_black", foreground="black")
                self.tree.item(item, tags=("symbol_black",))
                
                # Apply colors to individual columns only
                # 30d column
                if data['30d_change'] > 0:
                    self.tree.set(item, "30d", f"+{data['30d_change']:.2f}%")
                    self.tree.tag_configure("positive_30d", foreground="green")
                    self.tree.item(item, tags=("symbol_black", "positive_30d"))
                elif data['30d_change'] < 0:
                    self.tree.tag_configure("negative_30d", foreground="red")
                    self.tree.item(item, tags=("symbol_black", "negative_30d"))
                else:
                    self.tree.tag_configure("neutral_30d", foreground="black")
                    self.tree.item(item, tags=("symbol_black", "neutral_30d"))
                
                # 7d column
                if data['7d_change'] > 0:
                    self.tree.set(item, "7d", f"+{data['7d_change']:.2f}%")
                    self.tree.tag_configure("positive_7d", foreground="green")
                    self.tree.item(item, tags=("symbol_black", "positive_7d"))
                elif data['7d_change'] < 0:
                    self.tree.tag_configure("negative_7d", foreground="red")
                    self.tree.item(item, tags=("symbol_black", "negative_7d"))
                else:
                    self.tree.tag_configure("neutral_7d", foreground="black")
                    self.tree.item(item, tags=("symbol_black", "neutral_7d"))
                
                # 24h column
                if data['24h_change'] > 0:
                    self.tree.set(item, "24h", f"+{data['24h_change']:.2f}%")
                    self.tree.tag_configure("positive_24h", foreground="green")
                    self.tree.item(item, tags=("symbol_black", "positive_24h"))
                elif data['24h_change'] < 0:
                    self.tree.tag_configure("negative_24h", foreground="red")
                    self.tree.item(item, tags=("symbol_black", "negative_24h"))
                else:
                    self.tree.tag_configure("neutral_24h", foreground="black")
                    self.tree.item(item, tags=("symbol_black", "neutral_24h"))

    def fetch_stock_data(self, symbol):
        # Use Yahoo Finance API via rapidapi or unofficial endpoint
        # We'll use the Yahoo Finance chart API (no key required)
        # Example: https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1mo
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={self.selected_period}"
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"HTTP Error {resp.status_code} for {symbol}")
                return None
            data = resp.json()
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                print(f"No chart data found for {symbol}")
                return None
            result = data['chart']['result'][0]
            if 'timestamp' not in result or 'indicators' not in result:
                print(f"Missing timestamp or indicators for {symbol}")
                return None
            timestamps = result['timestamp']
            if not timestamps:
                print(f"No timestamps found for {symbol}")
                return None
            quote_data = result['indicators']['quote'][0]
            if 'close' not in quote_data:
                print(f"No close data found for {symbol}")
                return None
            closes = quote_data['close']
            # Remove None values (market closed days)
            price_data = [(datetime.datetime.fromtimestamp(ts), close) for ts, close in zip(timestamps, closes) if close is not None]
            if len(price_data) < 2:
                print(f"Insufficient price data for {symbol}: {len(price_data)} points")
                return None
            # Sort by date ascending
            price_data.sort()
            # Get current, 1d ago, 7d ago, 30d ago
            today = price_data[-1][0]
            close_today = price_data[-1][1]
            # Find closes for 1d, 7d, 30d ago (find closest available)
            def get_close(days_ago): 
                target = today - datetime.timedelta(days=days_ago)
                # Find the closest date before or equal to target
                for dt, price in reversed(price_data):
                    if dt <= target:
                        return price
                return price_data[0][1]  # fallback to oldest
            close_1d = get_close(1)
            close_7d = get_close(7)
            close_30d = get_close(30)
            change_24h = ((close_today - close_1d) / close_1d) * 100 if close_1d else 0
            change_7d = ((close_today - close_7d) / close_7d) * 100 if close_7d else 0
            change_30d = ((close_today - close_30d) / close_30d) * 100 if close_30d else 0
            
            # Get company name from meta data
            company_name = symbol  # Default to symbol if no company name found
            if 'meta' in result and 'shortName' in result['meta']:
                company_name = result['meta']['shortName']
            
            return {
                "price_data": price_data,
                "30d_change": change_30d,
                "7d_change": change_7d,
                "24h_change": change_24h,
                "company_name": company_name
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def add_graph_tab(self, symbol, data):
        frame = ttk.Frame(self.graph_notebook)
        fig, ax = plt.subplots(figsize=(6,3), dpi=100)
        dates = [dt for dt, price in data["price_data"]]
        prices = [price for dt, price in data["price_data"]]
        line, = ax.plot(dates, prices, marker='o', label='Historical Data')
        
        # Add ARIMA forecast if checkbox is checked
        if self.show_forecast.get() and len(prices) >= 10:  # Need at least 10 data points
            try:
                forecast_dates, forecast, lower_bound, upper_bound = self.generate_arima_forecast(dates, prices)
                ax.plot(forecast_dates, forecast, 'r-', linewidth=2, label='ARIMA Forecast')
                ax.fill_between(forecast_dates, lower_bound, upper_bound, alpha=0.2, color='red', label='95% Confidence Interval')
                ax.legend()
            except Exception as e:
                print(f"ARIMA forecast failed for {symbol}: {e}")
        
        # Set title based on selected period
        period_names = {"1wk": "1 Week", "1mo": "1 Month", "3mo": "3 Months", "1y": "1 Year"}
        period_name = period_names.get(self.selected_period, self.selected_period)
        title = f"{symbol} - Last {period_name}"
        if self.show_forecast.get():
            title += " (with ARIMA Forecast)"
        ax.set_title(title)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        fig.autofmt_xdate()
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax:
                # Find the closest point
                xdata, ydata = line.get_data()
                if len(xdata) > 0:
                    # Convert event x to datetime if needed
                    if isinstance(event.xdata, (int, float)):
                        # Convert matplotlib date number to datetime
                        event_date = mdates.num2date(event.xdata)
                        # Make timezone-naive if it's timezone-aware
                        if event_date.tzinfo is not None:
                            event_date = event_date.replace(tzinfo=None)
                        # Find closest date index
                        distances = [abs((d - event_date).total_seconds()) for d in xdata]
                        closest_idx = distances.index(min(distances))
                        x, y = xdata[closest_idx], ydata[closest_idx]
                    else:
                        x, y = event.xdata, event.ydata
                    
                    # Remove previous annotation
                    if self.annotation:
                        self.annotation.remove()
                    
                    # Create new annotation
                    self.annotation = ax.annotate(
                        f'${y:.2f}\n{x.strftime("%Y-%m-%d")}',
                        xy=(x, y),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=9
                    )
                    fig.canvas.draw_idle()
        
        def on_leave(event):
            # Remove annotation when mouse leaves the plot
            if self.annotation:
                self.annotation.remove()
                self.annotation = None
                fig.canvas.draw_idle()
        
        # Connect hover events
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('axes_leave_event', on_leave)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.graph_notebook.add(frame, text=symbol)

    def generate_arima_forecast(self, dates, prices):
        """Generate ARIMA forecast for the given price data"""
        try:
            # Convert to numpy array
            price_array = np.array(prices)
            
            # Fit ARIMA model (1,1,1) - can be adjusted for better results
            model = ARIMA(price_array, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast for next 5 days with confidence intervals
            forecast_steps = 5
            forecast_result = fitted_model.forecast(steps=forecast_steps, alpha=0.05)  # 95% confidence interval
            
            # Extract forecast values and confidence intervals
            if hasattr(forecast_result, 'predicted_mean'):
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            else:
                # Fallback for older statsmodels versions
                forecast = forecast_result
                # Estimate confidence intervals (simplified)
                std_dev = np.std(price_array) * 0.1  # Rough estimate
                lower_bound = forecast - 1.96 * std_dev
                upper_bound = forecast + 1.96 * std_dev
            
            # Generate forecast dates
            last_date = dates[-1]
            forecast_dates = []
            for i in range(1, forecast_steps + 1):
                # Add business days (skip weekends)
                next_date = last_date + datetime.timedelta(days=i)
                while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    next_date += datetime.timedelta(days=1)
                forecast_dates.append(next_date)
            
            return forecast_dates, forecast, lower_bound, upper_bound
            
        except Exception as e:
            print(f"Error generating ARIMA forecast: {e}")
            return [], [], [], []

    def on_forecast_toggle(self):
        """Handle ARIMA forecast checkbox toggle"""
        selected = self.tree.selection()
        if selected:
            symbol = self.tree.item(selected[0], "values")[0]
            if symbol in self.stock_data:
                self.show_stock_graph(symbol)

    def open_yahoo_finance(self):
        """Open the selected stock's Yahoo Finance page in the default browser"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a stock from the table first.")
            return
        
        symbol = self.tree.item(selected[0], "values")[0]
        url = f"https://finance.yahoo.com/quote/{symbol}"
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open browser: {e}")

def main():
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
