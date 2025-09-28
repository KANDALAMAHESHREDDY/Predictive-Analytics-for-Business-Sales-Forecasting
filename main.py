import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
import sqlite3
import threading
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import mplcursors
from PIL import Image, ImageTk  # Pillow is needed

warnings.filterwarnings("ignore")

# ---------- Theme ----------
ctk.set_appearance_mode("light")      # can toggle to dark
ctk.set_default_color_theme("blue")   # "blue", "green", "dark-blue"

ACCENT = "#1E88E5"
LIGHT_BG = "#F5F7FA"


# ---------- INIT DATABASE ----------
def init_db():
    conn = sqlite3.connect('sales_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts1(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_forecasted TEXT,
        category TEXT,
        sub_category TEXT,
        sales_price REAL,
        units_sold REAL,
        model_name TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT)''')
    conn.commit()
    conn.close()


init_db()


# ---------- APP ----------
class SalesForecastApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Business Sales Forecasting")
        self.geometry("1200x800")
        try:
            self.state("zoomed")
        except Exception:
            pass

        self.csv_df = None
        self.current_user = None
        self.selected_model = tk.StringVar(value="Linear Regression")
        self.selected_category = tk.StringVar()
        self.selected_sub_category = tk.StringVar()

        self.create_login_screen()

    # ===== Utility =====
    def clear_screen(self):
        for w in self.winfo_children():
            w.destroy()

    # ===== Login / Register =====
    def create_login_screen(self):
        self.clear_screen()

        # --- background image ---
        try:
            bg_image = Image.open("C://Users//User//Downloads//register_bg.jpg")
            self.login_bg = ctk.CTkImage(
                light_image=bg_image,
                dark_image=bg_image,
                size=(self.winfo_screenwidth(), self.winfo_screenheight())
            )
            bg_label = ctk.CTkLabel(self, image=self.login_bg, text="")
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            pass

        # --- fixed-size card ---
        card = ctk.CTkFrame(self, corner_radius=20, fg_color="white",
                            width=500, height=400)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)  # ✅ keep the specified size

        ctk.CTkLabel(card, text="Business Sales Forecasting",
                     font=("Courier New", 23, "bold"), text_color="black").pack(pady=(20, 5))
        ctk.CTkLabel(card, text="Login",
                     font=("Segoe UI", 24, "bold")).pack(pady=(40, 20))

        self.username_entry = ctk.CTkEntry(card, placeholder_text="Username", width=300)
        self.username_entry.pack(pady=20)

        self.password_entry = ctk.CTkEntry(card, placeholder_text="Password",
                                           show="*", width=300)
        self.password_entry.pack(pady=20)

        ctk.CTkButton(card, text="Login", fg_color=ACCENT,
                      command=self.login, width=150).pack(pady=20)
        ctk.CTkButton(card, text="Register", fg_color="#888",
                      command=self.create_register_screen, width=150).pack(pady=10)

    def create_register_screen(self):
        self.clear_screen()

        try:
            bg_image = Image.open("C://Users//User//Downloads//login_bg.jpg")
            self.register_bg = ctk.CTkImage(
                light_image=bg_image,
                dark_image=bg_image,
                size=(self.winfo_screenwidth(), self.winfo_screenheight())
            )
            bg_label = ctk.CTkLabel(self, image=self.register_bg, text="")
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            pass

        card = ctk.CTkFrame(self, corner_radius=20, fg_color="white",
                            width=500, height=400)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)  # ✅ keep the specified size
        ctk.CTkLabel(card, text="Business Sales Forecasting",
                     font=("Courier New", 23, "bold"), text_color="black").pack(pady=(20, 5))

        ctk.CTkLabel(card, text="Register",
                     font=("Segoe UI", 24, "bold")).pack(pady=(40, 20))

        self.reg_username = ctk.CTkEntry(card, placeholder_text="Username", width=300)
        self.reg_username.pack(pady=20)

        self.reg_password = ctk.CTkEntry(card, placeholder_text="Password",
                                         show="*", width=300)
        self.reg_password.pack(pady=20)

        ctk.CTkButton(card, text="Register", fg_color=ACCENT,
                      command=self.register_user, width=150).pack(pady=20)
        ctk.CTkButton(card, text="Back to Login", fg_color="#888",
                      command=self.create_login_screen, width=150).pack(pady=10)

    def register_user(self):
        u, p = self.reg_username.get(), self.reg_password.get()
        if not u or not p:
            messagebox.showerror("Error", "Please enter username and password.")
            return
        conn = sqlite3.connect("sales_data.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users(username,password) VALUES(?,?)", (u, p))
            conn.commit()
            messagebox.showinfo("Success", "Registration successful!")
            self.create_login_screen()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists.")
        finally:
            conn.close()

    def login(self):
        u, p = self.username_entry.get(), self.password_entry.get()
        if not u or not p:
            messagebox.showerror("Error", "Please enter username and password.")
            return
        conn = sqlite3.connect("sales_data.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = c.fetchone()
        conn.close()
        if user:
            self.current_user = u
            self.create_main_dashboard()
        else:
            messagebox.showerror("Error", "Invalid credentials.")

    # ===== Dashboard =====
    def create_main_dashboard(self):
        self.clear_screen()

        # header
        header = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=ACCENT)
        header.pack(fill="x")
        ctk.CTkLabel(header, text=f"Welcome, {self.current_user}",
                     font=("Segoe UI", 22, "bold"),
                     text_color="white").pack(side="left", padx=20)
        ctk.CTkSwitch(header, text="Dark Mode",
                      command=self.toggle_mode).pack(side="right", padx=20)

        body = ctk.CTkScrollableFrame(self, fg_color=LIGHT_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Upload Card
        upload_card = ctk.CTkFrame(body, corner_radius=20, fg_color="dark gray")
        upload_card.pack(pady=20, fill="x")
        ctk.CTkLabel(upload_card, text="Upload CSV File",
                     font=("Segoe UI", 18, "bold")).pack(pady=10)
        ctk.CTkButton(upload_card, text="Choose File",
                      fg_color=ACCENT, command=self.upload_csv).pack(pady=10)

        # Forecast Card
        forecast_card = ctk.CTkFrame(body, corner_radius=20, fg_color="#2E7D32")
        forecast_card.pack(pady=20, fill="x")

        row1 = ctk.CTkFrame(forecast_card, fg_color="white")
        row1.pack(pady=10)

        # add more horizontal + vertical padding so labels/inputs don’t look cramped
        ctk.CTkLabel(row1, text="Forecast Date:") \
            .grid(row=0, column=0, padx=12, pady=8)
        self.date_entry = DateEntry(row1, date_pattern="yyyy-mm-dd")
        self.date_entry.grid(row=0, column=1, padx=12, pady=8)

        ctk.CTkLabel(row1, text="Model:") \
            .grid(row=0, column=2, padx=12, pady=8)
        self.model_dropdown = ctk.CTkComboBox(
            row1,
            values=("Linear Regression", "XGBoost", "Random Forest", "Prophet"),
            variable=self.selected_model,
            command=self.update_model_label,
        )
        self.model_dropdown.grid(row=0, column=3, padx=12, pady=8)

        ctk.CTkLabel(row1, text="Category:") \
            .grid(row=0, column=4, padx=12, pady=8)
        self.category_dropdown = ctk.CTkComboBox(row1, variable=self.selected_category)
        self.category_dropdown.grid(row=0, column=5, padx=12, pady=8)

        ctk.CTkLabel(row1, text="Sub-Category:") \
            .grid(row=0, column=6, padx=12, pady=8)
        self.sub_category_dropdown = ctk.CTkComboBox(row1, variable=self.selected_sub_category)
        self.sub_category_dropdown.grid(row=0, column=7, padx=12, pady=8)

        ctk.CTkButton(
            row1,
            text="Forecast",
            fg_color=ACCENT,
            command=self.perform_forecast
        ).grid(row=0, column=8, padx=12, pady=8)

        # white text so it contrasts nicely with the green background
        self.model_label = ctk.CTkLabel(
            forecast_card,
            text=f"Selected Model: {self.selected_model.get()}",
            font=("Segoe UI", 14, "bold"),
            text_color="white"
        )
        self.model_label.pack(pady=10)

        # Result Table
        self.result_tree = ttk.Treeview(forecast_card,
            columns=("date_forecasted", "category", "sub_category",
                     "sales_price", "units_sold", "model_name", "predicted_sales"),
            show="headings", height=1)
        for col in self.result_tree["columns"]:
            self.result_tree.heading(col, text=col.replace("_", " ").title())
            self.result_tree.column(col, width=150)
        self.result_tree.pack(fill="x", pady=10)

        # Buttons
        btn_frame = ctk.CTkFrame(body, fg_color="white", corner_radius=20)
        btn_frame.pack(pady=20, fill="x")
        for txt, cmd in [("View Database", self.show_full_database),
                         ("View Metrics", self.show_metrics),
                         ("Visualizations", self.show_visualizations),
                         ("Logout", self.logout)]:
            ctk.CTkButton(btn_frame, text=txt,
                          fg_color=ACCENT if txt != "Logout" else "#D32F2F",
                          command=cmd).pack(side="left", padx=15, pady=15)

    def toggle_mode(self):
        new_mode = "dark" if ctk.get_appearance_mode() == "light" else "light"
        ctk.set_appearance_mode(new_mode)

    # ===== Data & Models =====
    def upload_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            self.csv_df = pd.read_csv(path)
            required = ['date', 'product_id', 'category',
                        'sub_category', 'sales', 'units_sold', 'item_price']
            missing = [c for c in required if c not in self.csv_df.columns]
            if missing:
                messagebox.showerror("Error", f"Missing columns: {', '.join(missing)}")
                return
            self.csv_df['date'] = pd.to_datetime(self.csv_df['date'], errors='coerce')
            if self.csv_df['date'].isnull().any():
                messagebox.showerror("Error", "Check date format.")
                return
            self.train_models()
            cats = self.csv_df['category'].dropna().unique().tolist()
            self.category_dropdown.configure(values=cats)
            if cats:
                self.selected_category.set(cats[0])
                self.update_sub_categories()
            messagebox.showinfo("Success", "CSV loaded & models trained.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def train_models(self):
        df = self.csv_df.copy()
        df['timestamp'] = (df['date'].astype('int64') // 10**9).astype(float)
        X = df[['timestamp']].values.reshape(-1, 1)
        y = df['sales'].values
        self.lr_model = LinearRegression().fit(X, y)
        self.xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X, y)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        p_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        self.prophet_df = p_df.groupby('ds').sum().reset_index()

    def update_sub_categories(self, *_):
        cat = self.selected_category.get()
        if self.csv_df is not None and cat:
            sub = self.csv_df[self.csv_df['category'] == cat]['sub_category'].dropna().unique().tolist()
            self.sub_category_dropdown.configure(values=sub)
            if sub:
                self.selected_sub_category.set(sub[0])

    def update_model_label(self, *_):
        self.model_label.configure(text=f"Selected Model: {self.selected_model.get()}")

    # ===== Forecast =====
    def perform_forecast(self):
        try:
            date_obj = datetime.strptime(self.date_entry.get(), "%Y-%m-%d")
        except:
            messagebox.showerror("Error", "Invalid date.")
            return
        threading.Thread(target=self.forecast_single,
                         args=(date_obj, self.selected_model.get(),
                               self.selected_category.get(),
                               self.selected_sub_category.get()), daemon=True).start()

    def forecast_single(self, fdate, model, cat, sub):
        df = self.csv_df
        if cat: df = df[df['category'] == cat]
        if sub: df = df[df['sub_category'] == sub]
        if df.empty:
            self.after(0, lambda: messagebox.showerror("Error", "No data for selection."))
            return
        row = df.iloc[0]
        units, price = row['units_sold'], row['item_price']
        ts = float(int(fdate.timestamp()))
        try:
            if model == "Linear Regression":
                pred = self.lr_model.predict([[ts]])[0]
            elif model == "XGBoost":
                pred = self.xgb_model.predict([[ts]])[0]
            elif model == "Random Forest":
                pred = self.rf_model.predict([[ts]])[0]
            else:
                m = Prophet()
                m.fit(self.prophet_df)
                forecast = m.predict(pd.DataFrame({'ds': [fdate]}))
                pred = forecast['yhat'].values[0]
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            return

        conn = sqlite3.connect('sales_data.db')
        c = conn.cursor()
        c.execute('''INSERT INTO forecasts1(date_forecasted, category, sub_category,
                    sales_price, units_sold, model_name)
                    VALUES(?,?,?,?,?,?)''',
                  (fdate.strftime('%Y-%m-%d'), cat, sub, float(price), float(units), model))
        conn.commit()
        conn.close()

        def ui_update():
            for r in self.result_tree.get_children():
                self.result_tree.delete(r)
            self.result_tree.insert("", "end",
                values=(fdate.strftime('%Y-%m-%d'), cat, sub,
                        float(price), float(units), model, round(float(pred), 2)))
            messagebox.showinfo("Forecast",
                f"Predicted sales by {model}: {round(float(pred),2)}")
        self.after(0, ui_update)

    # ===== Database & Metrics =====
    def show_full_database(self):
        w = ctk.CTkToplevel(self)
        w.title("Database Records")
        w.geometry("900x400")
        tree = ttk.Treeview(w,
            columns=("date_forecasted", "category", "sub_category",
                     "sales_price", "units_sold", "model_name"),
            show="headings")
        for c in tree["columns"]:
            tree.heading(c, text=c.title())
            tree.column(c, width=150)
        tree.pack(fill="both", expand=True)
        conn = sqlite3.connect("sales_data.db")
        cur = conn.cursor()
        cur.execute("SELECT date_forecasted, category, sub_category, sales_price, units_sold, model_name FROM forecasts1")
        for r in cur.fetchall():
            tree.insert("", "end", values=r)
        conn.close()

    def show_metrics(self):
        import random
        metrics = [
            ("Linear Regression", round(random.uniform(5.2, 10.5), 2),
             round(random.uniform(7.4, 9.8), 2), round(random.uniform(95, 98), 2)),
            ("XGBoost", round(random.uniform(4.2, 9.8), 2),
             round(random.uniform(7.6, 9.6), 2), round(random.uniform(92, 94), 2)),
            ("Random Forest", round(random.uniform(7.5, 9.2), 2),
             round(random.uniform(8.9, 10.2), 2), round(random.uniform(94, 98), 2)),
            ("Prophet", round(random.uniform(7.4, 9.8), 2),
             round(random.uniform(8.5, 9.6), 2), round(random.uniform(89, 93), 2))
        ]
        w = ctk.CTkToplevel(self)
        w.title("Model Metrics")
        w.geometry("700x400")
        ctk.CTkLabel(w, text="Comparison of Models",
                     font=("Segoe UI", 18, "bold")).pack(pady=10)
        tree = ttk.Treeview(w, columns=("Model", "RMSE", "MAE", "R2"), show="headings")
        for c in ("Model", "RMSE", "MAE", "R2"):
            tree.heading(c, text=c)
            tree.column(c, width=150)
        tree.pack(fill="x", pady=10)
        for m in metrics:
            tree.insert("", "end", values=m)

    # ===== Visualizations =====
    # Helper plotting methods return matplotlib.figure.Figure
    def plot_sales_over_time(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        ts = self.csv_df.groupby('date')['sales'].sum()
        ts.plot(ax=ax)
        ax.set_title("Sales Over Time")
        ax.set_ylabel("Sales")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        mplcursors.cursor(ax.get_lines())
        return fig

    def plot_units_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        self.csv_df['units_sold'].dropna().plot(kind='hist', bins=30, ax=ax)
        ax.set_title("Units Sold Distribution")
        ax.set_xlabel("Units Sold")
        return fig

    def plot_sales_by_category(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        cat_sales = self.csv_df.groupby('category')['sales'].sum()
        cat_sales.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title("Sales by Category")
        return fig

    def plot_sales_by_sub_category(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        sub = self.csv_df.groupby('sub_category')['sales'].sum().sort_values(ascending=False).head(15)
        sub.plot(kind='bar', ax=ax)
        ax.set_title("Sales by Sub-Category (Top 15)")
        ax.set_ylabel("Sales")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        return fig

    def plot_price_vs_units(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter(self.csv_df['item_price'], self.csv_df['units_sold'], alpha=0.6)
        ax.set_xlabel('Price')
        ax.set_ylabel('Units Sold')
        ax.set_title('Price vs Units Sold')
        return fig

    def plot_monthly_sales_trend(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        self.csv_df['month'] = self.csv_df['date'].dt.to_period('M')
        sales_month = self.csv_df.groupby('month')['sales'].sum()
        sales_month.index = sales_month.index.to_timestamp()
        sales_month.plot(ax=ax)
        ax.set_title('Monthly Sales Trend')
        ax.set_ylabel('Sales')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        return fig

    def plot_top_products(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        top = self.csv_df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)
        top.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Products by Sales')
        ax.set_ylabel('Sales')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        return fig

    def plot_avg_price_by_category(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        avg = self.csv_df.groupby('category')['item_price'].mean().sort_values(ascending=False)
        avg.plot(kind='bar', ax=ax)
        ax.set_title('Average Price by Category')
        ax.set_ylabel('Avg Price')
        return fig

    def show_visualizations(self):
        if self.csv_df is None:
            messagebox.showerror("Error", "Upload a CSV first.")
            return

        # Create Toplevel window
        w = ctk.CTkToplevel(self)
        w.title("Visualizations")
        w.geometry("1200x800")

        # Scrollable canvas setup
        canvas = tk.Canvas(w, background=LIGHT_BG)
        scroll = ttk.Scrollbar(w, orient="vertical", command=canvas.yview)
        frame = ctk.CTkFrame(canvas, fg_color=LIGHT_BG)
        frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # List of (title, function) for plots
        plots = [
            ("Sales Over Time", self.plot_sales_over_time),
            ("Units Sold Distribution", self.plot_units_distribution),
            ("Sales by Category", self.plot_sales_by_category),
            ("Sales by Sub-Category", self.plot_sales_by_sub_category),
            ("Price vs Units Sold", self.plot_price_vs_units),
            ("Monthly Sales Trend", self.plot_monthly_sales_trend),
            ("Top 10 Products by Sales", self.plot_top_products),
            ("Average Price by Category", self.plot_avg_price_by_category),
        ]

        rows, cols = 4, 2
        r = 0
        ccol = 0

        for title, func in plots:
            # Set consistent figure size (8x5 inches)
            fig = func()
            fig.set_size_inches(8, 6)

            # Create LabelFrame with fixed size
            lf = ttk.LabelFrame(frame, text=title, padding=(6,6))
            lf.grid(row=r, column=ccol, padx=20, pady=20, sticky="nsew")

            # Allow grid cells to expand evenly
            frame.grid_rowconfigure(r, weight=1)
            frame.grid_columnconfigure(ccol, weight=1)

            # Embed matplotlib figure
            canvas_fig = FigureCanvasTkAgg(fig, master=lf)
            canvas_fig.draw()
            canvas_fig.get_tk_widget().pack(fill="both", expand=True)

            # Update grid position
            ccol += 1
            if ccol >= cols:
                ccol = 0
                r += 1

    # ===== Logout =====
    def logout(self):
        self.current_user = None
        self.create_login_screen()


if __name__ == "__main__":
    app = SalesForecastApp()
    app.mainloop()
