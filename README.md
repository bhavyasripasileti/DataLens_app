# 🔬 DataLens - CSV Analyzer

A sleek, dark-themed interactive data analysis app built with Streamlit. Upload any CSV file and instantly explore, diagnose, and visualize your dataset.

---

## 🚀 Features

- **📋 Dataset Preview** — Browse your data with adjustable row display, column info, and descriptive statistics
- **🩺 Missing Value Analysis** — Detect missing data with summary tables, bar charts, and a missingness heatmap
- **🔗 Correlation Heatmap** — Pearson, Spearman, and Kendall correlation matrices with top correlated pairs
- **📊 Visualisations** — 6 chart types: Histogram, Box Plot, Scatter Plot, Bar Chart, Line Chart, and Pair Plot

---

## 🖥️ Demo

![DataLens Screenshot](screenshot.png)

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/DataLens-app.git
cd DataLens-app
```

**2. Create a virtual environment (Python 3.11 recommended)**
```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run csv_analyzer.py
```

---

## 📦 Requirements

- Python 3.11+
- Streamlit 1.43.2
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📁 Project Structure
```
DataLens-app/
│
├── csv_analyzer.py       # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## 🧪 How to Use

1. Launch the app with `streamlit run csv_analyzer.py`
2. Upload any `.csv` file using the sidebar
3. Navigate through the tabs:
   - **Preview** — inspect your raw data
   - **Missing Values** — identify data quality issues
   - **Correlation** — find relationships between numeric columns
   - **Visualisations** — generate interactive charts

---

## 🎨 Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web app framework |
| Pandas | Data manipulation |
| Matplotlib | Chart rendering |
| Seaborn | Statistical visualizations |
| NumPy | Numerical computing |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---
## 👨‍💻 Author

**Bhavya Sri Pasileti**

BTech Student | Data Science & AI Enthusiast  
Passionate about building **Passionate about Turning Data into Insights**.

---
Made with ❤️ using Streamlit
