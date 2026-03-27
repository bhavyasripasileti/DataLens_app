<div align="center">

# 🔬 DataLens — Smart CSV Analyzer Dashboard

### Explore, clean, and visualize your data — instantly.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?style=for-the-badge)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical--Viz-green?style=for-the-badge)](https://seaborn.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

> A powerful and interactive **data analysis dashboard** that allows users to upload any CSV file and instantly explore, diagnose, and visualize their dataset — all in a clean Streamlit interface.

<br/>

**[🖥️ Live App link](#-demo) · [📊 Features](#-key-features) · [📦 Setup](#-installation) · [🧪 Usage](#-how-to-use)**

</div>

---

## 🌐 Overview

Analyzing datasets manually can be time-consuming and inefficient, especially during early-stage exploration.

**DataLens** simplifies this process by providing an **all-in-one interactive dashboard** to quickly understand your data, detect issues, and generate meaningful visual insights — without writing extensive code.

---

## ✨ Key Features

| Feature                       | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| 📋 Dataset Preview            | View raw data, column info, and descriptive statistics              |
| 🩺 Missing Value Analysis     | Identify null values with summaries and heatmaps                    |
| 🔗 Correlation Analysis       | Pearson, Spearman, and Kendall correlation matrices                 |
| 📊 Interactive Visualizations | Histogram, Box Plot, Scatter Plot, Bar Chart, Line Chart, Pair Plot |
| 🎯 Clean UI                   | Dark-themed, intuitive Streamlit interface                          |



## 📂 Project Structure

```
DataLens-app/
│
├── csv_analyzer.py       # Streamlit app (UI + logic)
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category        | Technology          |
| --------------- | ------------------- |
| Language        | Python 3.11+        |
| Framework       | Streamlit           |
| Data Processing | Pandas, NumPy       |
| Visualization   | Matplotlib, Seaborn |

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DataLens-app.git
cd DataLens-app
```

### 2️⃣ Create Virtual Environment

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run csv_analyzer.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 How to Use

1. Launch the app using Streamlit
2. Upload a `.csv` file via the sidebar
3. Navigate through analysis tabs:

   * **Preview** → Inspect dataset structure
   * **Missing Values** → Identify data quality issues
   * **Correlation** → Discover relationships between variables
   * **Visualizations** → Generate charts interactively
4. Gain quick insights without writing code

---

## 💼 Project Highlights

* Built an **interactive data analysis tool** for quick EDA
* Implemented **data quality diagnostics (missing values, correlations)**
* Designed **multiple visualization modules**
* Developed a **user-friendly Streamlit dashboard**
* Enables **no-code data exploration for beginners**

---

## 🔮 Future Improvements

* Upload Excel & JSON support
* Download processed dataset
* Advanced filtering & query system
* Dashboard export (PDF/PNG)
* Integration with ML model insights

---

## 📜 License

MIT License — free to use and modify.

---

## 👤 Author

**Bhavya Sri Pasileti**

> BTech Student | Data Science & AI Enthusiast
> Passionate about turning raw data into meaningful insights.

[LinkedIn](https://www.linkedin.com/in/bhavya-sri-pasileti-16565a2a1)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

*Built with ❤️ by Bhavya Sri Pasileti*

</div>
