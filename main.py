import io
import base64
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/cloud deployments
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from flask import Flask, request, render_template_string, send_file, flash, redirect, url_for

# ================================
# VERCEL ENTRYPOINT & ROUTING
# ================================
app = Flask(__name__)
app.secret_key = "hr-analysis-ai-key-secret"

class PrefixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        if path_info.startswith("/main.py"):
            path_info = path_info[len("/main.py"):]
            if not path_info.startswith("/"):
                path_info = "/" + path_info
            environ["PATH_INFO"] = path_info
        return self.wsgi_app(environ, start_response)

app.wsgi_app = PrefixMiddleware(app.wsgi_app)

# Cache for generated PDF in memory
LAST_GENERATED_PDF = None

# ================================
# HTML TEMPLATE
# ================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI HR Workforce Optimizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --success: #10b981;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .badge {
            display: inline-block;
            background: #dbeafe;
            color: var(--primary);
            font-size: 0.8rem;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }

        .header h1 {
            font-size: 2.4rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            padding: 28px;
            margin-bottom: 25px;
        }

        .upload-zone {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 35px 20px;
            text-align: center;
            background: #f1f5f9;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: #eff6ff;
        }

        .upload-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .upload-btn {
            background: var(--primary);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 28px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: background 0.2s;
            margin-top: 15px;
        }

        .upload-btn:hover {
            background: var(--primary-hover);
        }

        .btn-demo {
            background: #ffffff;
            color: #334155;
            border: 1px solid #cbd5e1;
            font-weight: 600;
            padding: 11px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-block;
            margin-left: 10px;
        }

        .btn-demo:hover {
            background: #f8fafc;
            border-color: #94a3b8;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

        .stat-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
        }

        .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            margin-top: 4px;
        }

        .table-responsive {
            overflow-x: auto;
            margin-top: 15px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            background: #1e293b;
            color: white;
            padding: 14px 16px;
            font-size: 0.88rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        th:first-child { border-top-left-radius: 8px; }
        th:last-child { border-top-right-radius: 8px; }

        td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 0.92rem;
        }

        .pill {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.82rem;
        }

        .pill-hiring {
            background: #dcfce7;
            color: #15803d;
        }

        .pill-layoff {
            background: #fee2e2;
            color: #b91c1c;
        }

        .chart-box {
            text-align: center;
            padding: 20px;
        }

        .chart-box img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
        }

        .pdf-actions {
            text-align: center;
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }

        .btn-pdf {
            background: #10b981;
            color: white;
            text-decoration: none;
            padding: 14px 32px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: background 0.2s;
        }

        .btn-pdf:hover {
            background: #059669;
        }

        .alert-error {
            background: #fee2e2;
            border: 1px solid #f87171;
            color: #991b1b;
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }

        .instructions {
            margin-top: 25px;
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Machine Learning & HR Intelligence</span>
            <h1>AI Workforce Optimization System</h1>
            <p>Predict workforce attrition, layoff risk, and department hiring priorities powered by scikit-learn & ReportLab.</p>
        </div>

        {% if error %}
            <div class="alert-error">
                ⚠️ {{ error }}
            </div>
        {% endif %}

        <div class="card">
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <div class="upload-zone" onclick="document.getElementById('csvFileInput').click();">
                    <div class="upload-icon">📁</div>
                    <h3>Upload HR Dataset (.CSV)</h3>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">Click to select file or drag & drop</p>
                    <input type="file" id="csvFileInput" name="file" accept=".csv" style="display: none;" onchange="updateFileName(this);">
                    <div id="fileSelectedName" style="margin-top: 12px; font-weight: 600; color: var(--primary);"></div>
                </div>

                <div style="text-align: center; margin-top: 20px;">
                    <button type="submit" class="upload-btn">⚡ Run AI Analysis</button>
                    <a href="/demo" class="btn-demo">🧪 Try with Demo Data</a>
                </div>
            </form>

            <div class="instructions">
                <strong>Required CSV Columns:</strong>
                <code>department</code> (Department name),
                <code>salary</code> (e.g. low, medium, high),
                <code>left</code> (1 for attrition / 0 for retained),
                and any employee numerical features (e.g., satisfaction level, evaluation, hours).
            </div>
        </div>

        {% if results %}
        <div class="card">
            <h2 style="margin-bottom: 20px; font-size: 1.4rem;">📊 Optimization Summary</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{{ results.total_employees }}</div>
                    <div class="stat-label">Total Employees</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ results.total_depts }}</div>
                    <div class="stat-label">Departments</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #10b981;">{{ results.hiring_count }}</div>
                    <div class="stat-label">Hiring Priority</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color: #ef4444;">{{ results.layoff_count }}</div>
                    <div class="stat-label">Layoff Caution</div>
                </div>
            </div>

            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Department</th>
                            <th style="text-align: center;">Average Layoff Risk</th>
                            <th style="text-align: center;">AI Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in results.dept_data %}
                        <tr>
                            <td style="font-weight: 600;">{{ row.Department }}</td>
                            <td style="text-align: center; font-weight: 700;">{{ row.layoff_risk }}%</td>
                            <td style="text-align: center;">
                                {% if "Hiring" in row.Decision %}
                                    <span class="pill pill-hiring">{{ row.Decision }}</span>
                                {% else %}
                                    <span class="pill pill-layoff">{{ row.Decision }}</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            {% if results.pie_chart_b64 %}
            <div class="chart-box" style="margin-top: 30px;">
                <h3 style="margin-bottom: 12px; color: #1e293b;">Department Hiring Distribution</h3>
                <img src="data:image/png;base64,{{ results.pie_chart_b64 }}" alt="Hiring Pie Chart">
            </div>
            {% endif %}

            <div class="pdf-actions">
                <a href="/download-pdf" class="btn-pdf">📄 Download Executive PDF Report</a>
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        function updateFileName(input) {
            const fileNameDiv = document.getElementById('fileSelectedName');
            if (input.files && input.files[0]) {
                fileNameDiv.textContent = 'Selected: ' + input.files[0].name;
            }
        }
    </script>
</body>
</html>
"""

# ================================
# ML & REPORT CORE LOGIC
# ================================
def run_model_and_generate_report(df):
    global LAST_GENERATED_PDF

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    if "department" not in df.columns or "salary" not in df.columns or "left" not in df.columns:
        raise ValueError("CSV must contain 'department', 'salary', and 'left' columns.")

    # Encode categorical columns
    le_dept = LabelEncoder()
    le_sal = LabelEncoder()

    df["department_encoded"] = le_dept.fit_transform(df["department"].astype(str))
    df["salary_encoded"] = le_sal.fit_transform(df["salary"].astype(str))

    # Features: drop original string columns and target
    drop_cols = ["left", "department", "salary"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["left"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Predict layoff risk
    df["layoff_risk"] = model.predict_proba(X)[:, 1] * 100

    # Department-wise aggregation
    dept_agg = df.groupby("department_encoded")["layoff_risk"].mean().reset_index()
    dept_agg["Decision"] = dept_agg["layoff_risk"].apply(
        lambda x: "📈 Hiring Needed" if x <= 60 else "🚨 Layoffs Needed"
    )
    dept_agg["Department"] = le_dept.inverse_transform(dept_agg["department_encoded"])

    # ---------------- PIE CHART ----------------
    hiring_dept = dept_agg[dept_agg["Decision"] == "📈 Hiring Needed"]
    chart_buffer = io.BytesIO()

    plt.figure(figsize=(6, 6))
    if not hiring_dept.empty:
        plt.pie(
            hiring_dept["layoff_risk"],
            labels=hiring_dept["Department"],
            autopct="%1.1f%%",
            startangle=90
        )
    else:
        plt.text(0.5, 0.5, "No Departments Requiring Hiring", ha="center", va="center")
    plt.title("Hiring Needed: Department-wise Distribution")
    plt.tight_layout()
    plt.savefig(chart_buffer, format="png", dpi=120)
    plt.close()
    chart_buffer.seek(0)
    chart_b64 = base64.b64encode(chart_buffer.getvalue()).decode("utf-8")

    # ---------------- PDF GENERATION ----------------
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(
        "<font size=20><b>AI WORKFORCE OPTIMIZATION REPORT</b></font>",
        styles["Title"]
    ))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(
        f"<b>Generated On:</b> {datetime.now().strftime('%d %B %Y')}",
        styles["BodyText"]
    ))
    flow.append(Spacer(1, 15))

    table_data = [["Department", "Layoff Risk (%)", "AI Decision"]]
    row_styles = []
    row_index = 1

    for _, r in dept_agg.iterrows():
        table_data.append([
            r["Department"],
            round(r["layoff_risk"], 2),
            r["Decision"]
        ])

        if "Layoffs" in r["Decision"]:
            row_styles.append((
                "BACKGROUND", (0, row_index), (-1, row_index),
                colors.HexColor("#fdecea")
            ))
            row_styles.append((
                "TEXTCOLOR", (2, row_index), (2, row_index),
                colors.red
            ))
        else:
            row_styles.append((
                "BACKGROUND", (0, row_index), (-1, row_index),
                colors.HexColor("#eafaf1")
            ))
            row_styles.append((
                "TEXTCOLOR", (2, row_index), (2, row_index),
                colors.green
            ))
        row_index += 1

    table = Table(table_data, colWidths=[160, 120, 160])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ] + row_styles))

    flow.append(table)
    flow.append(Spacer(1, 20))
    flow.append(Paragraph("<b>Hiring Priority Overview</b>", styles["Heading2"]))
    flow.append(Spacer(1, 10))

    # Embed pie chart into ReportLab PDF
    chart_buffer.seek(0)
    flow.append(RLImage(chart_buffer, width=300, height=300))

    flow.append(Spacer(1, 20))
    flow.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "This AI-driven system analyzes employee behavior patterns to identify "
        "departments with growth potential and workforce risk, enabling "
        "data-driven HR decisions.",
        styles["BodyText"]
    ))

    doc.build(flow)
    pdf_buffer.seek(0)
    LAST_GENERATED_PDF = pdf_buffer.getvalue()

    # Prepare results for HTML view
    dept_rows = []
    for _, r in dept_agg.iterrows():
        dept_rows.append({
            "Department": r["Department"],
            "layoff_risk": round(r["layoff_risk"], 2),
            "Decision": r["Decision"]
        })

    return {
        "total_employees": len(df),
        "total_depts": len(dept_agg),
        "hiring_count": len(dept_agg[dept_agg["Decision"] == "📈 Hiring Needed"]),
        "layoff_count": len(dept_agg[dept_agg["Decision"] == "🚨 Layoffs Needed"]),
        "dept_data": dept_rows,
        "pie_chart_b64": chart_b64
    }

# ================================
# ROUTES
# ================================
@app.route("/", methods=["GET"])
@app.route("/main.py", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return render_template_string(HTML_TEMPLATE, error="No file part uploaded.")

    file = request.files["file"]
    if file.filename == "":
        return render_template_string(HTML_TEMPLATE, error="Please select a CSV file.")

    try:
        df = pd.read_csv(file)
        results = run_model_and_generate_report(df)
        return render_template_string(HTML_TEMPLATE, results=results)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

@app.route("/demo", methods=["GET"])
def demo():
    np.random.seed(42)
    n = 300
    depts = ["Sales", "Engineering", "HR", "Finance", "Marketing", "Support", "R&D"]
    salaries = ["low", "medium", "high"]

    data = {
        "department": np.random.choice(depts, n),
        "salary": np.random.choice(salaries, n),
        "satisfaction_level": np.random.uniform(0.1, 1.0, n),
        "last_evaluation": np.random.uniform(0.3, 1.0, n),
        "number_project": np.random.randint(2, 7, n),
        "average_monthly_hours": np.random.randint(130, 310, n),
        "time_spend_company": np.random.randint(2, 10, n),
        "work_accident": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "promotion_last_5years": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "left": np.random.choice([0, 1], n, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    results = run_model_and_generate_report(df)
    return render_template_string(HTML_TEMPLATE, results=results)

@app.route("/download-pdf", methods=["GET"])
def download_pdf():
    global LAST_GENERATED_PDF
    if LAST_GENERATED_PDF is None:
        return redirect(url_for("index"))

    return send_file(
        io.BytesIO(LAST_GENERATED_PDF),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="AI_Workforce_Optimization_Report.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)