import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime

import os

# ================================
# CORE FUNCTION
# ================================
def generate_report(csv_path, save_path):
    df = pd.read_csv(csv_path)

    # Encode categorical columns
    le_dept = LabelEncoder()
    le_sal = LabelEncoder()

    df["department"] = le_dept.fit_transform(df["department"])
    df["salary"] = le_sal.fit_transform(df["salary"])

    # Train model
    X = df.drop("left", axis=1)
    y = df["left"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Predict layoff risk
    df["layoff_risk"] = model.predict_proba(X)[:, 1] * 100

    # Department-wise decision
    dept = df.groupby("department")["layoff_risk"].mean().reset_index()
    dept["Decision"] = dept["layoff_risk"].apply(
        lambda x: "📈 Hiring Needed" if x <= 60 else "🚨 Layoffs Needed"
    )
    dept["Department"] = le_dept.inverse_transform(dept["department"])

    # ---------------- PIE CHART ----------------
    hiring_dept = dept[dept["Decision"] == "📈 Hiring Needed"]

    pie_path = os.path.join(os.path.dirname(save_path), "hiring_pie.png")

    plt.figure(figsize=(6,6))
    plt.pie(
        hiring_dept["layoff_risk"],
        labels=hiring_dept["Department"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Hiring Needed: Department-wise Distribution")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()

    # ---------------- PDF ----------------
    doc = SimpleDocTemplate(save_path, pagesize=A4)
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

    # Table
    table_data = [["Department", "Layoff Risk (%)", "AI Decision"]]
    row_styles = []
    row_index = 1

    for _, r in dept.iterrows():
        table_data.append([
            r["Department"],
            round(r["layoff_risk"], 2),
            r["Decision"]
        ])

        if "Layoffs" in r["Decision"]:
            row_styles.append((
                'BACKGROUND', (0, row_index), (-1, row_index),
                colors.HexColor("#fdecea")
            ))
            row_styles.append((
                'TEXTCOLOR', (2, row_index), (2, row_index),
                colors.red
            ))
        else:
            row_styles.append((
                'BACKGROUND', (0, row_index), (-1, row_index),
                colors.HexColor("#eafaf1")
            ))
            row_styles.append((
                'TEXTCOLOR', (2, row_index), (2, row_index),
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
    flow.append(Image(pie_path, width=300, height=300))

    flow.append(Spacer(1, 20))
    flow.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    flow.append(Paragraph(
        "This AI-driven system analyzes employee behavior patterns to identify "
        "departments with growth potential and workforce risk, enabling "
        "data-driven HR decisions.",
        styles["BodyText"]
    ))

    doc.build(flow)

# ================================
# TKINTER GUI
# ================================
def upload_csv():
    global csv_file
    csv_file = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv")]
    )
    if csv_file:
        file_label.config(text=os.path.basename(csv_file))

def generate_pdf():
    if not csv_file:
        messagebox.showerror("Error", "Please upload a CSV file first.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf")],
        title="Save AI HR Report"
    )

    if save_path:
        try:
            generate_report(csv_file, save_path)
            messagebox.showinfo(
                "Success",
                "PDF report generated successfully!"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ================================
# MAIN WINDOW
# ================================
root = tk.Tk()
root.title("AI HR Workforce Analyzer")
root.geometry("500x300")
root.resizable(False, False)

csv_file = None

title = tk.Label(
    root,
    text="AI Workforce Optimization System",
    font=("Helvetica", 16, "bold")
)
title.pack(pady=20)

upload_btn = tk.Button(
    root,
    text="Upload HR CSV File",
    command=upload_csv,
    width=30
)
upload_btn.pack(pady=10)

file_label = tk.Label(root, text="No file selected")
file_label.pack()

generate_btn = tk.Button(
    root,
    text="Generate & Download PDF Report",
    command=generate_pdf,
    bg="#2ecc71",
    fg="white",
    width=35,
    height=2
)
generate_btn.pack(pady=30)

root.mainloop()