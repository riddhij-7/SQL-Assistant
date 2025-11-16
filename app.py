import sqlite3
import pandas as pd
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)


# Extract SQL properly from Gemini response
def get_gemini_sql(question, prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([prompt, question])
    text = response.text.strip()

    # Only pick first SELECT-based line
    lines = text.splitlines()
    sql_candidates = [l for l in lines if l.strip().lower().startswith("select")]

    if sql_candidates:
        return sql_candidates[0]
    else:
        return text


# Generate SQL explanation
def explain_sql_query(query):
    explain_prompt = f"Explain this SQL query step-by-step in simple terms:\n{query}"
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(explain_prompt)
    return response.text.strip()


# Execute SQL on SQLite
def read_sql_query(sql, conn):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_name = [desc[0] for desc in cur.description]
        return rows, col_name

    except sqlite3.Error as e:
        return [("SQL Error", str(e))], ["Error"]


# --- STREAMLIT APP STARTS ---

st.set_page_config(page_title="Gemini SQL Assistant", layout="wide")

# Center the title
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0; '>Gemini SQL Assistant</h1>
    <p style='text-align: center; font-size: 20px;'>Upload a dataset and ask questions in English to generate & execute SQL queries!</p>
""", unsafe_allow_html=True)


# ----------- FILE UPLOAD -----------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # ----------- SMART EXAMPLE QUERIES (Dynamic) -----------
    with st.expander("Try These Smart Example Queries (Auto-Generated)"):

        # General examples
        examples = [
            f"- Show all data from the table.",
            f"- How many rows are there in the dataset?",
            f"- Show the first 10 rows.",
            f"- Show unique values of any column.",
        ]

        # Column-based examples
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if text_cols:
            examples.append(f"- Show unique values of {text_cols[0]}.")
            examples.append(f"- Filter rows where {text_cols[0]} contains 'a'.")

        if numeric_cols:
            examples.append(f"- Show rows where {numeric_cols[0]} > 100.")
            examples.append(f"- Show average of {numeric_cols[0]}.")
            examples.append(f"- Show max value of {numeric_cols[0]}.")
            examples.append(f"- Sort by {numeric_cols[0]} in descending order.")

        # Display all examples
        st.markdown("\n".join(examples))

    # Create in-memory SQLite DB
    conn = sqlite3.connect(":memory:")
    table_name = "uploaded_data"
    df.to_sql(table_name, conn, index=False, if_exists="replace")

    # Generate dynamic prompt
    columns = ", ".join(df.columns)

    prompt = f"""
# 1. Context:
You are helping the user interact with a SQLite database using natural language.

# 2. Table Description:
The uploaded dataset is stored in a table named '{table_name}'.

# 3. Columns Available:
{columns}

# 4. Your Task:
- Convert the user's English question into a clean, correct SQLite SELECT query.
- Use only SQLite syntax.
- Use the table name '{table_name}' exactly.
- Do NOT use semicolons.
- Use correct column names.
- Use LIKE for text filtering when needed.

# 5. Examples:
Q: Show all data.
A: SELECT * FROM {table_name}

Q: How many rows are there?
A: SELECT COUNT(*) FROM {table_name}

Q: Show top 10 rows.
A: SELECT * FROM {table_name} LIMIT 10

Q: Show unique values of any column.
A: SELECT DISTINCT column_name FROM {table_name}

Now generate the SQL query only, nothing else.
"""

    # -------------- USER QUESTION INPUT -------------
    question = st.text_input("Enter your question:")


    if st.button("RUN"):

        if question.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            sql_query = get_gemini_sql(question, prompt)

            st.subheader("Generated SQL Query:")
            st.code(sql_query, language="sql")

            # Run SQL
            result, columns_result = read_sql_query(sql_query, conn)

            if result and "SQL Error" in result[0]:
                st.error(f"Error: {result[0][1]}")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Query Result")
                    result_df = pd.DataFrame(result, columns=columns_result)
                    st.dataframe(result_df, use_container_width=True)

                with col2:
                    st.subheader("SQL Explanation")
                    explanation = explain_sql_query(sql_query)
                    st.write(explanation)
