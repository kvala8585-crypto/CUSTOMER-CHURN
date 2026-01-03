import sqlite3
import pandas as pd
#SQLite database se connect krna


# Database path
db_path = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\customer_churn.db"

# Connect to SQLite
conn = sqlite3.connect(db_path)

print("✅ Database connected successfully")
#Database की tables check करना
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)
print(tables)

#Database की tables check करना
df = pd.read_sql("SELECT * FROM customer_data", conn)

print(df.head())
print(df.shape)

conn.close()
print("🔒 Database connection closed")
#“I used sqlite3 in Python to connect with SQLite database and performed SQL queries directly into pandas DataFrames for analysis and modeling.”