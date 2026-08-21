import os
import glob
import psycopg
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL")

conn = psycopg.connect(DSN, autocommit=True)

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "migrations", "*.sql")))

for path in files:
    print("running", path)
    with open(path) as f:
        sql = f.read()
    conn.execute(sql)

print("done")