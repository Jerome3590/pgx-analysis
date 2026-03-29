import sqlite3

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

# Check field IDs for title, date, journal, DOI
print("=== FIELD IDs ===")
cur.execute("SELECT fieldID, fieldName FROM fields WHERE fieldName IN ('title','date','publicationTitle','DOI','url') ORDER BY fieldID")
for row in cur.fetchall():
    print(row)

# Check item types
print("\n=== ITEM TYPES ===")
cur.execute("SELECT itemTypeID, typeName FROM itemTypes ORDER BY typeName")
for row in cur.fetchall():
    print(row)

# Check all distinct tags
print("\n=== ALL TAGS (top 100 by count) ===")
cur.execute("""
SELECT t.name, COUNT(*) as cnt
FROM tags t
JOIN itemTags it ON t.tagID = it.tagID
GROUP BY t.name
ORDER BY cnt DESC
LIMIT 100
""")
for row in cur.fetchall():
    print(row)

# Count total items
print("\n=== TOTAL NON-DELETED ITEMS ===")
cur.execute("SELECT COUNT(*) FROM items WHERE itemID NOT IN (SELECT itemID FROM deletedItems)")
print(cur.fetchone())

con.close()
