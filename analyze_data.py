import pandas as pd

# Load and analyze the data
df = pd.read_csv('data/credit_card_transactions.csv')

print("=== CATEGORY ANALYSIS ===")
print("Total categories:", df['category'].nunique())
print("\nTop categories by frequency:")
print(df['category'].value_counts().head(10))

print("\n=== SAMPLE MERCHANTS BY CATEGORY ===")
for cat in df['category'].value_counts().head(8).index:
    print(f"\n{cat.upper()}:")
    merchants = df[df['category']==cat]['merchant'].head(5).tolist()
    for merchant in merchants:
        print(f"  - {merchant}")

print("\n=== CATEGORY MAPPING FOR BETTER UNDERSTANDING ===")
category_meanings = {
    'gas_transport': 'Gas stations and transportation',
    'grocery_pos': 'Grocery stores (point of sale)',
    'grocery_net': 'Online grocery shopping',
    'food_dining': 'Restaurants and dining',
    'entertainment': 'Entertainment venues',
    'shopping_pos': 'Retail shopping (point of sale)',
    'shopping_net': 'Online shopping',
    'misc_pos': 'Miscellaneous point of sale',
    'misc_net': 'Miscellaneous online',
    'personal_care': 'Personal care services',
    'health_fitness': 'Health and fitness',
    'travel': 'Travel related expenses',
    'kids_pets': 'Kids and pets expenses',
    'home': 'Home improvement/maintenance'
}

for cat, meaning in category_meanings.items():
    if cat in df['category'].values:
        count = len(df[df['category'] == cat])
        print(f"{cat}: {meaning} ({count} transactions)")