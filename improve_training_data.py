import pandas as pd
import numpy as np
import random

# Load original data
df = pd.read_csv('data/credit_card_transactions.csv')

# Create highly precise merchant mappings for each category with more specific keywords
realistic_merchants = {
    'gas_transport': [
        'Shell Gas Station', 'Exxon Mobil Gas', 'BP Gas Station', 'Chevron Gas', 'Texaco Station', 'Sunoco Gas',
        'Marathon Gas Station', 'Speedway Gas', 'Mobil Gas', 'Citgo Gas', 'Valero Gas', 'Phillips 66 Gas',
        'Uber Ride', 'Lyft Ride', 'Taxi Cab', 'Yellow Cab', 'Metro Bus', 'City Bus', 'Train Fare', 'Subway Fare',
        'Parking Garage', 'Parking Meter', 'Toll Road', 'Bridge Toll', 'Airport Parking', 'Gas Fill Up'
    ],
    'grocery_pos': [
        'Walmart Grocery', 'Target Grocery', 'Kroger Store', 'Safeway Grocery', 'Whole Foods Market',
        'Trader Joes', 'Costco Wholesale', 'Sams Club', 'Publix Supermarket', 'Food Lion Grocery',
        'Giant Eagle', 'Stop Shop Grocery', 'Harris Teeter', 'Wegmans Food', 'Fresh Market Grocery',
        'Aldi Grocery', 'Food 4 Less', 'Piggly Wiggly', 'IGA Grocery', 'Supermarket Shopping'
    ],
    'grocery_net': [
        'Amazon Fresh Delivery', 'Instacart Grocery', 'Walmart Grocery Delivery', 'Target Grocery Online',
        'Kroger Delivery', 'Whole Foods Delivery', 'FreshDirect Online', 'Peapod Delivery',
        'Shipt Grocery', 'Online Grocery Order', 'Grocery Delivery Service'
    ],
    'food_dining': [
        'McDonalds Restaurant', 'Burger King', 'Subway Sandwich', 'Starbucks Coffee', 'Dunkin Donuts', 'KFC Chicken',
        'Pizza Hut', 'Dominos Pizza', 'Taco Bell', 'Chipotle Mexican', 'Panera Bread', 'Wendys Restaurant',
        'Olive Garden Restaurant', 'Applebees Grill', 'Chilis Restaurant', 'TGI Fridays', 'Red Lobster',
        'Local Restaurant', 'Coffee Shop', 'Diner Breakfast', 'Food Truck', 'Fast Food', 'Restaurant Dinner',
        'Cafe Lunch', 'Bakery', 'Ice Cream Shop', 'Donut Shop'
    ],
    'entertainment': [
        'Netflix Subscription', 'Spotify Premium', 'Disney Plus', 'Hulu Streaming', 'Amazon Prime Video',
        'AMC Movie Theater', 'Regal Cinemas', 'Movie Tickets', 'Concert Tickets', 'Sports Game Tickets',
        'Bowling Alley', 'Mini Golf', 'Arcade Games', 'Theme Park', 'Amusement Park',
        'YouTube Premium', 'Apple Music', 'Gaming Subscription', 'Entertainment Venue'
    ],
    'shopping_pos': [
        'Target Store', 'Walmart Shopping', 'Best Buy Electronics', 'Macys Department', 'Nordstrom Store',
        'JCPenney', 'Kohls Department', 'TJ Maxx', 'Marshall Store', 'Ross Dress',
        'Old Navy Clothing', 'Gap Store', 'H&M Fashion', 'Zara Clothing', 'Forever 21',
        'Victoria Secret', 'Clothing Store', 'Department Store', 'Electronics Store', 'Fashion Retail'
    ],
    'shopping_net': [
        'Amazon Purchase', 'eBay Buy', 'Target Online', 'Walmart Online', 'Best Buy Online',
        'Macys Online', 'Nordstrom Online', 'Zappos Shoes', 'Etsy Handmade', 'Overstock Online',
        'Online Shopping', 'E-commerce Purchase', 'Internet Order', 'Web Store'
    ],
    'misc_pos': [
        'Dollar Tree', 'CVS Pharmacy', 'Walgreens Store', 'Rite Aid Pharmacy', 'Post Office',
        'Bank ATM Fee', 'Hardware Store', 'Convenience Store', '7-Eleven Store', 'Circle K',
        'General Store', 'Variety Store', 'Miscellaneous Purchase', 'Car Loan Payment', 'Auto Loan',
        'Student Loan Payment', 'Personal Loan', 'Credit Card Payment', 'Bank Fee', 'Loan Payment',
        'Insurance Premium', 'Car Insurance', 'Health Insurance', 'Life Insurance Payment'
    ],
    'misc_net': [
        'PayPal Payment', 'Venmo Transfer', 'Online Service Fee', 'Digital Download', 'App Store Purchase',
        'Software License', 'Cloud Storage Fee', 'Domain Registration', 'Web Hosting', 'Online Subscription',
        'Digital Service', 'Internet Payment'
    ],
    'personal_care': [
        'Hair Salon Cut', 'Barber Shop', 'Nail Salon Manicure', 'Day Spa', 'Massage Therapy',
        'Dentist Appointment', 'Eye Doctor Visit', 'Dermatologist', 'Beauty Supply Store', 'Sephora Cosmetics',
        'Ulta Beauty', 'Salon Services', 'Beauty Treatment', 'Personal Grooming'
    ],
    'health_fitness': [
        'Planet Fitness Gym', 'LA Fitness', 'Gold Gym Membership', 'Yoga Studio Class', 'Personal Trainer',
        'Physical Therapy', 'CVS Pharmacy Medicine', 'Walgreens Prescription', 'Doctor Visit',
        'Hospital Bill', 'Medical Appointment', 'Fitness Membership', 'Health Checkup', 'Prescription Medicine'
    ],
    'travel': [
        'Delta Airlines Flight', 'American Airlines', 'United Airlines', 'Southwest Airlines Ticket',
        'Marriott Hotel Stay', 'Hilton Hotel', 'Holiday Inn', 'Airbnb Rental', 'Expedia Booking',
        'Booking Hotel', 'Hertz Car Rental', 'Enterprise Rental', 'Avis Rental Car',
        'Hotel Accommodation', 'Flight Ticket', 'Travel Booking', 'Vacation Rental'
    ],
    'kids_pets': [
        'Toys R Us', 'Target Toys', 'Kids Clothing Store', 'Daycare Payment', 'School Supplies',
        'Petco Pet Store', 'PetSmart', 'Veterinarian Visit', 'Pet Grooming Service', 'Pet Food',
        'Baby Store Purchase', 'Diaper Purchase', 'Kids Activities', 'Pet Care', 'Child Care'
    ],
    'home': [
        'Home Depot Store', 'Lowes Home', 'Menards Hardware', 'Hardware Store', 'Paint Store',
        'IKEA Furniture', 'Furniture Store', 'Bed Bath Beyond', 'Home Goods Store',
        'Plumber Service', 'Electrician Work', 'Lawn Care Service', 'Cleaning Service',
        'Home Improvement', 'House Maintenance', 'Home Repair', 'Mortgage Payment', 'Home Loan',
        'Property Tax', 'HOA Fee', 'Homeowners Association', 'Rent Payment', 'Apartment Rent',
        'House Rent', 'Utilities Bill', 'Electric Bill', 'Water Bill', 'Internet Bill'
    ]
}

# Create improved dataset
improved_data = []

for _, row in df.iterrows():
    category = row['category']
    if category in realistic_merchants:
        # Replace generic merchant with realistic one
        new_merchant = random.choice(realistic_merchants[category])
        improved_data.append({
            'merchant': new_merchant,
            'category': category,
            'amt': row['amt'],
            'trans_date_trans_time': row['trans_date_trans_time'],
            'first': row['first'],
            'last': row['last']
        })

# Create DataFrame and save
improved_df = pd.DataFrame(improved_data)
improved_df.to_csv('data/improved_transactions.csv', index=False)

print(f"Created improved dataset with {len(improved_df)} transactions")
print("\nSample improved merchants by category:")
for cat in improved_df['category'].value_counts().head(5).index:
    print(f"\n{cat}:")
    merchants = improved_df[improved_df['category']==cat]['merchant'].unique()[:5]
    for merchant in merchants:
        print(f"  - {merchant}")