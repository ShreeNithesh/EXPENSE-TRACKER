"""
Data Management System
Handles CSV file updates and data synchronization
"""
import pandas as pd
import os
from datetime import datetime
import csv

class DataManager:
    def __init__(self, csv_path='data/improved_transactions.csv'):
        self.csv_path = csv_path
        self.backup_path = csv_path.replace('.csv', '_backup.csv')
        
    def append_to_csv(self, transaction_data):
        """Append new transaction data to CSV file"""
        try:
            # Create backup before modifying
            if os.path.exists(self.csv_path):
                df_backup = pd.read_csv(self.csv_path)
                df_backup.to_csv(self.backup_path, index=False)
            
            # Prepare data for CSV
            csv_row = {
                'merchant': transaction_data.get('merchant', ''),
                'category': transaction_data.get('category', ''),
                'amt': transaction_data.get('amt', 0.0),
                'trans_date_trans_time': transaction_data.get('trans_date_trans_time', ''),
                'first': transaction_data.get('first', ''),
                'last': transaction_data.get('last', '')
            }
            
            # Check if CSV exists and has data
            if os.path.exists(self.csv_path):
                # Read existing data
                df_existing = pd.read_csv(self.csv_path)
                
                # Append new row
                df_new = pd.concat([df_existing, pd.DataFrame([csv_row])], ignore_index=True)
            else:
                # Create new DataFrame
                df_new = pd.DataFrame([csv_row])
            
            # Save updated CSV
            df_new.to_csv(self.csv_path, index=False)
            
            print(f"✅ Transaction appended to {self.csv_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error appending to CSV: {str(e)}")
            return False
    
    def sync_db_to_csv(self, db_connection):
        """Synchronize database transactions to CSV file"""
        try:
            from src.db import fetch_all
            
            # Get all transactions from database
            rows = fetch_all(db_connection)
            
            if not rows:
                print("No transactions found in database")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in rows])
            
            # Select only the columns we need for CSV
            csv_columns = ['merchant', 'category', 'amt', 'trans_date_trans_time', 'first', 'last']
            df_csv = df[csv_columns]
            
            # Save to CSV
            df_csv.to_csv(self.csv_path, index=False)
            
            print(f"✅ Database synchronized to {self.csv_path} ({len(df_csv)} transactions)")
            return True
            
        except Exception as e:
            print(f"❌ Error syncing database to CSV: {str(e)}")
            return False
    
    def get_csv_stats(self):
        """Get statistics about the CSV file"""
        try:
            if not os.path.exists(self.csv_path):
                return {"error": "CSV file not found"}
            
            df = pd.read_csv(self.csv_path)
            
            stats = {
                "total_transactions": len(df),
                "unique_users": len(df[['first', 'last']].drop_duplicates()),
                "categories": df['category'].nunique(),
                "category_list": sorted(df['category'].unique().tolist()),
                "date_range": {
                    "earliest": df['trans_date_trans_time'].min(),
                    "latest": df['trans_date_trans_time'].max()
                },
                "total_amount": df['amt'].sum(),
                "avg_amount": df['amt'].mean(),
                "file_size_mb": os.path.getsize(self.csv_path) / (1024 * 1024)
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Error reading CSV stats: {str(e)}"}
    
    def validate_data_integrity(self, db_connection):
        """Validate data integrity between database and CSV"""
        try:
            from src.db import fetch_all
            
            # Get database data
            db_rows = fetch_all(db_connection)
            db_count = len(db_rows)
            
            # Get CSV data
            if os.path.exists(self.csv_path):
                csv_df = pd.read_csv(self.csv_path)
                csv_count = len(csv_df)
            else:
                csv_count = 0
            
            integrity_report = {
                "database_transactions": db_count,
                "csv_transactions": csv_count,
                "difference": abs(db_count - csv_count),
                "in_sync": db_count == csv_count,
                "last_check": datetime.now().isoformat()
            }
            
            return integrity_report
            
        except Exception as e:
            return {"error": f"Error validating data integrity: {str(e)}"}

# Global data manager instance
data_manager = DataManager()

def append_transaction_to_csv(transaction_data):
    """Helper function to append transaction to CSV"""
    return data_manager.append_to_csv(transaction_data)

def sync_database_to_csv(db_connection):
    """Helper function to sync database to CSV"""
    return data_manager.sync_db_to_csv(db_connection)

def get_data_stats():
    """Helper function to get data statistics"""
    return data_manager.get_csv_stats()

def validate_data_integrity(db_connection):
    """Helper function to validate data integrity"""
    return data_manager.validate_data_integrity(db_connection)