"""
Expense Advisor - Provides personalized savings tips and expense analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class ExpenseAdvisor:
    def __init__(self):
        # Category-specific savings tips
        self.category_tips = {
            'food_dining': [
                "🍽️ Cook more meals at home - you could save 60-70% on food costs",
                "☕ Make coffee at home instead of buying from cafes",
                "🥪 Pack lunch for work instead of eating out",
                "🍕 Limit takeout to 1-2 times per week",
                "💰 Use restaurant apps and coupons for discounts",
                "🍳 Meal prep on weekends to avoid impulse food purchases"
            ],
            'gas_transport': [
                "🚗 Combine errands into one trip to save gas",
                "🚌 Use public transportation when possible",
                "🚴 Walk or bike for short distances",
                "🚗 Carpool or use rideshare for longer trips",
                "⛽ Use gas station apps to find cheapest prices",
                "🔧 Keep your car well-maintained for better fuel efficiency"
            ],
            'grocery_pos': [
                "📝 Make a shopping list and stick to it",
                "🛒 Shop with a full stomach to avoid impulse buys",
                "🏷️ Use coupons and store loyalty programs",
                "🥫 Buy generic brands - they're often 20-30% cheaper",
                "📦 Buy in bulk for non-perishable items you use regularly",
                "🛍️ Shop sales and stock up on frequently used items"
            ],
            'grocery_net': [
                "📦 Avoid delivery fees by meeting minimum order requirements",
                "🛒 Compare prices between online and in-store",
                "⏰ Schedule deliveries to avoid rush hour surcharges",
                "🥫 Subscribe to regular deliveries for essentials to get discounts"
            ],
            'entertainment': [
                "📺 Share streaming subscriptions with family/friends",
                "🎬 Look for free entertainment options like parks and museums",
                "🎫 Buy movie tickets during matinee hours",
                "📚 Use your local library for books, movies, and events",
                "🎮 Wait for sales on games and entertainment purchases",
                "🏠 Host game nights instead of going out"
            ],
            'shopping_pos': [
                "🛍️ Wait 24 hours before making non-essential purchases",
                "🏷️ Shop end-of-season sales for clothing",
                "💳 Use cashback credit cards for purchases you'd make anyway",
                "🔍 Compare prices online before buying in-store",
                "👕 Shop your closet first - you might already have what you need",
                "🛒 Set a monthly shopping budget and track it"
            ],
            'shopping_net': [
                "🛒 Abandon your cart and wait for discount emails",
                "📦 Avoid impulse purchases by removing saved payment methods",
                "🚚 Group purchases to qualify for free shipping",
                "💰 Use browser extensions to find coupon codes automatically",
                "📱 Compare prices across multiple websites",
                "⏰ Shop during major sale events like Black Friday"
            ],
            'health_fitness': [
                "🏃 Use free workout videos instead of expensive gym classes",
                "🏥 Use generic medications when available",
                "👀 Get regular checkups to prevent costly health issues",
                "🦷 Practice good dental hygiene to avoid expensive procedures",
                "💊 Use pharmacy discount programs and apps",
                "🏋️ Look for community fitness programs and free trials"
            ],
            'personal_care': [
                "✂️ Extend time between salon visits by learning basic maintenance",
                "💅 Do your own manicures and pedicures at home",
                "🧴 Buy personal care items in bulk or during sales",
                "💄 Use drugstore alternatives to expensive beauty products",
                "🧼 Make your own face masks and scrubs with natural ingredients"
            ],
            'travel': [
                "✈️ Book flights 6-8 weeks in advance for best prices",
                "🏨 Use hotel comparison sites and book directly for perks",
                "🎒 Pack light to avoid baggage fees",
                "🍽️ Eat like a local instead of at tourist restaurants",
                "🚗 Consider alternative accommodations like Airbnb",
                "📱 Use travel apps to find deals and discounts"
            ],
            'home': [
                "🔧 Learn basic DIY repairs to avoid service calls",
                "💡 Switch to LED bulbs to reduce electricity costs",
                "🌡️ Use a programmable thermostat to save on heating/cooling",
                "🛒 Shop at discount home improvement stores",
                "🔨 Buy quality tools once instead of cheap ones repeatedly",
                "🏠 Regular maintenance prevents costly repairs later"
            ],
            'kids_pets': [
                "👶 Buy children's clothes at consignment shops",
                "🎮 Look for free or low-cost activities for kids",
                "🐕 Groom pets at home instead of professional grooming",
                "💉 Keep up with pet vaccinations to prevent expensive illnesses",
                "📚 Use library programs for kids' entertainment",
                "🧸 Buy toys during post-holiday sales"
            ],
            'misc_pos': [
                "💳 Avoid ATM fees by using your bank's ATMs",
                "📦 Buy household items in bulk to reduce per-unit cost",
                "🏪 Shop at dollar stores for basic household items",
                "📱 Use apps to find the best prices on everyday items"
            ],
            'misc_net': [
                "💻 Cancel unused subscriptions and memberships",
                "📱 Review your digital subscriptions monthly",
                "💾 Use free alternatives to paid software when possible",
                "☁️ Choose the right cloud storage plan for your needs"
            ]
        }
        
        # General financial tips
        self.general_tips = [
            "💰 Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings",
            "🏦 Build an emergency fund covering 3-6 months of expenses",
            "📊 Track your spending daily to identify patterns",
            "🎯 Set specific savings goals and automate transfers",
            "💳 Pay off high-interest debt first to save on interest",
            "🔍 Review and negotiate bills annually (insurance, phone, internet)",
            "📈 Start investing early, even small amounts compound over time",
            "🛡️ Avoid lifestyle inflation when your income increases",
            "📱 Use budgeting apps to monitor your financial health",
            "🎓 Invest in skills that can increase your earning potential"
        ]

    def analyze_spending_patterns(self, transactions_df: pd.DataFrame) -> Dict:
        """Analyze user's spending patterns and identify areas for improvement"""
        if transactions_df.empty:
            return {"error": "No transactions to analyze"}
        
        # Convert date column
        transactions_df['trans_date_trans_time'] = pd.to_datetime(transactions_df['trans_date_trans_time'])
        
        # Calculate spending by category
        category_spending = transactions_df.groupby('category')['amt'].agg(['sum', 'count', 'mean']).round(2)
        category_spending = category_spending.sort_values('sum', ascending=False)
        
        # Calculate monthly spending
        transactions_df['month'] = transactions_df['trans_date_trans_time'].dt.to_period('M')
        monthly_spending = transactions_df.groupby('month')['amt'].sum()
        
        # Identify high-spending categories (top 3)
        top_categories = category_spending.head(3).index.tolist()
        
        # Calculate spending trends
        recent_month = monthly_spending.index[-1] if len(monthly_spending) > 0 else None
        previous_month = monthly_spending.index[-2] if len(monthly_spending) > 1 else None
        
        trend = "stable"
        if recent_month and previous_month:
            recent_spending = monthly_spending[recent_month]
            previous_spending = monthly_spending[previous_month]
            change_pct = ((recent_spending - previous_spending) / previous_spending) * 100
            
            if change_pct > 10:
                trend = "increasing"
            elif change_pct < -10:
                trend = "decreasing"
        
        return {
            "category_spending": category_spending.to_dict('index'),
            "monthly_spending": monthly_spending.to_dict(),
            "top_categories": top_categories,
            "trend": trend,
            "total_spending": transactions_df['amt'].sum(),
            "avg_transaction": transactions_df['amt'].mean(),
            "transaction_count": len(transactions_df)
        }

    def get_personalized_tips(self, transactions_df: pd.DataFrame, monthly_income: float = None) -> Dict:
        """Generate personalized savings tips based on user's spending patterns"""
        analysis = self.analyze_spending_patterns(transactions_df)
        
        if "error" in analysis:
            return {"tips": self.general_tips[:3], "analysis": analysis}
        
        personalized_tips = []
        
        # Tips for top spending categories
        for category in analysis["top_categories"]:
            if category in self.category_tips:
                # Get 2 random tips for this category
                category_tips = np.random.choice(self.category_tips[category], 
                                               min(2, len(self.category_tips[category])), 
                                               replace=False).tolist()
                personalized_tips.extend(category_tips)
        
        # Add spending trend tips
        if analysis["trend"] == "increasing":
            personalized_tips.append("📈 Your spending has increased recently. Review your recent purchases to identify unnecessary expenses.")
        elif analysis["trend"] == "decreasing":
            personalized_tips.append("📉 Great job! Your spending has decreased recently. Keep up the good work!")
        
        # Budget-based tips
        if monthly_income:
            monthly_spending = analysis["total_spending"] / max(1, len(analysis["monthly_spending"]))
            spending_ratio = monthly_spending / monthly_income
            
            if spending_ratio > 0.8:
                personalized_tips.append("⚠️ You're spending over 80% of your income. Consider reducing expenses or increasing income.")
            elif spending_ratio < 0.5:
                personalized_tips.append("💰 You're doing great with spending less than 50% of your income. Consider increasing your savings rate!")
        
        # Add some general tips
        general_tips_sample = np.random.choice(self.general_tips, 3, replace=False).tolist()
        personalized_tips.extend(general_tips_sample)
        
        return {
            "tips": personalized_tips[:8],  # Limit to 8 tips
            "analysis": analysis
        }

    def get_category_insights(self, category: str, amount: float) -> List[str]:
        """Get insights for a specific category and amount"""
        insights = []
        
        if category in self.category_tips:
            # Get 2 tips for this category
            tips = np.random.choice(self.category_tips[category], 
                                  min(2, len(self.category_tips[category])), 
                                  replace=False).tolist()
            insights.extend(tips)
        
        # Amount-based insights
        if amount > 100:
            insights.append(f"💡 This is a significant expense (${amount:.2f}). Consider if it's necessary or if you can find a cheaper alternative.")
        elif amount < 5:
            insights.append(f"💰 Small expenses like this (${amount:.2f}) can add up. Track them to see your total spending in this category.")
        
        return insights[:3]  # Limit to 3 insights