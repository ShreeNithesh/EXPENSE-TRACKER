const express = require('express');
const { PythonShell } = require('python-shell');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();

const app = express();
app.use(cors());
app.use(express.json());

const db = new sqlite3.Database('./expenses.db');
db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT UNIQUE)`);
  db.run(`CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY, user_id INTEGER, merchant TEXT, amount REAL, category TEXT, date TEXT)`);
});

app.post('/register', (req, res) => {
  const { name } = req.body;
  db.run(`INSERT INTO users (name) VALUES (?)`, [name], function (err) {
    if (err) return res.status(400).json({ error: 'Name already exists' });
    res.json({ user_id: this.lastID });
  });
});

app.post('/login', (req, res) => {
  const { name } = req.body;
  db.get(`SELECT id FROM users WHERE name = ?`, [name], (err, row) => {
    if (err || !row) return res.status(404).json({ error: 'User not found' });
    res.json({ user_id: row.id });
  });
});

app.post('/add_expense', (req, res) => {
  const { user_id, merchant, amount, date } = req.body;
  PythonShell.run('src/prediction.py', { mode: 'text', args: [merchant, amount] }, (err, results) => {
    if (err) return res.status(500).json({ error: 'Prediction failed' });
    const [category, confidence] = results[0].split(',');
    const tx_date = date || new Date().toISOString();
    db.run(`INSERT INTO transactions (user_id, merchant, amount, category, date) VALUES (?, ?, ?, ?, ?)`,
      [user_id, merchant, parseFloat(amount), category.trim(), tx_date],
      (err) => {
        if (err) return res.status(500).json({ error: 'Insert failed' });
        res.json({ category: category.trim(), confidence: parseFloat(confidence) });
      }
    );
  });
});

app.get('/transactions/:user_id', (req, res) => {
  db.all(`SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC`, [req.params.user_id], (err, rows) => {
    if (err) return res.status(500).json({ error: 'Query failed' });
    res.json(rows);
  });
});

app.get('/trends/:user_id', (req, res) => {
  const user_id = req.params.user_id;
  db.all(`SELECT strftime('%Y-%m', date) as period, SUM(amount) as total FROM transactions WHERE user_id = ? GROUP BY period ORDER BY period`, [user_id], (err, monthly) => {
    if (err) return res.status(500).json({ error: 'Query failed' });
    db.all(`SELECT strftime('%Y-%W', date) as period, SUM(amount) as total FROM transactions WHERE user_id = ? GROUP BY period ORDER BY period`, [user_id], (err, weekly) => {
      db.all(`SELECT strftime('%Y', date) as period, SUM(amount) as total FROM transactions WHERE user_id = ? GROUP BY period ORDER BY period`, [user_id], (err, yearly) => {
        db.all(`SELECT category, SUM(amount) as total FROM transactions WHERE user_id = ? GROUP BY category ORDER BY total DESC`, [user_id], (err, categories) => {
          res.json({ monthly, weekly, yearly, categories });
        });
      });
    });
  });
});

app.get('/suggestions/:user_id', (req, res) => {
  db.all(`SELECT category, SUM(amount) as total FROM transactions WHERE user_id = ? GROUP BY category ORDER BY total DESC LIMIT 3`, [req.params.user_id], (err, rows) => {
    if (err) return res.status(500).json({ error: 'Query failed' });
    const categoryTips = {
      'grocery_pos': 'Use coupons, buy in bulk, plan meals to reduce grocery expenses.',
      'gas_transport': 'Use public transport, carpool, or bike to cut down on gas costs.',
      'food_dining': 'Eat out less, cook at home, and use home-brewed coffee.',
      'entertainment': 'Opt for free activities or streaming deals; cancel unused subscriptions.',
      'healthcare': 'Shop around for better insurance or use generic medications.',
      'hotels': 'Book in advance, use rewards points, or choose budget accommodations.',
      'misc_net': 'Review online subscriptions and cancel unnecessary ones.',
      'misc_pos': 'Track miscellaneous purchases and set a monthly limit.',
      'personal_care': 'Buy generics and look for sales on personal items.',
      'shopping_net': 'Use cashback sites and compare prices online.',
      'shopping_pos': 'Make a shopping list to avoid impulse buys.',
      'travel': 'Travel off-peak, use points, or choose staycations.',
      'utilities': 'Reduce energy use with efficient appliances and unplug devices.',
      'grocery_net': 'Similar to groceries: plan and bulk buy online.'
    };
    const generalTips = [
      'Create a budget and track your spending daily.',
      'Set savings goals and automate transfers to a high-yield savings account.',
      'Pay off high-interest debt first to save on interest.',
      'Follow the 50/30/20 rule: 50% on needs, 30% on wants, 20% on savings/debt.',
      'Build an emergency fund covering 3-6 months of expenses.',
      'Shop around for better deals on phone, streaming, and insurance services.',
      'Automate bill payments to avoid late fees.'
    ];
    const specific = rows.map(row => ({
      category: row.category,
      tip: categoryTips[row.category] || 'Monitor and reduce spending in this category.'
    }));
    res.json({ specific, general: generalTips });
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server on port ${PORT}`));