import React, { useState } from 'react';

function ExpenseForm({ userId, onAdd }) {
  const [merchant, setMerchant] = useState('');
  const [amount, setAmount] = useState('');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/add_expense', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, merchant, amount: parseFloat(amount), date })
      });
      const data = await res.json();
      if (res.ok) {
        setResult(data);
        onAdd();
      } else {
        setError(data.error);
      }
    } catch {
      setError('Server error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl mb-4">Add Expense</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          value={merchant}
          onChange={(e) => setMerchant(e.target.value)}
          placeholder="Merchant"
          className="block w-full p-2 border rounded"
          required
        />
        <input
          type="number"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          placeholder="Amount"
          className="block w-full p-2 border rounded"
          step="0.01"
          required
        />
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="block w-full p-2 border rounded"
        />
        <button disabled={loading} className="w-full bg-blue-600 text-white p-2 rounded disabled:bg-gray-400">
          {loading ? 'Adding...' : 'Add and Categorize'}
        </button>
      </form>
      {result && (
        <div className="mt-4 p-4 bg-green-100 rounded">
          <p><strong>Category:</strong> {result.category}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
      {error && <div className="mt-4 p-4 bg-red-100 rounded">{error}</div>}
    </div>
  );
}

export default ExpenseForm;