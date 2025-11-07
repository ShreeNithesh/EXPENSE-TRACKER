import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import ExpenseForm from './ExpenseForm';
import TrendsChart from './TrendsChart';
import Suggestions from './Suggestions';

function Dashboard() {
  const { userId } = useParams();
  const [transactions, setTransactions] = useState([]);
  const [trends, setTrends] = useState(null);
  const [suggestions, setSuggestions] = useState(null);

  useEffect(() => {
    fetchData();
  }, [userId]);

  const fetchData = async () => {
    const txRes = await fetch(`http://localhost:5000/transactions/${userId}`);
    setTransactions(await txRes.json());
    const trendsRes = await fetch(`http://localhost:5000/trends/${userId}`);
    setTrends(await trendsRes.json());
    const sugRes = await fetch(`http://localhost:5000/suggestions/${userId}`);
    setSuggestions(await sugRes.json());
  };

  const handleAdd = () => fetchData();

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
      <ExpenseForm userId={userId} onAdd={handleAdd} />
      <h2 className="text-2xl mt-8 mb-4">Transactions</h2>
      <table className="w-full border-collapse border">
        <thead>
          <tr>
            <th className="border p-2">Date</th>
            <th className="border p-2">Merchant</th>
            <th className="border p-2">Amount</th>
            <th className="border p-2">Category</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map(tx => (
            <tr key={tx.id}>
              <td className="border p-2">{tx.date}</td>
              <td className="border p-2">{tx.merchant}</td>
              <td className="border p-2">${tx.amount.toFixed(2)}</td>
              <td className="border p-2">{tx.category}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2 className="text-2xl mt-8 mb-4">Expenditure Trends</h2>
      {trends && <TrendsChart trends={trends} />}
      <h2 className="text-2xl mt-8 mb-4">Suggestions to Save Money</h2>
      {suggestions && <Suggestions suggestions={suggestions} />}
    </div>
  );
}

export default Dashboard;