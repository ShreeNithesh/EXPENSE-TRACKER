import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Home() {
  const [newName, setNewName] = useState('');
  const [existingName, setExistingName] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName })
      });
      const data = await res.json();
      if (res.ok) {
        navigate(`/dashboard/${data.user_id}`);
      } else {
        setError(data.error);
      }
    } catch {
      setError('Server error');
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: existingName })
      });
      const data = await res.json();
      if (res.ok) {
        navigate(`/dashboard/${data.user_id}`);
      } else {
        setError(data.error);
      }
    } catch {
      setError('Server error');
    }
  };

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-lg">
        <h1 className="text-2xl font-bold mb-6 text-center">Expense Tracker</h1>
        {error && <p className="text-red-500 mb-4">{error}</p>}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h2 className="text-xl mb-4">New User</h2>
            <form onSubmit={handleRegister} className="space-y-4">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Enter your name"
                className="block w-full p-2 border rounded"
                required
              />
              <button type="submit" className="w-full bg-green-600 text-white p-2 rounded">Register</button>
            </form>
          </div>
          <div>
            <h2 className="text-xl mb-4">Existing User</h2>
            <form onSubmit={handleLogin} className="space-y-4">
              <input
                type="text"
                value={existingName}
                onChange={(e) => setExistingName(e.target.value)}
                placeholder="Enter your name"
                className="block w-full p-2 border rounded"
                required
              />
              <button type="submit" className="w-full bg-blue-600 text-white p-2 rounded">Login</button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;