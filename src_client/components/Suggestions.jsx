import React from 'react';

function Suggestions({ suggestions }) {
  return (
    <div className="space-y-4">
      <h3 className="text-xl">Specific Tips for Top Categories</h3>
      <ul className="list-disc pl-5">
        {suggestions.specific.map((s, i) => (
          <li key={i}><strong>{s.category}:</strong> {s.tip}</li>
        ))}
      </ul>
      <h3 className="text-xl">General Savings Tips</h3>
      <ul className="list-disc pl-5">
        {suggestions.general.map((g, i) => (
          <li key={i}>{g}</li>
        ))}
      </ul>
    </div>
  );
}

export default Suggestions;