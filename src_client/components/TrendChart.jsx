import React from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

function TrendsChart({ trends }) {
  const lineOptions = {
    responsive: true,
    plugins: { legend: { position: 'top' } }
  };

  const monthlyData = {
    labels: trends.monthly.map(m => m.period),
    datasets: [{ label: 'Monthly Total', data: trends.monthly.map(m => m.total), borderColor: 'blue' }]
  };

  const weeklyData = {
    labels: trends.weekly.map(w => w.period),
    datasets: [{ label: 'Weekly Total', data: trends.weekly.map(w => w.total), borderColor: 'green' }]
  };

  const yearlyData = {
    labels: trends.yearly.map(y => y.period),
    datasets: [{ label: 'Yearly Total', data: trends.yearly.map(y => y.total), borderColor: 'red' }]
  };

  const categoryData = {
    labels: trends.categories.map(c => c.category),
    datasets: [{ label: 'Category Total', data: trends.categories.map(c => c.total), backgroundColor: 'orange' }]
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <h3>Monthly Trends</h3>
        <Line data={monthlyData} options={lineOptions} />
      </div>
      <div>
        <h3>Weekly Trends</h3>
        <Line data={weeklyData} options={lineOptions} />
      </div>
      <div>
        <h3>Yearly Trends</h3>
        <Line data={yearlyData} options={lineOptions} />
      </div>
      <div>
        <h3>Category Breakdown</h3>
        <Bar data={categoryData} options={lineOptions} />
      </div>
    </div>
  );
}

export default TrendsChart;