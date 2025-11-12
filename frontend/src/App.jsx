import { useState } from 'react';
import axios from "axios";
import VehicleMap from "./components/VehicleMap";
import './App.css'

function App() {
  const [plate, setPlate] = useState("");
  const [route, setRoute] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.get(`http://localhost:8000/vehicle/${plate}`);
      setRoute(res.data.route);
    } catch (err) {
      setRoute([]);
      setError(err.response?.data?.detail || "Error fetching data");
    } finally {
      setLoading(false);
    }
  };

   return (
    <div style={{ padding: "2rem" }}>
      <h1>Vehicle Tracking</h1>
      <div>
        <input
          type="text"
          placeholder="Enter plate"
          value={plate}
          onChange={(e) => setPlate(e.target.value)}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
      {error && <p style={{ color: "red" }}>{error}</p>}

      {route.length > 0 && (
        <>
          <h2>Route ({route.length} points)</h2>
          <VehicleMap route={route} />
          <table border="1" cellPadding="5" style={{ marginTop: "1rem" }}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Camera</th>
                <th>Location</th>
                <th>Lat</th>
                <th>Lon</th>
              </tr>
            </thead>
            <tbody>
              {route.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.timestamp}</td>
                  <td>{r.camera_id}</td>
                  <td>{r.location}</td>
                  <td>{r.lat}</td>
                  <td>{r.lon}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

export default App
