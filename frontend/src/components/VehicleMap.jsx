// import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
// import "leaflet/dist/leaflet.css";
// import L from "leaflet";

// // fix icon default của leaflet
// delete L.Icon.Default.prototype._getIconUrl;
// L.Icon.Default.mergeOptions({
//   iconRetinaUrl:
//     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
//   iconUrl:
//     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
//   shadowUrl:
//     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
// });

// export default function VehicleMap({ route }) {
//   if (!route || route.length === 0) return null;

//   // lấy vị trí đầu tiên làm trung tâm
//   const firstLat = route[0].lat || 0;
//   const firstLon = route[0].lon || 0;

//   return (
//     <MapContainer
//       center={[firstLat, firstLon]}
//       zoom={15}
//       style={{ height: "400px", width: "100%", marginTop: "1rem" }}
//     >
//       <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
//       {route.map((point, idx) => (
//         point.lat && point.lon && (
//           <Marker key={idx} position={[point.lat, point.lon]}>
//             <Popup>
//               {point.timestamp} <br />
//               {point.camera_id} - {point.location}
//             </Popup>
//           </Marker>
//         )
//       ))}
//     </MapContainer>
//   );
// }

import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect } from "react";
import L from "leaflet";

// Fix icon mặc định của Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

// Component dùng để update vị trí map khi route thay đổi
function MapUpdater({ route }) {
  const map = useMap();

  useEffect(() => {
    if (route && route.length > 0) {
      const { lat, lon } = route[0];
      if (lat && lon) map.setView([lat, lon], 15);
    }
  }, [route, map]);

  return null;
}

export default function VehicleMap({ route }) {
  return (
    <MapContainer
      center={[10.762622, 106.660172]} // vị trí mặc định HCM
      zoom={13}
      style={{ height: "500px", width: "100%", marginTop: "1rem" }}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {/* Tự update vị trí map khi route thay đổi */}
      <MapUpdater route={route} />

      {/* Vẽ marker của route */}
      {route &&
        route.map((p, idx) => 
          p.lat && p.lon && (
            <Marker key={idx} position={[p.lat, p.lon]}>
              <Popup>
                <strong>Thời gian:</strong> {new Date(p.timestamp).toLocaleString("vi-VN")} <br />
                {/* <strong>Camera:</strong> {p.camera_id} <br />
                <strong>Vị trí:</strong> {p.location} */}
              </Popup>
            </Marker>
          )
        )}
    </MapContainer>
  );
}
