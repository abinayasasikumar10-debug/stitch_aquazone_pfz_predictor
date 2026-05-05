/**
 * AquaZone PFZ Predictor – Map interaction
 * Handles the Leaflet map on the index page.
 */

(function () {
  "use strict";

  // ── Initialise map ──────────────────────────────────────────────────────
  var map = L.map("map").setView([13.0, 80.0], 5);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 18,
  }).addTo(map);

  // ── Marker (created lazily on first click) ──────────────────────────────
  var marker = null;

  function setCoords(lat, lng) {
    var latInput = document.getElementById("latitude");
    var lonInput = document.getElementById("longitude");
    var coordsDisplay = document.getElementById("map-coords");

    latInput.value = lat.toFixed(4);
    lonInput.value = lng.toFixed(4);
    coordsDisplay.textContent =
      "Selected: " + lat.toFixed(4) + "°N,  " + lng.toFixed(4) + "°E";

    if (marker) {
      marker.setLatLng([lat, lng]);
    } else {
      marker = L.marker([lat, lng]).addTo(map);
    }
    marker.bindPopup(lat.toFixed(4) + "°N, " + lng.toFixed(4) + "°E").openPopup();
  }

  // ── Click on map ─────────────────────────────────────────────────────────
  map.on("click", function (e) {
    setCoords(e.latlng.lat, e.latlng.lng);
  });

  // ── Quick-location chips ─────────────────────────────────────────────────
  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var lat = parseFloat(this.dataset.lat);
      var lng = parseFloat(this.dataset.lon);
      map.setView([lat, lng], 7);
      setCoords(lat, lng);
    });
  });
})();
