# Thermal Digital Twin for Data Centres

Interactive thermal digital twin visualization tool developed for data centres.

## Features

- **3D Point Cloud Visualization**: Three-dimensional temperature and airflow distribution of the data centre
- **2D Slice Analysis**: Sliceable views along X, Y, Z axes
- **Interactive Controls**: Slice thickness, grid resolution, and point count adjustments
- **KPI Monitoring**: Data centre performance indicators (RCI, RTI, RHI, RI, CCI)

## Usage

### Visualization Options

1. **Field Selection**: Temperature (°C) or Airflow Velocity (m/s)
2. **View Mode**: 
   - 3D Scatter: View of all data points
   - 2D Slice: Cross-section view on a specific axis

### 2D Slice Settings

- **Slice Axis**: Select slice plane (X/Y/Z)
- **Slice Thickness**: Thickness of the slice (1-5%)
- **Grid Resolution**: Interpolation resolution (500-2000)

## Data Structure

The NPZ file must contain the following data:
- `points`: Mesh point coordinates
- `cells`: Cell connectivity information
- `cell_types`: PyVista cell types
- `T`: Temperature field (Kelvin)
- `U`: Velocity vector field (m/s)

---

**Developed by**: DSTECH Team

---

## License

Copyright © 2026 Design and Simulation Technologies Inc.

DC-T2 Community Edition is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

Commercial use, including offering DC-T2 as a managed service, enterprise deployment, or integration into proprietary systems, requires a separate commercial license. If you require a commercial license for proprietary or enterprise use (e.g., integrating this software into a closed-source product, offering it as a managed service, or deploying it in an enterprise environment without AGPL obligations), please contact **Design and Simulation Technologies Inc.** regarding our Enterprise License options.
