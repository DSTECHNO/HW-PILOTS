# Thermal Digital Twin for EMPA Data Centre

Interactive thermal digital twin visualization tool developed for Aalborg University data centre.

## Features

- **3D Point Cloud Visualization**: Three-dimensional temperature and airflow distribution of the data centre
- **2D Slice Analysis**: Sliceable views along X, Y, Z axes
- **Interactive Controls**: Slice thickness, grid resolution, and point count adjustments
- **KPI Monitoring**: Data centre performance indicators (RCI, RTI, RHI, RI, CCI)

## Installation

```bash
pip install streamlit numpy plotly pyvista scipy
```

## Required Files

- `viewer.py` - Main application
- `validationCase.npz` - CFD simulation data (mesh, temperature, velocity)

## Running the Application

```bash
streamlit run viewer.py
```

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
