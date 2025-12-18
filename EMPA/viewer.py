import os
import gc
import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import griddata


# =================================================
# 0) STREAMLIT SETTINGS  (MUST BE FIRST STREAMLIT CALL)
# =================================================
st.set_page_config(
    page_title="Thermal Digital Twin for EMPA Pilot",
    layout="wide"
)


# =================================================
# 1) PASSWORD AUTHENTICATION
# =================================================
def check_password():
    """Returns True if the user has the correct password."""
    def password_entered():
        if st.session_state.get("password", "") == st.secrets.get("password", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Please enter the password:",
        type="password",
        on_change=password_entered,
        key="password"
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect. Please try again.")

    return False


if not check_password():
    st.stop()


# =================================================
# 2) GLOBAL CSS
# =================================================
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { margin-top: 0rem; padding-top: 0rem; }
    h1 { margin-top: 0rem; padding-top: 0rem; }
    .element-container { margin-top: 0rem; }

    /* Sidebar slider color (track + active range + handle border) */
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div {
        background: #dbeafe;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: #2563eb;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background: #ffffff;
        border: 2px solid #2563eb;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.35);
    }

    /* Sidebar info card */
    .sidebar-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
        border: 1px solid #e5e7eb;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 13px;
    }
    .sidebar-card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .sidebar-card-metric {
        font-size: 13px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.1rem;
    }

    /* KPI table */
    .kpi-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .kpi-table th {
        background-color: #2c3e50;
        color: white;
        padding: 10px 6px;
        text-align: left;
        font-weight: bold;
        font-size: 13px;
    }
    .kpi-table td {
        padding: 10px 6px;
        border-bottom: 1px solid #ddd;
        font-size: 13px;
    }
    .kpi-table tr:hover { background-color: #f5f5f5; }
    .powered-by { margin-top: 8px; margin-bottom: 2px; font-weight: bold; font-size: 20px; }
    .website-link { margin-top: 0px; font-size: 13px; }
    .kpi-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# =================================================
# 3) TAB CHANGE CLEANUP (CRITICAL STABILITY)
# =================================================
def on_tab_change():
    # Remove large cached UI results from session_state
    for k in [
        "last_slice_key",
        "last_slice_result",
        "last_fig_marker",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    gc.collect()


# =================================================
# 4) SIDEBAR HEADER + TAB SELECTION (EARLY!)
# =================================================
st.sidebar.markdown(
    """
    <h2 style='text-align: center; font-size:20px; margin-bottom: 0.2rem;'>
        DC-T²: Data Centre Thermal Twin
    </h2>
    <p style='text-align: left; font-size:14px; margin-top: 0rem; color:#6b7280;'>
        CFD-Enabled Thermal Twin for Data Centres
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

view_tab = st.sidebar.radio(
    "",
    ["Thermal Twin", "About"],
    index=0,
    key="view_tab",
    on_change=on_tab_change
)

# Optional: stability reset button
if st.sidebar.button("🧹 Clear caches (stability)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in list(st.session_state.keys()):
        if k not in ["password_correct", "view_tab"]:
            del st.session_state[k]
    st.rerun()


# =================================================
# 5) ABOUT PAGE (NO NETWORK/VTK/NPZ LOADS HERE!)
# =================================================
def render_about_page():
    st.title("Thermal Twin for EMPA Pilot")
    st.image(
        # Deploy-safe: URL (NOT local file)
        "https://raw.githubusercontent.com/DSTECHNO/HW-PILOTS/main/EMPA/empa.png",
        caption="EMPA NEST",
        width=500
    )

    st.markdown("""
### Facility Information
- **Location:** Empa NEST Building, Switzerland
- **Cooling Technology:** Hybrid cooling (air+liquid) infrastructure

### IT Infrastructure
- **Number of Server Racks:** 32
- **Total IT Capacity:** 2.83 kW
- **Rack Power Range:** 68 W - 309 W
- **Configuration:** Variable thermal loads

### Cooling System
- **Cooling Method:** Air-based cooling units
- **Number of Cooling Units:** 4
- **Total Cooling Capacity:** 5 kW (each one is 1250 W)
- **Air Flow Rate:** 0.207 - 1.553 kg/s per unit

### Operating Conditions
- **Ambient Temperature:** 24.5°C
- **Inlet Air Temperature:** 23.50 - 27.50°C
- **Outlet Air Temperature:** 26.50 - 54.00°C
- **Rack Air Flow:** 0.0091 - 0.0148 kg/s
""")

    st.markdown("""
### Scientific Validation & Methodology

The numerical model was validated using experimental test cases conducted at the EMPA NEST pilot data centre under different server utilization levels, workload distributions, total IT loads and cooling settings. During each test, cooling system parameters such as fan speed and valve opening ratio were adjusted to ensure stable operation, and IPMI-based measurements of air and water temperatures were recorded over 15-minute intervals and time-averaged. The same operating conditions, including measured cooling coil inlet water temperatures, were applied as boundary conditions in the CFD simulations. Simulated air temperatures and water outlet temperatures at the cooling coil outlets were directly compared with measurements, showing good agreement across all cases. The validation confirms that the Conjugate Heat Transfer (CHT) model accurately captures airflow recirculation, air–water heat exchange in the cooling coil, and temperature distributions within the data centre under realistic operating conditions. For a comprehensive analysis of the methodology and results, please refer to our published research.

**Maximizing waste heat recovery from a building-integrated edge data center** - [Read the paper](https://www.nature.com/articles/s41598-025-22498-x)
""")

    st.markdown("""
---
<h3 style='font-style: italic;'>D&S Tech | Digital Twin Solutions</h3>
<p style='font-size: 14px;'><a href='https://dstechs.net/' target='_blank'>https://dstechs.net/</a></p>
<p style='font-size: 16px;'>Get Your Thermal Digital Twin. Contact us today: <strong>datacenter@dstechs.net</strong></p>
""", unsafe_allow_html=True)

    # Sidebar footer (About)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <p style='font-size: 12px; color:#6b7280; font-weight:600; margin-bottom:4px;'>
            Developed by D&STECH © 2025
        </p>
        <p style='font-size: 11px; color:#6b7280; line-height:1.4;'>
            This tool was developed as part of the Heatwise Project. The Heatwise Project has received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101138491 and the Swiss Secretariat for Education, Research, and Innovation (SERI) under contract No 23.00606.
        </p>
        """,
        unsafe_allow_html=True
    )

    logo_col1, logo_col2 = st.sidebar.columns(2)
    with logo_col1:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <a href='https://heatwise.eu' target='_blank'>
                <img src='https://raw.githubusercontent.com/DSTECHNO/HW-PILOTS/main/AAU/heatwise_logo.svg'
                     alt='HEATWISE' style='width: 100%; max-width: 120px;'>
            </a>
        </div>
        """, unsafe_allow_html=True)
    with logo_col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <a href='https://dstechs.net/' target='_blank'>
                <img src='https://raw.githubusercontent.com/DSTECHNO/HW-PILOTS/main/AAU/dstech_logo.png'
                     alt='D&S Tech' style='width: 100%; max-width: 120px;'>
            </a>
        </div>
        """, unsafe_allow_html=True)


if view_tab == "About":
    render_about_page()
    st.stop()


# =================================================
# 6) THERMAL TWIN PAGE (HEAVY WORK STARTS HERE)
# =================================================
st.title("Thermal Twin for EMPA Pilot")

HF_USER = "mkuzaay"

NPZ_URL = f"https://huggingface.co/datasets/{HF_USER}/hw-pilots-data/resolve/main/validationCaseEMPA.npz"
VTK_URL = f"https://huggingface.co/datasets/{HF_USER}/hw-pilots-data/resolve/main/validationCaseEMPA.vtk"


@st.cache_data(show_spinner=False)
def ensure_file(url: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()

    tmp = local_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp, local_path)
    return local_path


@st.cache_resource
def load_case_cached(npz_path: str, vtk_path: str):
    """Heavy: reads NPZ(mmap) + reads VTK via pyvista. Cached as resource."""
    import pyvista as pv

    data = np.load(npz_path, mmap_mode="r")
    mesh = pv.read(vtk_path)

    T = data["T"] if "T" in data.files else None
    U = data["U"] if "U" in data.files else None
    return mesh, T, U


@st.cache_resource
def load_outer_geometry_cached(vtk_path: str):
    """Heavy: read VTK, extract outer surface and triangulate. Cached as resource."""
    import pyvista as pv

    mesh = pv.read(vtk_path)
    surface = mesh.extract_surface().triangulate()

    pts = surface.points
    faces = surface.faces.reshape(-1, 4)
    triangles = faces[:, 1:]

    xg = pts[:, 0]
    yg = pts[:, 1]
    zg = pts[:, 2]
    ig = triangles[:, 0]
    jg = triangles[:, 1]
    kg = triangles[:, 2]
    return xg, yg, zg, ig, jg, kg


def get_coords_and_field(mesh, T_field, U_field, field_choice: str):
    if field_choice == "Temperature":
        if T_field is None:
            raise ValueError("NPZ does not contain 'T'.")
        field = T_field - 273.15
        color_label = "T [°C]"
    else:
        if U_field is None:
            raise ValueError("NPZ does not contain 'U'.")
        field = np.linalg.norm(U_field, axis=1)
        color_label = "|U| [m/s]"

    # Decide whether field is per-cell or per-point
    if len(field) == mesh.n_cells:
        pts = mesh.cell_centers().points
    elif len(field) == mesh.n_points:
        pts = mesh.points
    else:
        raise ValueError(f"Field length ({len(field)}) != n_cells ({mesh.n_cells}) and != n_points ({mesh.n_points}).")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return x, y, z, field, color_label


def density_aware_downsample(x, y, z, field, max_points: int, n_side: int = 40):
    N = x.size
    if N <= max_points:
        return x, y, z, field

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())

    ex = xmax - xmin if xmax > xmin else 1e-9
    ey = ymax - ymin if ymax > ymin else 1e-9
    ez = zmax - zmin if zmax > zmin else 1e-9

    ix = ((x - xmin) / ex * n_side).astype(int)
    iy = ((y - ymin) / ey * n_side).astype(int)
    iz = ((z - zmin) / ez * n_side).astype(int)

    ix = np.clip(ix, 0, n_side - 1)
    iy = np.clip(iy, 0, n_side - 1)
    iz = np.clip(iz, 0, n_side - 1)

    key = ix + n_side * (iy + n_side * iz)
    counts = np.bincount(key)
    density = counts[key]

    weights = 1.0 / density
    weights_sum = weights.sum()
    weights = weights / weights_sum if weights_sum > 0 else np.ones_like(weights) / len(weights)

    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=max_points, replace=False, p=weights)
    return x[idx], y[idx], z[idx], field[idx]


def interpolate_slice(axis1_s, axis2_s, f_s, grid_resolution):
    axis1_min, axis1_max = axis1_s.min(), axis1_s.max()
    axis2_min, axis2_max = axis2_s.min(), axis2_s.max()

    grid_axis1 = np.linspace(axis1_min, axis1_max, grid_resolution)
    grid_axis2 = np.linspace(axis2_min, axis2_max, grid_resolution)
    A1, A2 = np.meshgrid(grid_axis1, grid_axis2)

    grid_field = griddata(
        (axis1_s, axis2_s),
        f_s,
        (A1, A2),
        method="linear",
        fill_value=np.nan
    )

    nan_mask = np.isnan(grid_field)
    if nan_mask.any():
        grid_field_nearest = griddata(
            (axis1_s, axis2_s),
            f_s,
            (A1, A2),
            method="nearest"
        )
        grid_field[nan_mask] = grid_field_nearest[nan_mask]

    return grid_axis1, grid_axis2, grid_field


# ---- Download only on Thermal Twin page ----
npz_path = ensure_file(NPZ_URL, "validationCaseEMPA.npz")
vtk_path = ensure_file(VTK_URL, "validationCaseEMPA.vtk")

mesh, T_field, U_field = load_case_cached(npz_path, vtk_path)

# =================================================
# 7) SIDEBAR CONTROLS (Thermal Twin)
# =================================================
field_choice = st.sidebar.selectbox("Field to Display", ["Temperature", "Airflow Velocity"], index=0)
mode = st.sidebar.selectbox("View Mode", ["3D Scatter", "2D Slice"], index=0)

# Slice controls
if mode == "2D Slice":
    slice_axis = st.sidebar.selectbox("Slice Axis", ["X", "Y", "Z"], index=2)
    thickness_percent = st.sidebar.slider(
        "Slice Thickness (%)",
        1, 5, 3, 1,
        help="Thicker slice = more points but less precise location."
    )
    grid_resolution = st.sidebar.slider(
        "Grid Resolution",
        400, 1500, 800, 100,
        help="Higher = sharper but slower. 700-900 recommended."
    )
else:
    slice_axis = None
    thickness_percent = 3
    grid_resolution = 800

# Hard stability cap
grid_resolution = min(grid_resolution, 900)

# Extract coords + field
x, y, z, field, color_label = get_coords_and_field(mesh, T_field, U_field, field_choice)
total_cells = x.size

# Range slider
field_min = float(np.nanmin(field))
field_max = float(np.nanmax(field))
low_default = float(np.nanpercentile(field, 0))
high_default = float(np.nanpercentile(field, 100))

value_min, value_max = st.sidebar.slider(
    f"{color_label} Range Filter",
    min_value=field_min,
    max_value=field_max,
    value=(low_default, high_default),
    help=f"Only show points between {color_label} = [{low_default:.2f}, {high_default:.2f}]"
)

# 3D point cap
max_points = st.sidebar.slider(
    "Maximum Points (3D only)",
    min_value=1000,
    max_value=5000,
    value=3000,
    step=500,
    help="More points = slower performance."
)

# Sidebar card
displayed_points = min(total_cells, max_points)
st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Grid Size</div>
        <div class="sidebar-card-metric">Total cells: {total_cells:,}</div>
        <div class="sidebar-card-metric">Displayed: {displayed_points:,}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar footer (Thermal Twin)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <p style='font-size: 12px; color:#6b7280; font-weight:600; margin-bottom:4px;'>
        Developed by D&STECH © 2025
    </p>
    <p style='font-size: 11px; color:#6b7280; line-height:1.4;'>
        This tool was developed as part of the Heatwise Project. The Heatwise Project has received funding from the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101138491 and the Swiss Secretariat for Education, Research, and Innovation (SERI) under contract No 23.00606.
    </p>
    """,
    unsafe_allow_html=True
)

logo_col1, logo_col2 = st.sidebar.columns(2)
with logo_col1:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <a href='https://heatwise.eu' target='_blank'>
            <img src='https://raw.githubusercontent.com/DSTECHNO/HW-PILOTS/main/AAU/heatwise_logo.svg'
                 alt='HEATWISE' style='width: 100%; max-width: 120px;'>
        </a>
    </div>
    """, unsafe_allow_html=True)

with logo_col2:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <a href='https://dstechs.net/' target='_blank'>
            <img src='https://raw.githubusercontent.com/DSTECHNO/HW-PILOTS/main/AAU/dstech_logo.png'
                 alt='D&S Tech' style='width: 100%; max-width: 120px;'>
        </a>
    </div>
    """, unsafe_allow_html=True)


# =================================================
# 8) MAIN LAYOUT
# =================================================
col1, col2 = st.columns([7, 3])

with col1:
    viz_title = "🌡️ Data Centre Thermal Map" if field_choice == "Temperature" else "💨 Airflow Velocity Field"
    st.subheader(viz_title)

    # -----------------------------------------
    # 3D SCATTER
    # -----------------------------------------
    if mode == "3D Scatter":
        x_plot, y_plot, z_plot, f_plot = density_aware_downsample(x, y, z, field, max_points)

        # Range filter
        m = (f_plot >= value_min) & (f_plot <= value_max)
        x_plot, y_plot, z_plot, f_plot = x_plot[m], y_plot[m], z_plot[m], f_plot[m]

        fig = go.Figure()

        # Geometry shell (cached)
        try:
            gx, gy, gz, gi, gj, gk = load_outer_geometry_cached(vtk_path)
            fig.add_trace(go.Mesh3d(
                x=gx, y=gy, z=gz,
                i=gi, j=gj, k=gk,
                opacity=0.25,
                color="gray",
                name="DC Geometry",
                showscale=False,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0),
                hoverinfo="skip"
            ))
        except Exception as e:
            st.warning(f"Outer geometry could not be loaded: {e}")

        fig.add_trace(go.Scatter3d(
            x=x_plot, y=y_plot, z=z_plot,
            mode="markers",
            marker=dict(
                size=1.6,
                color=f_plot,
                colorscale="Turbo",
                cmin=float(value_min),
                cmax=float(value_max),
                opacity=0.7,
                colorbar=dict(
                    title=dict(text=color_label, font=dict(color="black", size=14)),
                    tickfont=dict(color="black", size=12),
                ),
            ),
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<br>' +
                          color_label + ': %{marker.color:.3f}<extra></extra>'
        ))

        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Z [m]",
                bgcolor="white",
                aspectmode="data",
            ),
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="white",
        )

        st.session_state["last_fig_marker"] = True
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # 2D SLICE inside 3D geometry
    # -----------------------------------------
    else:
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        z_min, z_max = float(z.min()), float(z.max())

        if slice_axis == "X":
            coord = x
            coord_min, coord_max = x_min, x_max
            axis1, axis2 = y, z
        elif slice_axis == "Y":
            coord = y
            coord_min, coord_max = y_min, y_max
            axis1, axis2 = x, z
        else:
            coord = z
            coord_min, coord_max = z_min, z_max
            axis1, axis2 = x, y

        default_coord = 0.5 * (coord_min + coord_max)
        slice_coord = st.slider(
            f"{slice_axis}-slice location",
            min_value=coord_min,
            max_value=coord_max,
            value=default_coord,
            step=(coord_max - coord_min) / 200,
        )

        thickness = (thickness_percent / 100.0) * (coord_max - coord_min)
        mask = np.abs(coord - slice_coord) <= thickness

        axis1_s = axis1[mask]
        axis2_s = axis2[mask]
        f_s = field[mask]

        if axis1_s.size == 0:
            st.warning(f"No points at this {slice_axis} location, adjust the slice.")
        else:
            # Auto downscale for huge slice point counts
            if axis1_s.size > 250_000:
                grid_resolution = min(grid_resolution, 600)

            st.caption(
                f"{axis1_s.size:,} points in slice (thickness: {thickness_percent}%) → "
                f"Interpolating to {grid_resolution}x{grid_resolution} grid"
            )

            # Throttle: re-interpolate only if slice params changed
            slice_key = (
                slice_axis,
                float(slice_coord),
                int(thickness_percent),
                int(grid_resolution),
                float(value_min),
                float(value_max),
                field_choice
            )

            if "last_slice_key" not in st.session_state:
                st.session_state["last_slice_key"] = None
                st.session_state["last_slice_result"] = None

            if slice_key != st.session_state["last_slice_key"]:
                ga1, ga2, grid_field = interpolate_slice(axis1_s, axis2_s, f_s, grid_resolution)

                # Filter out-of-range values as NaN for visibility control
                grid_field_filtered = grid_field.copy()
                grid_field_filtered[(grid_field_filtered < value_min) | (grid_field_filtered > value_max)] = np.nan

                st.session_state["last_slice_key"] = slice_key
                st.session_state["last_slice_result"] = (ga1, ga2, grid_field_filtered)

            ga1, ga2, grid_field_filtered = st.session_state["last_slice_result"]
            A1, A2 = np.meshgrid(ga1, ga2)

            if slice_axis == "X":
                X_plane = np.full_like(A1, slice_coord); Y_plane = A1; Z_plane = A2
            elif slice_axis == "Y":
                Y_plane = np.full_like(A1, slice_coord); X_plane = A1; Z_plane = A2
            else:
                Z_plane = np.full_like(A1, slice_coord); X_plane = A1; Y_plane = A2

            fig_slice3d = go.Figure()

            # Geometry shell
            try:
                gx, gy, gz, gi, gj, gk = load_outer_geometry_cached(vtk_path)
                fig_slice3d.add_trace(go.Mesh3d(
                    x=gx, y=gy, z=gz,
                    i=gi, j=gj, k=gk,
                    opacity=0.15,
                    color="gray",
                    name="DC Geometry",
                    showscale=False,
                    hoverinfo="skip"
                ))
            except Exception as e:
                st.warning(f"Outer geometry could not be loaded: {e}")

            # Slice plane
            fig_slice3d.add_trace(go.Surface(
                x=X_plane,
                y=Y_plane,
                z=Z_plane,
                surfacecolor=grid_field_filtered,
                colorscale="Turbo",
                cmin=float(value_min),
                cmax=float(value_max),
                colorbar=dict(
                    title=dict(text=color_label, font=dict(color="black", size=14)),
                    tickfont=dict(color="black", size=12),
                ),
                opacity=0.9,
                name="Slice"
            ))

            fig_slice3d.update_layout(
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(
                    xaxis_title="X [m]",
                    yaxis_title="Y [m]",
                    zaxis_title="Z [m]",
                    bgcolor="white",
                    aspectmode="data",
                ),
                paper_bgcolor="white",
            )

            st.session_state["last_fig_marker"] = True
            st.plotly_chart(fig_slice3d, use_container_width=True)


# =================================================
# 9) STATS + KPI PANEL
# =================================================
with col2:
    st.markdown("<h3 style='font-size: 20px; font-weight: bold;'>📊 Summary Statistics</h3>", unsafe_allow_html=True)

    st.write("Minimum =", f"{np.nanmin(field):.2f}")
    st.write("Maximum =", f"{np.nanmax(field):.2f}")
    st.write("Mean =", f"{np.nanmean(field):.2f}")
    st.write("Std. Dev. =", f"{np.nanstd(field):.2f}")

    st.markdown("---")

    st.markdown("""
    <div class="kpi-title">KPI Assessment for EMPA Data Centre</div>

    <table class="kpi-table">
        <tr>
            <th>KPI</th>
            <th>Value</th>
            <th>Assessment</th>
        </tr>
        <tr>
            <td><strong>RCI<sub>HI</sub></strong></td>
            <td>190.83</td>
            <td>🔥 Hot-air recirculation</td>
        </tr>
        <tr>
            <td><strong>RCI<sub>LO</sub></strong></td>
            <td>248.62</td>
            <td>❄️ Cold-air bypass</td>
        </tr>
        <tr>
            <td><strong>RTI</strong></td>
            <td>96.58</td>
            <td>⚠️ Overcooling</td>
        </tr>
        <tr>
            <td><strong>RHI</strong></td>
            <td>0.95</td>
            <td>🔥 Moderate hot-air recirculation</td>
        </tr>
        <tr>
            <td><strong>RI</strong></td>
            <td>97.82</td>
            <td>🔥 Hot-air recirculation</td>
        </tr>
        <tr>
            <td><strong>CCI</strong></td>
            <td>1.35</td>
            <td>⚠️ Overcooling and inefficient airflow management</td>
        </tr>
        <tr>
            <td><strong>LI</strong></td>
            <td>142.10</td>
            <td>⚠️ Leakage index</td>
        </tr>
    </table>

    <div class="powered-by">Powered by D&S Tech</div>
    <div class="website-link"><a href="https://dstechs.net/" target="_blank">https://dstechs.net/</a></div>
    """, unsafe_allow_html=True)
