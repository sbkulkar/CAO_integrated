"""ElderlyCare AI — Medicine Verification & Fall Safety System
Main Streamlit application entrypoint uniting Medicine Verification and Fall Monitoring.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.medicine_adapter import MedicineSystemAdapter
from modules.event_store import EventStore
from modules.alert_manager import AlertManager
from pages import dashboard, medicine_verification, fall_monitoring, alerts_history, system_information

# Configure Streamlit page
st.set_page_config(
    page_title="ElderlyCare AI — Medicine & Fall Safety",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #1976d2;
        margin-bottom: 10px;
    }
    .badge-match {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-mismatch {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-review {
        background-color: #fffde7;
        color: #f57f17;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-unreadable {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize cached shared resources
@st.cache_resource(show_spinner=False)
def initialize_care_system():
    med_system = MedicineSystemAdapter.get_instance()
    evt_store = EventStore()
    al_manager = AlertManager(evt_store)
    return med_system, evt_store, al_manager

medicine_system, event_store, alert_manager = initialize_care_system()

# Preserve session state across page navigation
if "parsed_prescription" not in st.session_state:
    st.session_state.parsed_prescription = None
if "uploaded_package_data" not in st.session_state:
    st.session_state.uploaded_package_data = []
if "verification_results" not in st.session_state:
    st.session_state.verification_results = None

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/health-calendar.png", width=64)
    st.title("ElderlyCare AI")
    st.caption("Medicine Verification & Fall Safety")

    # Hardware Device Status Badge
    dev = medicine_system.device_info
    if dev.get("cuda_available", False):
        st.success(f"⚡ Device: {dev.get('device_name')} (GPU)")
    else:
        st.info(f"💻 Device: {dev.get('device_name')} (CPU Fallback)")

    st.markdown("---")
    
    # Navigation Radio
    selected_page = st.radio(
        "Navigation Menu:",
        [
            "🏠 Care Dashboard",
            "💊 Medicine Verification",
            "🚨 Fall Safety Monitoring",
            "🔔 Care Alert Centre",
            "ℹ️ System Information"
        ]
    )

    st.markdown("---")
    st.subheader("📊 System Status")
    stats = event_store.get_statistics()
    st.write(f"🚨 **Active Falls:** {stats.get('new_falls', 0)}")
    st.write(f"💊 **Prescriptions Audited:** {stats.get('total_medicine_checks', 0)}")

    try:
        conn = medicine_system.catalogue.connect()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM medicines;")
        cat_count = cur.fetchone()[0]
        st.caption(f"Catalogue: {cat_count:,} items")
    except Exception:
        pass

    st.markdown("---")
    if st.button("🔄 Reset Active Session", use_container_width=True):
        st.session_state.parsed_prescription = None
        st.session_state.uploaded_package_data = []
        st.session_state.verification_results = None
        st.rerun()

# Route to selected page
if selected_page == "🏠 Care Dashboard":
    dashboard.render(medicine_system, event_store)
elif selected_page == "💊 Medicine Verification":
    medicine_verification.render(medicine_system, alert_manager)
elif selected_page == "🚨 Fall Safety Monitoring":
    fall_monitoring.render(alert_manager)
elif selected_page == "🔔 Care Alert Centre":
    alerts_history.render(alert_manager)
elif selected_page == "ℹ️ System Information":
    system_information.render(medicine_system)
