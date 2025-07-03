from datetime import datetime, timedelta, time, timezone
import streamlit as st

def get_time_range(device_ids):
    now = datetime.now(timezone.utc)
    presets = [
        "Last 30 minutes",
        "Last hour",
        "Past 24 hours",
        "Past 7 days",
        "Custom range"
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected = st.radio("Time Range", presets, index=2, horizontal=True)
    with col2:
        selected_devices = st.multiselect(
            "Devices", device_ids + ["any"], "any",
            format_func=lambda x: x.split("__")[1] if "__" in x else x
        )

    if selected == "Custom range":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            start_date = st.date_input("Start date", now.date())
        with col2:
            start_time = st.time_input("Start time", time.min)
        with col3:
            end_date = st.date_input("End date", now.date())
        with col4:
            end_time = st.time_input("End time", now.time())
        start_dt = datetime.combine(start_date, start_time)
        end_dt = datetime.combine(end_date, end_time)
        if start_dt >= end_dt:
            st.error("Start time must be before end time.")
    elif selected == "Last 15 minutes":
        start_dt = now - timedelta(minutes=15)
        end_dt = now
    elif selected == "Last 30 minutes":
        start_dt = now - timedelta(minutes=30)
        end_dt = now
    elif selected == "Last hour":
        start_dt = now - timedelta(hours=1)
        end_dt = now
    elif selected == "Today":
        start_dt = datetime(now.year, now.month, now.day)
        end_dt = now
    elif selected == "Past 24 hours":
        start_dt = now - timedelta(hours=24)
        end_dt = now
    elif selected == "Yesterday":
        yesterday = now.date() - timedelta(days=1)
        start_dt = datetime.combine(yesterday, datetime.min.time())
        end_dt = datetime.combine(yesterday, datetime.max.time())
    elif selected == "Past 7 days":
        start_dt = now - timedelta(days=7)
        end_dt = now

    # st.markdown(f"🔎 Querying events from **{start_dt.strftime('%Y-%m-%d %H:%M:%S')}** to **{end_dt.strftime('%Y-%m-%d %H:%M:%S')}**")

    return start_dt, end_dt, selected_devices