import streamlit as st
import random

import pandas as pd
from app_time_filters import get_time_range
from event_query import EventQuery, get_event_stats

event_query = EventQuery(region_name="eu-west-1", table_name="events")
st.set_page_config(page_title="Event Filter", layout="wide")
st.title("Event Record Filter")

# CLASSES_TO_STORE = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#                     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']


device_ids = event_query.get_all_device_ids()

start_dt, end_dt, selected_devices = get_time_range(device_ids)

tab1, tab2, tab3 = st.tabs(["Filter by object", "License plate", "Specific event"])
with tab1:
    col_obj1, col_obj2, col_obj3 = st.columns([2,1,1])
    with col_obj1:
        class_filter = st.multiselect(
            "Select classes (optional):",
            options=['person', 'car_plate', 'person_w_ppe', 'person_wout_ppe', 'animals', 'dog', 'sheep', 'cow', 'bird', 'vehicles', 'car', 'bicycle', 'motorcycle', 'train', 'truck'], 
            default=["person"]
        )
    with col_obj2:
        force_or = "animals" in class_filter or "vehicles" in class_filter
        if force_or:
            st.radio(
                "Match condition", options=["OR"], index=0,
                disabled=True, horizontal=True, key="forced_or"
            )
            logic_operator = "OR"
        else:
            logic_operator = st.radio(
                "Match condition", options=["OR", "AND"], index=0,
                disabled=len(class_filter) <= 1, horizontal=True, key="user_logic"
            )
    with col_obj3:
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="threshold_object")

    
    if st.button("🔍 Filter Events"):
        if not selected_devices:
            st.warning("Please select at least one device (or 'any').")
        else:
            if "any" in selected_devices:
                actual_devices = device_ids
            else:
                actual_devices = selected_devices
            st.session_state.actual_selected_devices = actual_devices  # store selecte devices in session state

            print(f"Querying events from {start_dt} to {end_dt} for devices: {actual_devices}, classes: {class_filter}, threshold: {threshold}, condition: {logic_operator}")
            results = event_query.query_events(
                start_date=start_dt,
                end_date=end_dt,
                target_classes=class_filter,
                threshold=threshold,
                condition=logic_operator,
                device_ids=actual_devices
            )
            st.session_state.filtered_results = results
            st.session_state.random_batch = random.sample(results, min(12, len(results)))
with tab2:
    col_lr1, col_lr2, col_lr3 = st.columns([2,1,1])
    with col_lr1:
        search_plate = st.text_input("Search plate (ex: 'ABC1234')", "")
    with col_lr2:
        plate_threshold = st.slider("Plate threshold", 0.0, 1.0, 0.5, 0.05, key="plate_threshold")
    with col_lr3:
        ocr_threshold = st.slider("OCR threshold", 0.0, 1.0, 0.5, 0.05, key="ocr_threshold", disabled=True)

    if st.button("🔍 Search plate"):
        if not selected_devices or not search_plate:
            st.warning("Please select at least one device (or 'any') and enter a plate to search for.")
        else:
            if "any" in selected_devices:
                actual_devices = device_ids
            else:
                actual_devices = selected_devices
            st.session_state.actual_selected_devices = actual_devices
            print(f"Querying events from {start_dt} to {end_dt} for devices: {selected_devices}, plate: {search_plate}, plate threshold: {plate_threshold}, OCR threshold: {ocr_threshold}")
            results = event_query.query_events_by_plate(
                start_date=start_dt,
                end_date=end_dt,
                search_plate=search_plate,
                plate_threshold=plate_threshold,
                ocr_threshold=ocr_threshold,
                device_ids=actual_devices
            )
            st.session_state.filtered_results = results
            st.session_state.random_batch = random.sample(results, min(12, len(results)))
with tab3:
    col_se1, col_se2 = st.columns(2)
    with col_se1:
        selected_devices = st.selectbox("Device ID", device_ids,
            format_func=lambda x: x#.split("__")[1] if "__" in x else x
        )
    with col_se2:
        search_timestamp = st.text_input("Search timestamp (ex: '2025-07-02T17:21:14.861490+00:00')", "")
    if st.button("🔍 Get event"):
        if not search_timestamp:
            st.warning("Please enter a timestamp to search for.")
        else:
            results = event_query.get_event_by_id_and_timestamp(
                device_id=selected_devices,
                event_timestamp=search_timestamp
            )
            st.session_state.filtered_results = [results]
            st.session_state.random_batch = random.sample([results], min(12, len([results])))

    

if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = []
if "random_batch" not in st.session_state:
    st.session_state.random_batch = []


if st.session_state.filtered_results:
    st.markdown(f"### Showing {len(st.session_state.random_batch)} of {len(st.session_state.filtered_results)} Results")

    if st.button("🔀 Randomize"):
        st.session_state.random_batch = random.sample(
            st.session_state.filtered_results,
            min(12, len(st.session_state.filtered_results))
        )

    rows = [st.session_state.random_batch[i:i + 3] for i in range(0, len(st.session_state.random_batch), 3)]
    for row in rows:
        cols = st.columns(3)
        for i, item in enumerate(row):
            with cols[i]:
                if "video_url" not in item:
                    # probably the video is not available yet.
                    continue
                st.video(item["video_url"])
                st.caption(f"📹 {item.get('device_id', 'Unknown')} • 📅 {item['event_timestamp']}")
                if "detection_stats" in item:
                    class_scores = [
                        f"{cls} ({item['detection_stats'][cls]['max_confidence']:.2f})"
                        for cls in item.get("seen_classes", [])
                        if cls in item["detection_stats"]
                    ]
                    st.caption(f"🏷️ {', '.join(class_scores)}")
                else:
                    st.caption(f"🏷️ {', '.join(item.get('seen_classes', []))}")

    selected_for_stats = st.session_state.get("actual_selected_devices", [])
    if st.session_state.filtered_results and selected_for_stats:
        stats = get_event_stats(
            st.session_state.filtered_results,
            device_ids
        )

        st.markdown("---")
        st.subheader("📊 Event Stats per Device")
        col_stats, col_missing = st.columns([2, 1])
        with col_stats:
            stats_with_events = [s for s in stats if s["Events"] > 0]
            if stats_with_events:
                df = pd.DataFrame(stats_with_events).drop(columns=["Color"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No events found for selected devices.")
        with col_missing:
            missing = [s["Device"] for s in stats if s["Events"] == 0 and s["Device"] in selected_for_stats]
            if missing:
                st.markdown("#### ❌ Devices with no events")
                for d in missing:
                    st.markdown(f"- :orange[{d}]")
            else:
                st.markdown("All selected devices have events. 🎉")

