import streamlit as st
from datetime import date, datetime, time
from event_query import EventQuery
import random

event_query = EventQuery(region_name="eu-west-1", table_name="events")
st.set_page_config(page_title="Event Filter", layout="wide")
st.title("🎥 Event Record Filter")

col1, col2, col3, col4 = st.columns(4)
with col1:
    start_date = st.date_input("Start date", value=date.today())
with col2:
    end_date = st.date_input("End date", value=date.today())
with col3:
    class_filter = st.multiselect(
        "Select classes (optional):",
        options=['person', 'car', 'motorcycle', 'bus', 'train', 'truck', 'bird', 'dog'],
        default=["person"]
    )
with col4:
    logic_operator = st.radio(
        "Match condition", options=["OR", "AND"], index=0,
        disabled=len(class_filter) <= 1, horizontal=True
    )

device_ids = event_query.get_all_device_ids()
col_device, col_threshold = st.columns([2, 1])
with col_device:
    selected_devices = st.multiselect(
        "Devices", device_ids, device_ids[:1],
        format_func=lambda x: x.split("__")[1] if "__" in x else x
    )
with col_threshold:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)

start_dt = datetime.combine(start_date, time.min)
end_dt = datetime.combine(end_date, time.max)

if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = []
if "random_batch" not in st.session_state:
    st.session_state.random_batch = []

if st.button("🔍 Filter Events"):
    if not selected_devices or not class_filter:
        st.warning("Please select at least one device and one class.")
    else:
        results = event_query.query_events(
            start_date=start_dt,
            end_date=end_dt,
            target_classes=class_filter,
            threshold=threshold,
            condition=logic_operator,
            device_ids=selected_devices
        )
        st.session_state.filtered_results = results
        st.session_state.random_batch = random.sample(results, min(12, len(results)))

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
                if item["video_url"]:
                    st.video(item["video_url"])
                st.caption(f"📅 {item['timestamp']}")
                if "detection_stats" in item:
                    class_scores = [
                        f"{cls} ({item['detection_stats'][cls]['max_confidence']:.2f})"
                        for cls in item.get("seen_classes", [])
                        if cls in item["detection_stats"]
                    ]
                    st.caption(f"🏷️ {', '.join(class_scores)}")
                else:
                    st.caption(f"🏷️ {', '.join(item.get('seen_classes', []))}")