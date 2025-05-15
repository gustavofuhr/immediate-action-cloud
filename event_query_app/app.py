import streamlit as st
from datetime import date, datetime, time
from event_query import EventQuery

event_query = EventQuery(region_name="eu-west-1", table_name="events")
st.set_page_config(page_title="Event Filter", layout="wide")
st.title("Event Record Filter")


col1, col2, col3, col4 = st.columns(4)
with col1:
    start_date = st.date_input("Start date", value=date.today())
with col2:
    end_date = st.date_input("End date", value=date.today())

# col3, col4 = st.columns([1, 2])
with col3:
    device_name = st.selectbox(
        "Device name",
        options=["axis-p3827-front-far", "axis-p3827-panoramic-tree"],
        format_func=lambda x: "Select a device" if x == "" else x
    )
with col4:
    class_filter = st.text_input(
        "Class filter (optional)",
        placeholder="Type class (e.g. person, dog)",
        value="person",
    )

if st.button("Filter Events"):
    print(start_date, end_date, device_name, class_filter)
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)
    print(start_dt, end_dt)
    print(type(start_dt), type(end_dt))

    video_urls = event_query.query_events(device_name, start_dt, end_dt, [class_filter])[:12]
    print(video_urls)
    st.markdown("### 🎥 Filtered Videos")
    rows = [video_urls[i:i + 3] for i in range(0, len(video_urls), 3)]

    for row in rows:
        cols = st.columns(3)
        for i, url in enumerate(row):
            with cols[i]:
                st.video(url)
