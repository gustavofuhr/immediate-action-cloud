import streamlit as st
from datetime import date
from event_query import COCO_CLASSES, EventQuery

event_query = EventQuery(region_name="eu-west-1", table_name="event_ai")
st.set_page_config(page_title="Event Filter", layout="wide")
st.title("Event Record Filter")


col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=date.today())
with col2:
    end_date = st.date_input("End date", value=date.today())

col3, col4 = st.columns([1, 2])
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
        help="Suggestions: " + ", ".join(list(COCO_CLASSES.values())[:10]) + ", ..."
    )

if st.button("Filter Events"):
    video_urls = event_query.get_filtered_videos(start_date, end_date, class_filter, device_name)
    
    st.markdown("### 🎥 Filtered Videos")
    rows = [video_urls[i:i + 3] for i in range(0, len(video_urls), 3)]

    for row in rows:
        cols = st.columns(3)
        for i, url in enumerate(row):
            with cols[i]:
                st.video(url)
