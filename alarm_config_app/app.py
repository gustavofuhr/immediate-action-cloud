import json
import streamlit as st
from alarm_config import AlarmConfigController, get_all_stream_ids

st.set_page_config(page_title="Alarm Config", layout="wide")
st.title("Alarm config")

alarm_config_controller = AlarmConfigController()
device_ids = get_all_stream_ids()

selected_stream_id = st.selectbox(
    "Stream ID",
    device_ids,
    format_func=lambda x: x
)

if "config_json" not in st.session_state:
    st.session_state.config_json = {}

if selected_stream_id not in st.session_state.config_json:
    alarm_config = alarm_config_controller.get_stream_alarm_config(selected_stream_id)
    if "config" in alarm_config:
        st.session_state.config_json[selected_stream_id] = json.dumps(alarm_config["config"], indent=2)
    else:
        st.session_state.config_json[selected_stream_id] = "{}"
        st.caption("No existing config found for this stream. Starting with empty config.")

if "success_message" in st.session_state:
    st.success(st.session_state.success_message)
    del st.session_state.success_message  # show once

st.subheader("JSON config")
config_json = st.session_state.config_json[selected_stream_id]
st.code(config_json, language="json")

# --- Parse JSON for editing ---
try:
    config_dict = json.loads(config_json)
except json.JSONDecodeError:
    st.error("⚠️ Invalid JSON format — cannot parse.")
    st.stop()

# --- EDIT CHANNELS (COLLAPSIBLE) ---
with st.expander("Edit Channels", expanded=False):
    channels = config_dict.get("channels", {})

    # WhatsApp
    whatsapp_numbers = channels.get("whatsapp", {}).get("numbers", [])
    whatsapp_input = st.text_area(
        "WhatsApp Numbers (comma-separated)",
        value=", ".join(whatsapp_numbers),
        placeholder="+5511999999999, +5511888888888"
    )

    # Email
    email_recipients = channels.get("email", {}).get("recipients", [])
    email_input = st.text_area(
        "Email Recipients (comma-separated)",
        value=", ".join(email_recipients),
        placeholder="user1@example.com, user2@example.com"
    )

    if st.button("Apply channel changes"):
        whatsapp_list = [x.strip() for x in whatsapp_input.split(",") if x.strip()]
        email_list = [x.strip() for x in email_input.split(",") if x.strip()]

        config_dict["channels"] = {}
        if whatsapp_list:
            config_dict["channels"]["whatsapp"] = {"numbers": whatsapp_list}
        if email_list:
            config_dict["channels"]["email"] = {"recipients": email_list}

        try:
            alarm_config_controller.put_stream_alarm_config(selected_stream_id, config_dict)
            st.session_state.config_json[selected_stream_id] = json.dumps(config_dict, indent=2)
            st.session_state.success_message = "✅ Channels updated and saved to DynamoDB"
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save config: {e}")

# --- EDIT PLATE RULES (COLLAPSIBLE) ---
with st.expander("Edit Plate Rules", expanded=False):
    rules = config_dict.setdefault("rules", {})
    plate_section = rules.setdefault("plate", {})
    targets = plate_section.setdefault("targets", {})

    default_plate_conf = 0.7
    default_ocr_conf = 0.8
    if targets:
        first_rule = next(iter(targets.values()))
        default_plate_conf = float(first_rule.get("min_plate_confidence", 0.7))
        default_ocr_conf = float(first_rule.get("min_ocr_confidence", 0.8))

    plate_text_input = st.text_area(
        "Plate numbers (comma-separated)",
        value=", ".join(targets.keys()),
        placeholder="ABC1234, XYZ9876, 162D11338",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        global_plate_conf = st.slider(
            "Min plate confidence (global)", 0.0, 1.0, value=default_plate_conf, step=0.01
        )
    with col2:
        global_ocr_conf = st.slider(
            "Min OCR confidence (global)", 0.0, 1.0, value=default_ocr_conf, step=0.01
        )

    if st.button("Apply plate rule changes"):
        plate_list = [p.strip().upper() for p in plate_text_input.split(",") if p.strip()]
        new_targets = {
            plate: {
                "min_plate_confidence": float(global_plate_conf),
                "min_ocr_confidence": float(global_ocr_conf)
            }
            for plate in plate_list
        }

        config_dict["rules"]["plate"]["targets"] = new_targets

        try:
            alarm_config_controller.put_stream_alarm_config(selected_stream_id, config_dict)
            st.session_state.config_json[selected_stream_id] = json.dumps(config_dict, indent=2)
            st.session_state.success_message = "✅ Plate rules updated and saved to DynamoDB"
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save plate rules: {e}")


# --- EDIT OBJECT RULES (COLLAPSIBLE) ---
with st.expander("Edit Object Rules", expanded=True):
    rules = config_dict.setdefault("rules", {})
    object_section = rules.setdefault("object", {})
    obj_targets = object_section.setdefault("targets", {})

    default_obj_conf = 0.75
    if obj_targets:
        first_obj_rule = next(iter(obj_targets.values()))
        default_obj_conf = float(first_obj_rule.get("min_confidence", 0.75))

    existing_classes = sorted(obj_targets.keys())

    CLASS_OPTIONS = [
        'animals', 'vehicles', 'person_with_ppe', 'person_wout_ppe',
        'dog', 'sheep', 'cow', 'bird', 'horse', 'elephant', 'bear', 'zebra', 'giraffe',
        'car', 'motorcycle', 'bus', 'bicycle', 'train', 'truck', 'boat', 'airplane',
        'person', 'car_plate', 'cat',
        'person_ppe_upper', 'person_ppe_bottom', 'person_ppe_full', 'person_ppe_noppe',
    ]

    class_filter = st.multiselect(
        "Select classes (optional):",
        options=CLASS_OPTIONS,
        default=existing_classes
    )

    obj_conf_slider = st.slider(
        "Min confidence (applies to all selected classes)",
        min_value=0.0, max_value=1.0, value=default_obj_conf, step=0.01
    )

    # --- Expansion logic for grouped classes ---
    EXPANSIONS = {
        'animals': ['dog', 'sheep', 'cow', 'bird', 'horse', 'elephant', 'bear', 'zebra', 'giraffe'],
        'vehicles': ['car', 'motorcycle', 'bus', 'bicycle', 'train', 'truck', 'boat', 'airplane'],
        'person_with_ppe': ['person_ppe_upper', 'person_ppe_bottom', 'person_ppe_full'],
        'person_wout_ppe': ['person_ppe_noppe'],
    }

    def expand_classes(selected: list[str]) -> list[str]:
        out = []
        seen = set()
        queue = list(selected)
        while queue:
            cls = queue.pop(0)
            if cls in EXPANSIONS:
                queue.extend(EXPANSIONS[cls])
                continue
            if cls not in seen:
                seen.add(cls)
                out.append(cls)
        return out

    if st.button("Apply object rule changes"):
        target_classes = expand_classes(class_filter)

        new_obj_targets = {
            cls: {"min_confidence": float(obj_conf_slider)}
            for cls in target_classes
        }
        config_dict["rules"]["object"]["targets"] = new_obj_targets

        try:
            alarm_config_controller.put_stream_alarm_config(selected_stream_id, config_dict)
            st.session_state.config_json[selected_stream_id] = json.dumps(config_dict, indent=2)
            st.session_state.success_message = "✅ Object rules updated and saved to DynamoDB"
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save object rules: {e}")
