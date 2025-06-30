import fiftyone as fo

if __name__ == "__main__":
    session = fo.launch_app(address="0.0.0.0", auto=True, port=5152)
    session.wait(-1)