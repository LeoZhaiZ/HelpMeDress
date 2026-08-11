import requests
import streamlit as st
from PIL import Image


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="HelpMeDress",
    page_icon="👕",
    layout="wide"
)

st.title("HelpMeDress")
st.write("Upload a clothing image to find similar items and generate outfit recommendations.")

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

top_k = st.slider(
    "Number of similar results",
    min_value=1,
    max_value=10,
    value=5
)

category_filter = st.selectbox(
    "Optional category filter",
    options=["", "top", "bottom", "shoes", "accessory", "outerwear"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    st.divider()

    if st.button("Find Similar Items"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        data = {
            "top_k": top_k
        }

        if category_filter:
            data["category"] = category_filter

        response = requests.post(
            f"{API_BASE_URL}/search/similar",
            files=files,
            data=data
        )

        if response.status_code != 200:
            st.error(response.text)
        else:
            results = response.json()["results"]

            st.subheader("Similar Items")

            if not results:
                st.warning("No results found.")
            else:
                cols = st.columns(3)

                for index, item in enumerate(results):
                    with cols[index % 3]:
                        st.image(item["image_path"], use_container_width=True)
                        st.markdown(f"**{item['name']}**")
                        st.write(f"Category: {item['category']}")
                        st.write(f"Style: {item['style']}")
                        st.write(f"Brand: {item['brand']}")
                        st.write(f"Price: ${item['price']}")
                        st.write(f"Similarity: {item['similarity_score']:.3f}")
                        st.link_button("View Product", item["product_url"])

    st.divider()

    st.subheader("Generate Outfit")

    anchor_category = st.selectbox(
        "What type of item did you upload?",
        options=["top", "bottom", "shoes", "accessory", "outerwear"]
    )

    if st.button("Generate Outfit"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        data = {
            "anchor_category": anchor_category
        }

        response = requests.post(
            f"{API_BASE_URL}/outfit/generate",
            files=files,
            data=data
        )

        if response.status_code != 200:
            st.error(response.text)
        else:
            outfit = response.json()

            st.subheader("Outfit Recommendations")

            for slot, items in outfit["items"].items():
                st.markdown(f"### {slot.title()}")

                if not items:
                    st.warning(f"No items found for {slot}.")
                    continue

                cols = st.columns(3)

                for index, item in enumerate(items):
                    with cols[index % 3]:
                        st.image(item["image_path"], use_container_width=True)
                        st.markdown(f"**{item['name']}**")
                        st.write(f"Category: {item['category']}")
                        st.write(f"Style: {item['style']}")
                        st.write(f"Price: ${item['price']}")
                        st.write(f"Similarity: {item['similarity_score']:.3f}")
                        st.link_button("View Product", item["product_url"])