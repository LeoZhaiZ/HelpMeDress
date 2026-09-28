import requests
import streamlit as st
from PIL import Image


API_BASE_URL = "http://127.0.0.1:8000"


def display_item_card(item: dict):
    """Display one result while safely handling optional metadata."""
    image_path = item.get("image_path")
    if image_path:
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Image unavailable")

    st.markdown(f"**{item.get('name') or 'Unnamed item'}**")

    category = item.get("category")
    if category:
        st.write(f"Category: {category}")

    source_category = item.get("source_category")
    if source_category:
        readable_category = str(source_category).replace("_", " ").title()
        st.write(f"Dataset category: {readable_category}")

    source_domain = item.get("source_domain")
    if source_domain:
        st.write(f"Dataset source: {str(source_domain).title()}")

    style = item.get("style")
    if style not in (None, ""):
        st.write(f"Style: {style}")

    brand = item.get("brand")
    if brand not in (None, ""):
        st.write(f"Brand: {brand}")

    price = item.get("price")
    if price is not None:
        st.write(f"Price: ${price}")

    similarity_score = item.get("similarity_score")
    if isinstance(similarity_score, (int, float)):
        st.write(f"Similarity: {similarity_score:.3f}")

    product_url = item.get("product_url")
    if (
        isinstance(product_url, str)
        and product_url.startswith(("http://", "https://"))
    ):
        st.link_button("View Product", product_url)


st.set_page_config(
    page_title="HelpMeDress",
    page_icon="👕",
    layout="wide"
)

st.title("HelpMeDress")
st.write("Upload a clothing image to find similar items and generate outfit recommendations.")

with st.sidebar:
    st.subheader("API Status")

    try:
        health_response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3
        )

        if health_response.status_code == 200:
            st.success("Backend connected")

            try:
                count_response = requests.get(
                    f"{API_BASE_URL}/items/count",
                    timeout=3
                )

                if count_response.status_code == 200:
                    count_data = count_response.json()
                    st.metric("Indexed items", count_data["count"])
                    st.caption(f"Collection: {count_data['collection']}")
                else:
                    st.warning("Indexed item count is unavailable.")
            except (requests.RequestException, ValueError, KeyError):
                st.warning("Indexed item count is unavailable.")
        else:
            st.error("Backend is unavailable.")
    except requests.RequestException:
        st.error("Backend is unavailable.")

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
    options=["", "top", "bottom", "shoes", "accessory", "outerwear", "dress"]
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
                        display_item_card(item)

    st.divider()

    st.subheader("Generate Outfit")

    anchor_category = st.selectbox(
        "What type of item did you upload?",
        options=["top", "bottom", "shoes", "accessory", "outerwear", "dress"]
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
                        display_item_card(item)
