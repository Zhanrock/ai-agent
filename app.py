#app.py
import os
import sys

# This check ensures the code only runs in a deployed environment
# where the issue occurs, not locally.
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # Local environment without pysqlite3 → fall back to built-in sqlite3
    pass
import streamlit as st
import pandas as pd
from scheduler import solve_schedule, swap_shift
from arai_rag import answer_question

st.set_page_config(layout="wide")
st.title("FAN - Franchise AI Navigator")

# Dark mode styling
st.markdown("""
<style>
.stApp {background-color: #1e1e1e;}
h1 {color: #f0f0f0; font-weight: bold;}
h2, h3, h4, h5, h6, p, span {color: #e0e0e0;}
.stTextInput>div>div>input {color: #f0f0f0; background-color: #2a2a2a;}
.stButton>button {color: #f0f0f0; background-color: #333333; border: none;}
</style>
""", unsafe_allow_html=True)

def highlight_preview(val):
    return 'background-color: lightyellow; color: black' if val == 1 else ''

def highlight_schedule(val):
    return 'background-color: lightgreen; color: black' if val == 1 else ''

tabs = st.tabs(["Arai - Onboarding", "Oai - Scheduling", "About"])

# ---- Arai Tab ----
with tabs[0]:
    st.header("Ask Arai (Onboarding Agent)")
    
    # Use a form to group input and button
    with st.form("arai_form"):
        query = st.text_input("Type your question:")
        style_choice = st.selectbox("Answer style:", ["bullet", "sentence"])
        
        # Use a submit button for the form
        submitted = st.form_submit_button("Ask Arai")

    if "arai_history" not in st.session_state:
        st.session_state['arai_history'] = []

    # Process form submission
    if submitted and query:
        with st.spinner('Thinking...'):
            response, sources = answer_question(query, style=style_choice)
        st.session_state['arai_history'].append({
            "query": query,
            "answer": response,
            "sources": sources
        })

    # Clear button action
    if st.button("Clear Answers"):
        st.session_state['arai_history'] = []
        st.rerun()  # Rerun to clear the display immediately

    # Display history from newest to oldest
    for h_idx, h in reversed(list(enumerate(st.session_state['arai_history']))):
        st.markdown("---")
        st.write(f"**Q{len(st.session_state['arai_history']) - h_idx}:** {h['query']}")
        
        # Display answer with improved formatting
        if h['answer'].startswith("•"):
            # Replace • with - for proper Markdown bullets
            md_ans = h['answer'].replace("•", "-")
            st.markdown(f"**A:**\n{md_ans}")
        else:
            st.markdown(f"**A:**\n{h['answer']}")

        # Display sources with an expander
        with st.expander(f"Show sources for '{h['query']}'"):
            for s_idx, src in enumerate(h["sources"]):
                st.markdown(f"**Source {s_idx+1}:** {src['section']}")
                st.write(f"**Preview:** {src['preview']}...")

# ---- Oai Tab ----
with tabs[1]:
    st.header("Generate Weekly Schedule")
    uploaded_file = st.file_uploader("Upload availability CSV", type="csv")
    
    if uploaded_file:
        avail_df = pd.read_csv(uploaded_file)
        st.subheader("Preview Availability")
        st.dataframe(avail_df.set_index('Employee').style.applymap(highlight_preview))

        # ปุ่ม Generate Schedule
        if st.button("Generate Schedule"):
            schedule = solve_schedule(avail_df)
            st.session_state['schedule'] = schedule.copy()
            st.session_state['original_schedule'] = schedule.copy()
            st.success("✅ Schedule generated successfully!")

    # ถ้ามี schedule อยู่แล้ว
    if 'schedule' in st.session_state:
        schedule = st.session_state['schedule']
        employees = schedule.index.tolist()
        shifts = schedule.columns.tolist()

        st.subheader("Generated Schedule")
        schedule_display = st.dataframe(schedule.style.applymap(highlight_schedule))

        # Swap shifts
        st.subheader("Swap Shifts")
        col1, col2, col3 = st.columns(3)
        with col1:
            emp1 = st.selectbox("Employee 1", employees)
        with col2:
            emp2 = st.selectbox("Employee 2", employees)
        with col3:
            shift_sel = st.selectbox("Shift to swap", shifts)

        if st.button("Swap Shift"):
            success = swap_shift(schedule, emp1, emp2, shift_sel)
            if success:
                st.session_state['schedule'] = schedule
                st.success(f"Swapped {shift_sel} between {emp1} and {emp2}")
            else:
                st.warning("Swap not allowed: both employees must have this shift assigned (1).")
            st.dataframe(schedule.style.applymap(highlight_schedule))

        # Download CSV
        csv = schedule.to_csv(index=True).encode('utf-8')
        st.download_button("Download CSV", csv, "schedule.csv", "text/csv")

        # Reset schedule
        if st.button("Reset Schedule"):
            st.session_state['schedule'] = st.session_state['original_schedule'].copy()
            st.success("🔄 Schedule has been reset to the original version.")
            schedule_display.dataframe(st.session_state['schedule'].style.applymap(highlight_schedule))
# ---- About Tab ----
with tabs[2]:
    st.write("FAN - Franchise AI Navigator")
    st.write("Developed by Gimmie and Zharock")

