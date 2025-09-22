# import streamlit as st
# from datetime import datetime
# import json

# def apply_custom_css():
#     """Apply custom CSS to use Manrope font throughout the app"""
#     st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;300;400;500;600;700;800&display=swap');
    
#     /* Apply Manrope to all text elements */
#     html, body, [class*="css"] {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Specifically target Streamlit elements */
#     .stApp {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Headers */
#     h1, h2, h3, h4, h5, h6 {
#         font-family: 'Manrope', sans-serif !important;
#     }
    
#     /* Text elements */
#     p, div, span, label {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Buttons */
#     .stButton button {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Selectbox and other inputs */
#     .stSelectbox label, .stTextArea label, .stMetric label {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] button {
#         font-family: 'Manrope', sans-serif;
#     }
    
#     /* Code blocks - you might want to keep these as monospace */
#     .stCode {
#         font-family: 'Manrope', monospace;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def main():
#     # Apply custom CSS first
#     apply_custom_css()
    
#     st.title("🤖 Customer Service Analysis Agent")
#     st.markdown("---")

#     # Main content area
#     st.header("Transcript Input")

#     # Mock transcript selection
#     transcript_names = ["conversation_001.txt", "conversation_002.txt", "conversation_003.txt"]
#     selected_file = st.selectbox("Select a conversation:", transcript_names)

#     if selected_file:
#         st.subheader(f"Selected Conversation: {selected_file}")

#         # Mock transcript content
#         mock_content = """Customer: Hi, I'm having trouble with my payment processing.
# Agent: I understand you're experiencing issues with payment processing. Can you tell me more about the specific problem?
# Customer: Yes, my transactions keep failing and I'm getting error messages.
# Agent: I apologize for the inconvenience. Let me look into this for you..."""

#         st.text_area(
#             "Transcript Content",
#             mock_content,
#             height=400,
#         )

#     # Analysis section
#     st.markdown("---")
#     st.header("Analysis")

#     if st.button("🚀 Start Analysis", type="primary"):
#         with st.spinner("Initializing analysis..."):
#             # Mock progress
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             import time
            
#             for i in range(100):
#                 progress_bar.progress(i + 1)
#                 if i < 20:
#                     status_text.text("Starting analysis...🛫")
#                 elif i < 40:
#                     status_text.text("Extracting customer information...ℹ️")
#                 elif i < 60:
#                     status_text.text("Performing root cause analysis...🔍")
#                 elif i < 80:
#                     status_text.text("Generating recommendations...💬")
#                 else:
#                     status_text.text("Finalizing results...🛬")
#                 time.sleep(0.02)
            
#             status_text.text("Analysis completed!")
#             st.session_state.analysis_complete = True
        
#         st.success("✅ Analysis completed successfully!")

#     # Results section
#     if st.session_state.get('analysis_complete', False):
#         st.markdown("---")
#         st.header("📊 Analysis Results")

#         # Display results in tabs
#         tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Root Cause Analysis", "Recommendations", "Raw Response"])

#         with tab1:
#             st.subheader("📑Extracted Information")
#             col1, col2 = st.columns(2)

#             with col1:
#                 st.metric("🧑Customer Name", "John Doe")
#                 st.metric("❓Main Issue", "Payment Processing Failure")

#             with col2:
#                 st.metric("❓Status", "Resolved")

#             st.subheader("📑Summary")
#             st.write("Customer experienced payment processing failures with recurring error messages. Issue was traced to outdated payment method configuration.")

#         with tab2:
#             st.subheader("🔍Root Cause Analysis")
#             st.write("**Root Cause:**")
#             st.write("Expired credit card information in customer's payment profile causing automatic transaction rejections.")

#             st.markdown("---")
#             st.write("⛓️**Patterns:**")
#             st.write("Similar payment failures have increased by 15% over the past month, primarily affecting customers with payment methods expiring in Q3.")

#             st.markdown("---")
#             st.write("🗓️**Timeline:**")
#             st.write("- Initial failure: 3 days ago\n- Customer contact: Today\n- Resolution: Within 1 hour of contact")

#             st.markdown("---")
#             st.write("**Rationale:**")
#             st.write("The payment gateway was correctly rejecting expired payment methods, but the error messaging to customers was unclear, causing confusion and delayed resolution.")

#         with tab3:
#             st.subheader("📑Recommendations")
#             st.write("""
# **Immediate Actions:**
# 1. Implement proactive payment method expiration notifications (30, 15, and 7 days before expiry)
# 2. Improve error messaging to clearly indicate when payment methods need updating
# 3. Add direct payment method update links in error messages

# **Long-term Improvements:**
# 1. Develop automated payment method verification system
# 2. Create customer portal for easy payment method management
# 3. Implement payment method backup/failover system

# **Process Enhancements:**
# 1. Train support agents on payment troubleshooting workflow
# 2. Create self-service payment update guides
# 3. Monitor payment failure patterns for early intervention
# """)

#         with tab4:
#             st.subheader("Raw Agent Response")
#             mock_response = """Agent Analysis Complete:

# 1. Content Analysis: Extracted customer name (John Doe), issue (Payment Processing Failure), status (Resolved)
# 2. Context Check: Issue resolved, performing RCA for learning purposes
# 3. Root Cause Analysis: Identified expired payment method as primary cause
# 4. Recommendation Generation: Created actionable recommendations for prevention

# Analysis completed successfully with full context and recommendations."""

#             st.code(mock_response, language="text")

#             st.subheader("Full Context Data")
#             mock_context = {
#                 "name": "John Doe",
#                 "issue": "Payment Processing Failure",
#                 "status": "Resolved",
#                 "summary": "Customer experienced payment processing failures with recurring error messages.",
#                 "rootcause": "Expired credit card information causing transaction rejections",
#                 "recommendations": "Implement proactive notifications and improve error messaging"
#             }
#             st.json(mock_context)

#     # Export functionality
#     st.markdown("---")
#     st.subheader("Export Results")

#     if st.button("📥 Export Analysis Results"):
#         export_data = {
#             "timestamp": datetime.now().isoformat(),
#             "extracted_data": {
#                 "name": "John Doe",
#                 "issue": "Payment Processing Failure",
#                 "status": "Resolved"
#             },
#             "raw_response": "Mock analysis response data",
#         }
        
#         json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
#         st.download_button(
#             label="Download JSON Report",
#             data=json_str,
#             file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#             mime="application/json"
#         )

# if __name__ == "__main__":
#     main()


import streamlit as st
from datetime import datetime
import json

def apply_custom_css():
    """Apply custom CSS to use Manrope font throughout the app"""
    # st.markdown("""
    # <style>
    # @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;300;400;500;600;700;800&display=swap');
    
    # /* Apply Manrope to all text elements */
    # html, body, [class*="css"] {
    #     font-family: 'Manrope', sans-serif;
    # }
    
    # /* Full-page blue background */
    # [data-testid="stAppViewContainer"] {
    #     background-color: #1E3A8A;
    #     min-height: 100vh;
    # }

    # /* Center the main content area */
    # [data-testid="stMain"] {
    #     display: flex;
    #     justify-content: center;
    #     padding-top: 2rem;
    #     padding-bottom: 2rem;
    # }

    # /* Style the Streamlit content wrapper as a white card */
    # [data-testid="stMain"] > div:nth-child(1) {
    #     background-color: #ffffff;
    #     padding: 2rem;
    #     border-radius: 12px;
    #     box-shadow: 0 8px 30px rgba(2,6,23,0.18);
    #     max-width: 900px;
    #     width: 100%;
    # }

    # /* Optional: keep sidebar visually separate (if used) */
    # [data-testid="stSidebar"] {
    #     background-color: transparent;
    # }
    
    # /* Headers */
    # h1, h2, h3, h4, h5, h6 {
    #     font-family: 'Manrope', sans-serif !important;
    # }
    
    # /* Text elements */
    # p, div, span, label {
    #     font-family: 'Manrope', sans-serif;
    # }
    
    # /* Buttons */
    # .stButton button {
    #     font-family: 'Manrope', sans-serif;
    # }
    
    # /* Selectbox and other inputs */
    # .stSelectbox label, .stTextArea label, .stMetric label {
    #     font-family: 'Manrope', sans-serif;
    # }
    
    # /* Tabs */
    # .stTabs [data-baseweb="tab-list"] button {
    #     font-family: 'Manrope', sans-serif;
    # }
    
    # /* Code blocks */
    # .stCode {
    #     font-family: 'Manrope', monospace;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    /* Full-page blue background */
    [data-testid="stAppViewContainer"] {
      background-color: #1E3A8A;
      min-height: 100vh;
    }

    /* Center the main content area */
    [data-testid="stMain"] {
      display: flex;
      justify-content: center;
      padding-top: 2rem;
      padding-bottom: 2rem;
    }

    /* White scrollable container */
    [data-testid="stMain"] > div {
      background-color: #ffffff;
      padding: 2rem;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.18);
      max-width: 900px;
      width: 100%;
      overflow-y: auto;    /* enable scrolling */
      max-height: 90vh;    /* prevent overshooting */
    }

    /* Optional: sidebar transparent */
    [data-testid="stSidebar"] {
      background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Apply custom CSS first
    apply_custom_css()
    
    st.title("🤖 Customer Service Analysis Agent")
    st.markdown("---")

    # Main content area
    st.header("Transcript Input")

    # Mock transcript selection
    transcript_names = ["conversation_001.txt", "conversation_002.txt", "conversation_003.txt"]
    selected_file = st.selectbox("Select a conversation:", transcript_names)

    if selected_file:
        st.subheader(f"Selected Conversation: {selected_file}")

        # Mock transcript content
        mock_content = """Customer: Hi, I'm having trouble with my payment processing.
Agent: I understand you're experiencing issues with payment processing. Can you tell me more about the specific problem?
Customer: Yes, my transactions keep failing and I'm getting error messages.
Agent: I apologize for the inconvenience. Let me look into this for you..."""

        st.text_area(
            "Transcript Content",
            mock_content,
            height=400,
        )

    # Analysis section
    st.markdown("---")
    st.header("Analysis")

    if st.button("🚀 Start Analysis", type="primary"):
        with st.spinner("Initializing analysis..."):
            # Mock progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            import time
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Starting analysis...🛫")
                elif i < 40:
                    status_text.text("Extracting customer information...ℹ️")
                elif i < 60:
                    status_text.text("Performing root cause analysis...🔍")
                elif i < 80:
                    status_text.text("Generating recommendations...💬")
                else:
                    status_text.text("Finalizing results...🛬")
                time.sleep(0.02)
            
            status_text.text("Analysis completed!")
            st.session_state.analysis_complete = True
        
        st.success("✅ Analysis completed successfully!")

    # Results section
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.header("📊 Analysis Results")

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Root Cause Analysis", "Recommendations", "Raw Response"])

        with tab1:
            st.subheader("📑Extracted Information")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🧑Customer Name", "John Doe")
                st.metric("❓Main Issue", "Payment Processing Failure")

            with col2:
                st.metric("❓Status", "Resolved")

            st.subheader("📑Summary")
            st.write("Customer experienced payment processing failures with recurring error messages. Issue was traced to outdated payment method configuration.")

        with tab2:
            st.subheader("🔍Root Cause Analysis")
            st.write("**Root Cause:**")
            st.write("Expired credit card information in customer's payment profile causing automatic transaction rejections.")

            st.markdown("---")
            st.write("⛓️**Patterns:**")
            st.write("Similar payment failures have increased by 15% over the past month, primarily affecting customers with payment methods expiring in Q3.")

            st.markdown("---")
            st.write("🗓️**Timeline:**")
            st.write("- Initial failure: 3 days ago\n- Customer contact: Today\n- Resolution: Within 1 hour of contact")

            st.markdown("---")
            st.write("**Rationale:**")
            st.write("The payment gateway was correctly rejecting expired payment methods, but the error messaging to customers was unclear, causing confusion and delayed resolution.")

        with tab3:
            st.subheader("📑Recommendations")
            st.write("""
**Immediate Actions:**
1. Implement proactive payment method expiration notifications (30, 15, and 7 days before expiry)
2. Improve error messaging to clearly indicate when payment methods need updating
3. Add direct payment method update links in error messages

**Long-term Improvements:**
1. Develop automated payment method verification system
2. Create customer portal for easy payment method management
3. Implement payment method backup/failover system

**Process Enhancements:**
1. Train support agents on payment troubleshooting workflow
2. Create self-service payment update guides
3. Monitor payment failure patterns for early intervention
""")

        with tab4:
            st.subheader("Raw Agent Response")
            mock_response = """Agent Analysis Complete:

1. Content Analysis: Extracted customer name (John Doe), issue (Payment Processing Failure), status (Resolved)
2. Context Check: Issue resolved, performing RCA for learning purposes
3. Root Cause Analysis: Identified expired payment method as primary cause
4. Recommendation Generation: Created actionable recommendations for prevention

Analysis completed successfully with full context and recommendations."""

            st.code(mock_response, language="text")

            st.subheader("Full Context Data")
            mock_context = {
                "name": "John Doe",
                "issue": "Payment Processing Failure",
                "status": "Resolved",
                "summary": "Customer experienced payment processing failures with recurring error messages.",
                "rootcause": "Expired credit card information causing transaction rejections",
                "recommendations": "Implement proactive notifications and improve error messaging"
            }
            st.json(mock_context)

    # Export functionality
    st.markdown("---")
    st.subheader("Export Results")

    if st.button("📥 Export Analysis Results"):
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "extracted_data": {
                "name": "John Doe",
                "issue": "Payment Processing Failure",
                "status": "Resolved"
            },
            "raw_response": "Mock analysis response data",
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()