import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
import json
import re
import requests
import pandas as pd
from io import BytesIO

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()
os.environ["GROQ_API_KEY"] = "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"


# Initialize the LLM
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Initialize the parser
parser = StrOutputParser()

# Define the prompt
prompt = PromptTemplate(
    input_variables=["user_query"],
    template="""
You are a helpful assistant that converts user queries into structured API parameters for FDA device APIs. Based on the user query, determine which API to use:
- Classification API: https://api.fda.gov/device/classification.json
- 510k API: https://api.fda.gov/device/510k.json
only use class clasiification when user used word classfication/class ok otherwise use 510k.json url only
### Classification API Schema (valid fields, example values):
- third_party_flag: "N" or "Y"
- life_sustain_support_flag: "N" or "Y"
- gmp_exempt_flag: "N" or "Y"
- summary_malfunction_reporting: e.g. "Ineligible" or "Eligible"
- product_code: e.g. "NFK" (always make in to upercase everychar if user write lower)
- review_panel: e.g. "GU" (always make in to upercase everychar if user write lower)
- medical_specialty: e.g. "GU" (always make in to upercase everychar if user write lower)
- device_name: e.g. "Kit, Repair, Catheter, Hemodialysis" (make user input first  letter Capital)
- review_code: e.g. ""
- unclassified_reason: e.g. ""
- medical_specialty_description: e.g. "Gastroenterology, Urology" (make user input first  letter Capital)
- device_class: e.g. "2"
- definition: e.g. "Hemodialysis Tray"
- regulation_number: e.g. "876.5540"
- implant_flag: "N" or "Y"
- submission_type_id: e.g. "1" 

### 510k API Schema (valid fields, example values):
- third_party_flag: "N" or "Y"
- city: e.g. "LINTHICUM" (always make in to upercase everychar if user write lower)
- advisory_committee_description: e.g. "Cardiovascular" (make user input first  letter Capital)
- address_1: e.g. "611 NORTH HAMMONDS FERRY ROAD" (always make in to upercase everychar if user write lower)
- address_2: e.g. ""
- product_code: e.g. "DRX" (always make in to upercase everychar if user write lower)
- zip_code: e.g. "21090-1356"
- applicant: e.g. "AMBU, INC." (always make in to upercase everychar if user write lower)
- decision_date:(also approved date) e.g. [YYYY-MM-DD] take in this format or try to understnad user date 
- decision_code: e.g. "SESE" (always make in to upercase everychar if user write lower)
- country_code: e.g. "US" (always make in to upercase everychar if user write lower)
- device_name: e.g. "AMBU BLUE SENSOR, MRX, ECG ELECTRODE PRODUCT #:MRX-00-S"
- advisory_committee: e.g. "CV" (always make in to upercase everychar if user write lower)
- contact: e.g. "SANJAY PARIKH" (always make in to upercase everychar if user write lower)
- expedited_review_flag: e.g. ""
- k_number: e.g. "K041026"
- state: e.g. "MD" (always make in to upercase everychar if user write lower)
- date_received: e.g. [YYYY-MM-DD] take in this format or try to understnad user date
- review_advisory_committee: e.g. "CV" 
- postal_code: e.g. "21090-1356"
- decision_description: e.g. "Substantially Equivalent" (make user input first  letter Capital)
- clearance_type: e.g. "Traditional" (make user input first  letter Capital)

### Rules:
- Determine the intent based on the query:
  - Use "classification" for queries about device classification, class, regulation, etc.
  - Use "510k" for queries about 510k submissions, clearances, applicants, decision dates, etc.
- Extract the field(s), term(s), and optional limit from the user query.
- Terms are user-specified values for the fields (not limited to the examples above).
- If multiple fields/terms are mentioned, join them with AND (e.g., field1:term1+AND+field2:term2).
- Default limit = 100 if not specified.
- Return JSON in the following format:

{{
  "search": "field:term[+AND+field:term...]",
  "limit": number,
  "intent": "510k" or "classification"
}}

### Example Queries and Outputs:
- Query: "show me three devices with class 2"
  Output: {{"search": "device_class:2", "limit": 3, "intent": "classification"}}
- Query: "find devices with product code DRX and decision date 2004-06-28"
  Output: {{"search": "product_code:DRX+AND+decision_date:2004-06-28", "limit": 1, "intent": "510k"}}

Now process this query:
"{user_query}"
"""
)

# Create the chain
chain = prompt | llm | parser

# Streamlit UI
st.set_page_config(page_title="FDA Chatbot Demo", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Welcome to the FDA Chatbot Demo")
    st.subheader("We are using official FDA API to get best results for your query")
    
    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., show me three devices with class 2")
    
    # Suggested queries
    st.markdown("**Suggested Queries:**")
    suggested_queries = [
    "show me 5 devices with class 2",  # Classification, explicit limit
    "find devices with product code DRX and decision date 2004-06-28",  # 510k, default limit
    "show me devices with medical specialty GU",  # Classification, default limit
    "find 2 devices with k_number K041026",  # 510k, explicit limit
    "show me 3 devices with implant_flag Y",  # Classification, explicit limit
    "find devices with applicant AMBU, INC.",  # 510k, default limit
    "show me 4 devices with regulation_number 876.5540",  # Classification, explicit limit
    "find devices with decision_code SESE and state MD",  # 510k, default limit
    "show me 2 devices with medical_specialty_description Cardiology",  # Classification, explicit limit
    "find 3 devices with clearance_type Traditional"  # 510k, explicit limit
    ]
    for query in suggested_queries:
        if st.button(query):
            user_query = query
            st.session_state.user_query = user_query  # Store in session state

# Main panel
st.header("FDA API Results")

# Process query if provided
if user_query or "user_query" in st.session_state:
    query = user_query or st.session_state.user_query
    
    # Invoke the chain
    result = chain.invoke({"user_query": query})
    
    # Extract JSON block
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        st.error("❌ No JSON object found in LLM output")
    else:
        json_str = match.group(0)
        
        # Parse JSON
        try:
            parsed = json.loads(json_str)
            search = parsed.get("search", "")
            limit = parsed.get("limit", 1)
            intent = parsed.get("intent", "")
            
            # Build FDA API URL
            base_url = "https://api.fda.gov/device/"
            endpoint = "classification.json" if intent == "classification" else "510k.json"
            url = f"{base_url}{endpoint}?search={search}&limit={limit}"
            # Display parsed JSON output
            st.subheader("Query Parameters url")
            st.json(parsed)
            st.write(url)
            # Call FDA API
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    st.success(f"✅ Found {len(results)} record(s)")
                    
                    # Create DataFrame for results
                    df = pd.DataFrame(results)
                    if "openfda" in df.columns:
                        df = df.drop(columns=["openfda"])  # Drop openfda column
                    # Convert DataFrame to Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="FDA_Results")
                    excel_data = output.getvalue()
                    
                    # Download button
                    st.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name="fda_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    # Display results
                    for i, record in enumerate(results, 1):
                        st.subheader(f"Record {i}")
                        for key, value in record.items():
                            if key == "openfda":
                                continue
                            st.write(f"**{key}**: {value}")
                    
                    # # Convert DataFrame to Excel
                    # output = BytesIO()
                    # with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    #     df.to_excel(writer, index=False, sheet_name="FDA_Results")
                    # excel_data = output.getvalue()
                    
                    # # Download button
                    # st.download_button(
                    #     label="Download Results as Excel",
                    #     data=excel_data,
                    #     file_name="fda_results.xlsx",
                    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    # )
                else:
                    st.warning("⚠️ No results found in API response")
            else:
                st.error(f"❌ API Error: Status Code {response.status_code} - {response.text}")
        except json.JSONDecodeError:
            st.error("❌ Failed to parse JSON from LLM output")
else:

    st.info("Please enter a query or select a suggested query to see results.")
