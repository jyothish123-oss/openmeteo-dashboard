import base64
import pandas as pd

def format_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Formats DataFrame for neat display in Streamlit."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if "time" in col:
            df_copy[col] = pd.to_datetime(df_copy[col]).dt.strftime("%Y-%m-%d")
    return df_copy

def download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
