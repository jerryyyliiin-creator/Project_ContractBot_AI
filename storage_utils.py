import streamlit as st
import boto3
from botocore.exceptions import ClientError
import io

def init_s3_client():
    """Initialize the S3 client using credentials stored in Streamlit Secrets."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name=st.secrets["aws"]["s3_region_name"]
        )
        return s3_client
    except Exception as e:
        st.error(f"Unable to connect to AWS S3. Please check the credential configuration. Error message: {e}")
        return None

def upload_report_to_storage(report_content, filename="contract_analysis_report.md"):
    """Upload the report content to the specified S3 bucket."""
    s3_client = init_s3_client()
    if not s3_client:
        return None

    try:
        bucket_name = st.secrets["aws"]["s3_bucket_name"]
    except KeyError:
        st.error("S3 bucket name not found. Please set aws.s3_bucket_name in secrets.toml.")
        return None

    # Convert the Markdown report content into a byte stream.
    report_bytes_io = io.BytesIO(report_content.encode('utf-8'))

    try:
        with st.spinner(f"Uploading '{filename}' to Amazon S3..."):
            s3_client.upload_fileobj(
                report_bytes_io,
                bucket_name,
                filename,
                ExtraArgs={'ContentType': 'text/markdown'}
            )
        
        st.success(f"Report '{filename}' has been successfully stored to Amazon S3！")
        return True
    except ClientError as e:
        st.error(f"Error occurred while uploading the file to S3: {e}")
        return None