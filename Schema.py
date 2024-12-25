from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# import AppLog
import json
from utils.logger import AppLog 
# Configure AppLog
# AppLog.basicConfig(level=AppLog.INFO)

def create_keyfile_dict():
    variables_keys = {
        "type": os.getenv("TYPE"),
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        "private_key": os.getenv("PRIVATE_KEY"),
        "client_email": os.getenv("CLIENT_EMAIL"),
        "client_id": os.getenv("CLIENT_ID"),
        "auth_uri": os.getenv("AUTH_URI"),
        "token_uri": os.getenv("TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL")
    }
    return variables_keys
async def create_table_with_schema(client, full_table_id, schema):
    """Creates a BigQuery table with the specified schema."""
    try:
        # Check if the table already exists
        table = client.get_table(full_table_id)
        AppLog.info(f"Table {full_table_id} already exists.")
    except NotFound:
        # If the table does not exist, create it
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        AppLog.info(f"Table {full_table_id} created successfully with schema.")
    except Exception as e:
        AppLog.info(f"An error occurred: {e}")
        raise

async def create_schema():
    # Get environment variables
    project_id = os.getenv("PROJECT_ID")  # Ensure this is set
    dataset_id = os.getenv("DATASET_ID")  # Ensure this is set
    table_id = os.getenv("TABLE_ID")      # Ensure this is set
    location = "US"

    # AppLog.info values for debugging
    AppLog.info(f"Project ID: {project_id}")
    AppLog.info(f"Dataset ID: {dataset_id}")
    AppLog.info(f"Table ID: {table_id}")

    # Check for missing environment variables
    if not project_id or not dataset_id or not table_id:
        raise ValueError("Ensure PROJECT_ID, DATASET_ID, and TABLE_ID are set in environment variables.")

    # Combine to form the full table reference
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    AppLog.info(f"Full Table ID: {full_table_id}")


    # Load nội dung JSON của tệp service account từ biến môi trường
    service_account_json = create_keyfile_dict()

    # Chuyển JSON sang chuỗi và tạo client BigQuery từ chuỗi JSON
    try:
        client = bigquery.Client.from_service_account_info(service_account_json)
        AppLog.info("BigQuery client created successfully.")
    except Exception as e:
        AppLog.info(f"Error creating BigQuery client: {e}")

    # Define schema directly in code
    schema = [
        bigquery.SchemaField("id", "INTEGER"),
        bigquery.SchemaField("date_review", "STRING"),
        bigquery.SchemaField("day_review", "INTEGER"),
        bigquery.SchemaField("month_review", "STRING"),
        bigquery.SchemaField("month_review_num", "INTEGER"),
        bigquery.SchemaField("year_review", "INTEGER"),
        bigquery.SchemaField("verified", "STRING"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("month_fly", "STRING"),
        bigquery.SchemaField("month_fly_num", "FLOAT"),
        bigquery.SchemaField("year_fly", "FLOAT"),
        bigquery.SchemaField("month_year_fly", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("aircraft", "STRING"),
        bigquery.SchemaField("aircraft_1", "STRING"),
        bigquery.SchemaField("aircraft_2", "STRING"),
        bigquery.SchemaField("type", "STRING"),
        bigquery.SchemaField("seat_type", "STRING"),
        bigquery.SchemaField("route", "STRING"),
        bigquery.SchemaField("origin", "STRING"),
        bigquery.SchemaField("destination", "STRING"),
        bigquery.SchemaField("transit", "STRING"),
        bigquery.SchemaField("seat_comfort", "FLOAT"),
        bigquery.SchemaField("cabin_serv", "FLOAT"),
        bigquery.SchemaField("food", "FLOAT"),
        bigquery.SchemaField("ground_service", "FLOAT"),
        bigquery.SchemaField("wifi", "FLOAT"),
        bigquery.SchemaField("money_value", "INTEGER"),
        bigquery.SchemaField("score", "FLOAT"),
        bigquery.SchemaField("experience", "STRING"),
        bigquery.SchemaField("recommended", "STRING"),
        bigquery.SchemaField("review", "STRING"),
    ]

    # Try to create the table
    await create_table_with_schema(client, full_table_id, schema)

if __name__ == "__main__":
    create_schema()