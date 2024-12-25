import os
from typing import Dict
import requests
from bs4 import BeautifulSoup
import pandas as pd
# import AppLog
from utils.logger import AppLog 
from google.cloud import bigquery
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
import time
import json
# Load environment variables from .env file
load_dotenv()

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

class ETL:
    def __init__(self):
        self.base_url = os.getenv("BASE_URL")
        self.page_size = int(os.getenv("PAGE_SIZE"))
        self.project_id = os.getenv("PROJECT_ID")
        self.dataset_id = os.getenv("DATASET_ID")
        self.table_id = os.getenv("TABLE_ID")
        self.client = self.init_client()

    def init_client(self):
            # Lấy đường dẫn tới thư mục chứa file hiện tại (schema.py)
        # current_folder = os.path.dirname(os.path.abspath(__file__))

        # # Xây dựng đường dẫn tới file JSON
        # key_path = os.path.join(current_folder, "group-8-445019-10957479e54c.json")

        # if not os.path.exists(key_path):
        #     raise FileNotFoundError(f"Service account key file not found: {key_path}")

        # # Tạo client BigQuery từ tệp JSON
        # client = bigquery.Client.from_service_account_json(json_credentials_path=key_path)
        # return client



        # Load nội dung JSON của tệp service account từ biến môi trường
        service_account_json = create_keyfile_dict()

        # Chuyển JSON sang chuỗi và tạo client BigQuery từ chuỗi JSON
        try:
            client = bigquery.Client.from_service_account_info(service_account_json)
            print("BigQuery client created successfully.")
            return client
        except Exception as e:
            print(f"Error creating BigQuery client: {e}")
        return None
            
       
    # -------------------- STEP 1: Extract --------------------
    def number_pages(self):
        response = requests.get(self.base_url)
        AppLog.info(self.base_url)
        parsed_content = BeautifulSoup(response.content, 'html.parser')
        div_content = parsed_content.find('div', class_='pagination-total').text
        total_reviews = int(div_content.split('of')[1].split('Reviews')[0].strip())
        page = total_reviews / self.page_size
        page = int(page) + 1 if page % 1 != 0 else int(page)
        return page
    def extract(self) -> pd.DataFrame:
        """
        Extracts review data from British Airways reviews on AirlineQuality.com.

        Args:
            number_of_pages (int): The number of pages to scrape.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted review data.
        """
        reviews_data = []

        for page in range(1, self.number_pages() + 1):
            AppLog.info(f"Scraping page {page}")

            url = f"{self.base_url}/page/{page}/?sortby=post_date%3ADesc&pagesize={self.page_size}"
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.RequestException as e:
                AppLog.error(f"Failed to fetch page {page}: {e}")
                continue

            parsed_content = BeautifulSoup(response.content, 'html.parser')
            reviews = parsed_content.select('article[class*="comp_media-review-rated"]')

            for review in reviews:
                review_data = self.extract_review_data(review)
                reviews_data.append(review_data)

        df = pd.DataFrame(reviews_data)
        return df

    def extract_review_data(self,review: BeautifulSoup) -> Dict[str, str]:
        """
        Extracts relevant data from a single review.

        Args:
            review (BeautifulSoup): The parsed HTML of a single review.

        Returns:
            Dict[str, str]: A dictionary containing extracted review data.
        """
        review_data = {
            'dates': self.extract_text(review, "time", itemprop="datePublished"),
            'customer_names': self.extract_text(review, "span", itemprop="name"),
            'countries': self.extract_country(review),
            'review_bodies': self.extract_text(review, "div", itemprop="reviewBody")
        }

        self.extract_ratings(review, review_data)

        return review_data

    def extract_text(self,element: BeautifulSoup, tag: str, **attrs) -> str:
        """
        Extracts text from a BeautifulSoup element.

        Args:
            element (BeautifulSoup): The BeautifulSoup element to search within.
            tag (str): The HTML tag to look for.
            **attrs: Additional attributes to filter the search.

        Returns:
            str: The extracted text, or None if not found.
        """
        found = element.find(tag, attrs)
        return found.text.strip() if found else None

    def extract_country(self,review: BeautifulSoup) -> str:
        """
        Extracts the country from a review.

        Args:
            review (BeautifulSoup): The parsed HTML of a single review.

        Returns:
            str: The extracted country, or None if not found.
        """
        country = review.find(string=lambda string: string and '(' in string and ')' in string)
        return country.strip('()') if country else None

    def extract_ratings(self,review: BeautifulSoup, review_data: Dict[str, str]) -> None:
        """
        Extracts ratings from a review and adds them to the review_data dictionary.

        Args:
            review (BeautifulSoup): The parsed HTML of a single review.
            review_data (Dict[str, str]): The dictionary to update with extracted ratings.
        """
        review_ratings = review.find('table', class_='review-ratings')
        if not review_ratings:
            return

        for row in review_ratings.find_all('tr'):
            header = row.find('td', class_='review-rating-header')
            if not header:
                continue

            header_text = header.text.strip()
            if header_text in ['Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 'Wifi & Connectivity', 'Value For Money']:
                stars_td = row.find('td', class_='review-rating-stars')
                if stars_td:
                    stars = stars_td.find_all('span', class_='star fill')
                    review_data[header_text] = len(stars)
            else:
                value = row.find('td', class_='review-value')
                if value:
                    review_data[header_text] = value.text.strip()
                        
    # -------------------- STEP 2: Transform --------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform the data."""
        AppLog.info("Starting data transformation...")

        df = (df.pipe(self.clean_country)
                .pipe(self.clean_review)
                .pipe(self.clean_date_review)
                .pipe(self.clean_date_flown)
                .pipe(self.clean_space)
                .pipe(self.create_id)
                .pipe(self.rename_columns)
                .pipe(self.reorder_columns_before_fe)
                .pipe(self.clean_route)
                .pipe(self.split_aircraft_column)
                .pipe(self.clean_aircraft, 'aircraft_1')
                .pipe(self.clean_aircraft, 'aircraft_2')
                .pipe(self.calculate_experience)
                .pipe(self.calculate_service_score)
                .pipe(self.replace_yes_no_upcase,'recommended')
                .pipe(self.replace_yes_no_with_bool, 'verified')
                .pipe(self.reorder_columns_after_fe)
                .pipe(self.fill_nulls))

        AppLog.info("Data transformation completed.")
        return df


    # --------------------- STEP 3: Load --------------------
    def load(self, df: pd.DataFrame):
        """Load data into BigQuery."""
        try:
            import pyarrow  # Check if pyarrow is installed
        except ImportError as e:
            AppLog.error("Pyarrow is not installed. Please install it using 'pip install pyarrow'.")
            raise e

        table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Overwrite data each time TRUCATED
        )
        
        try:
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()  # Wait for the job to complete
            AppLog.info(f"Uploaded {len(df)} rows to BigQuery table {self.table_id}")
        except Exception as e:
            AppLog.error(f"Failed to load data to BigQuery: {e}")
            raise

    # -------------------- TRANSFORM FUNCTIONS --------------------
    def clean_country(self, df: pd.DataFrame) -> pd.DataFrame:
        df['countries'] = df['countries'].str.replace(r'[()]', '', regex=True)
        return df
    def clean_review(self,df: pd.DataFrame) -> pd.DataFrame:
        if 'review_bodies' not in df.columns:
            return df

        split_df = df['review_bodies'].str.split('|', expand=True)

        if len(split_df.columns) == 1:
            df['review'] = split_df[0]
            df['verified'] = pd.NA
        else:
            df['verified'], df['review'] = split_df[0], split_df[1]

        mask = df['review'].isnull() & df['verified'].notnull()
        df.loc[mask, ['review', 'verified']] = df.loc[mask, ['verified', 'review']].values

        df['verified'] = df['verified'].str.contains('Trip Verified', case=False, na=False)

        df.drop(columns=['review_bodies'], inplace=True)
        return df
    def clean_date_review(self, df: pd.DataFrame) -> pd.DataFrame:
        # Split the 'dates' column into Day, Month, and Year components
        df[['Day Review', 'Month Review', 'Year Review']] = df['dates'].str.split(expand=True)

        # Remove the ordinal suffix (e.g., "12th" -> "12") from the day component
        df['Day Review'] = df['Day Review'].str[:-2]

        # Concatenate the components into a single string in the desired date format
        df['Dates Review'] = (
            df['Day Review'] + ' ' + df['Month Review'] + ' ' + df['Year Review']
        )

        # Ensure the result remains a string
        df['Dates Review'] = df['Dates Review'].astype(str)

        return df

    def clean_date_flown(self,df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns={'Date Flown': 'Month Flown'}, inplace=True)
        df[['Month Flown', 'Year Flown']] = df['Month Flown'].str.split(' ', expand=True)

        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }

        df['Month Flown Number'] = df['Month Flown'].map(month_mapping)
        df['Month Review Number'] = df['Month Review'].map(month_mapping)

        df['Month Flown Number'] = pd.to_numeric(df['Month Flown Number'], errors='coerce').astype('Int64')

        df['Month Year Flown'] = pd.to_datetime(
            df['Year Flown'].astype(str) + '-' + df['Month Flown Number'].astype(str).str.zfill(2) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        ).dt.strftime('%m-%Y')

        return df

    def clean_space(self,df: pd.DataFrame) -> pd.DataFrame:
        return df.map(lambda x: x.strip() if isinstance(x, str) else x)

    def create_id(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by='Dates Review', ascending=False)
        df['id'] = range(len(df))
        return df

    def rename_columns(self,df: pd.DataFrame) -> pd.DataFrame:
        new_column_names: Dict[str, str] = {
            'Dates Review': 'date_review',
            'Day Review': 'day_review',
            'Month Review': 'month_review',
            'Month Review Number': 'month_review_num',
            'Year Review': 'year_review',
            'customer_names': 'name',
            'Month Flown': 'month_fly',
            'Month Flown Number': 'month_fly_num',
            'Year Flown': 'year_fly',
            'Month Year Flown': 'month_year_fly',
            'countries': 'country',
            'Aircraft': 'aircraft',
            'Type Of Traveller': 'type',
            'Seat Type': 'seat_type',
            'Route': 'route',
            'Seat Comfort': 'seat_comfort',
            'Cabin Staff Service': 'cabin_serv',
            'Food & Beverages': 'food',
            'Ground Service': 'ground_service',
            'Wifi & Connectivity': 'wifi',
            'Value For Money': 'money_value',
            'Recommended': 'recommended'
        }
        return df.rename(columns=new_column_names)

    def reorder_columns_before_fe(self,df: pd.DataFrame) -> pd.DataFrame:
        column_order = [
            'id', 'verified', 'date_review', 'day_review', 'month_review', 'month_review_num', 'year_review',
            'name', 'month_fly', 'month_fly_num', 'year_fly', 'month_year_fly', 'country', 'aircraft', 'type',
            'seat_type', 'route', 'seat_comfort', 'cabin_serv', 'food', 'ground_service', 'wifi', 'money_value',
            'recommended', 'review'
        ]
        return df[column_order]
    def fill_nulls(self,df):
        for col in df.columns:
            if df[col].dtype == 'object':  # Cột dạng object (chuỗi)
                df[col] = df[col].fillna("Not mentioned")
            elif np.issubdtype(df[col].dtype, np.number):  # Cột dạng số (int, float)
                df[col] = df[col].fillna(np.nan)
            elif np.issubdtype(df[col].dtype, np.datetime64):  # Cột dạng datetime
                df[col] = df[col].fillna(np.nan)
        return df
    def clean_route(self,df: pd.DataFrame) -> pd.DataFrame:

        def split_route(route: str) -> tuple:
            if pd.isna(route):
                return pd.NA, pd.NA, pd.NA
            if ' to ' in route:
                parts = route.split(' to ')
                origin = parts[0].strip()
                if len(parts) > 1:
                    destination, transit = parts[1].split(' via ') if ' via ' in parts[1] else (parts[1], None)
                else:
                    destination, transit = None, None
            else:
                parts = route.split('-')
                origin = parts[0].strip()
                destination = parts[1].strip() if len(parts) > 1 else None
                transit = None
            return origin.strip(), destination.strip() if destination else None, transit.strip() if transit else None

        df[['origin', 'destination', 'transit']] = df['route'].apply(split_route).apply(pd.Series)

        def normalize_city_names(city_name):
        # Chuyển về chữ thường để dễ xử lý
          city_name = city_name.lower()

          # Từ điển ánh xạ lỗi sai
          city_mapping = {
              'hanoi': 'Ha Noi', 'hanoi city': 'Ha Noi', 'hanoi, vietnam': 'Ha Noi',
              'han': 'Ha Noi', 'hnoi': 'Ha Noi', 'hanoI': 'Ha Noi', 'h.noi': 'Ha Noi', 'hn': 'Ha Noi',
              'ho chi minh city': 'Ho Chi Minh City', 'hcm': 'Ho Chi Minh City', 'hcmc': 'Ho Chi Minh City',
              'ho chi minh': 'Ho Chi Minh City', 'saigon': 'Ho Chi Minh City', 'hcm city': 'Ho Chi Minh City',
              'ho chi minh': 'Ho Chi Minh City',
              'da nang': 'Da Nang', 'danang': 'Da Nang', 'dn': 'Da Nang', 'danang city': 'Da Nang'
          }

          # Kiểm tra nếu tên thành phố có trong từ điển
          for key in city_mapping:
              if key in city_name:
                  return city_mapping[key]

          return city_name.title()
        # Áp dụng hàm chuẩn hóa cho các cột 'origin', 'destination', 'transit'
        df[['origin', 'destination', 'transit']] = df[['origin', 'destination', 'transit']].applymap(lambda x: normalize_city_names(str(x)))
        return df

    def split_aircraft_column(self,df: pd.DataFrame) -> pd.DataFrame:

        split_aircraft = df['aircraft'].str.split('/|-|,|&', expand=True)
        split_aircraft.columns = [f'aircraft_{i+1}' for i in range(split_aircraft.shape[1])]
        split_aircraft = split_aircraft[['aircraft_1', 'aircraft_2']]
        return pd.concat([df, split_aircraft], axis=1)

    def clean_aircraft(self,df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        replacements = [
            (r'(?i)Boeing (\d+)', r'B\1'),
            (r'(?i)777', 'B777'),
            (r'(?i)A(\d+)', r'A\1'),
            (r'(?i)170', 'E170'),
            (r'(?i)190', 'E190'),
        ]

        for pattern, replacement in replacements:
            df[column_name] = df[column_name].str.replace(pattern, replacement, regex=True)

        df[column_name] = df[column_name].str.extract(r'(?i)([A-Z]\d+)', expand=False)

        return df
    def calculate_experience(self,df: pd.DataFrame) -> pd.DataFrame:
        conditions = [
            (df['money_value'] <= 2),
            (df['money_value'] == 3),
            (df['money_value'] >= 4)
        ]
        choices = ['Poor', 'Fair', 'Good']
        df['experience'] = np.select(conditions, choices, default='unknown')
        return df

    def calculate_service_score(self, df: pd.DataFrame) -> pd.DataFrame:
        # List of columns to calculate the score
        columns = ['seat_comfort', 'cabin_serv', 'food', 'ground_service']

        # Replace "Not mentioned" with 0 in the selected columns
        df[columns] = df[columns].replace("Not mentioned", 0)

        # Ensure the columns are of numeric type
        df[columns] = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Calculate the mean score
        df['score'] = df[columns].mean(axis=1)

        return df

    def replace_yes_no_with_bool(self,df: pd.DataFrame, column: str) -> pd.DataFrame:
      df[column] = df[column].replace({True: "Yes", False: "No"})
      return df
    def replace_yes_no_upcase(self,df: pd.DataFrame, column: str) -> pd.DataFrame:
      df[column] = df[column].replace({"yes": "Yes", "no": "No"})
      return df

    def reorder_columns_after_fe(self,df: pd.DataFrame) -> pd.DataFrame:
        column_order = [
            'id', 'date_review', 'day_review', 'month_review', 'month_review_num',
            'year_review', 'verified', 'name', 'month_fly', 'month_fly_num',
            'year_fly', 'month_year_fly', 'country', 'aircraft', 'aircraft_1',
            'aircraft_2', 'type', 'seat_type', 'route', 'origin', 'destination', 'transit',
            'seat_comfort', 'cabin_serv', 'food', 'ground_service', 'wifi', 'money_value',
            'score', 'experience', 'recommended', 'review'
        ]
        return df[column_order]

    # -------------------- ETL Runner --------------------
def run_etl():
    """Execute the ETL process."""
    etl = ETL()
    AppLog.info("Starting ETL process...")
    # Step 1: Extract
    df = etl.extract()
    AppLog.info(f"Extracted {len(df)} rows of data.")
    # Step 2: Transform
    df = etl.transform(df)
    # Step 3: Load
    etl.load(df)
    AppLog.info("ETL process completed.")

if __name__ == "__main__":
    run_etl()



    

