from selenium import webdriver
import os
import time
from fake_useragent import UserAgent
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor
import warnings
import json

warnings.filterwarnings("ignore")

# Chrome options
options = webdriver.ChromeOptions()

class Scrapper():
    def __init__(self, url):
        ua = UserAgent()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'user-agent={ua.random}')
        # options.add_argument('--headless')  # Comment out for debugging

        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values": {
                "cookies": 2, "images": 2, 
                "plugins": 2, "popups": 2, "geolocation": 2, 
                "notifications": 2, "auto_select_certificate": 2, "fullscreen": 2, 
                "mouselock": 2, "mixed_script": 2, "media_stream": 2, 
                "media_stream_mic": 2, "media_stream_camera": 2, 
                "protocol_handlers": 2, "ppapi_broker": 2, "automatic_downloads": 2, 
                "midi_sysex": 2, "push_messaging": 2, "ssl_cert_decisions": 2, 
                "metro_switch_to_desktop": 2, "protected_media_identifier": 2, 
                "app_banner": 2, "site_engagement": 2, "durable_storage": 2
            }
        }
        options.add_experimental_option("prefs", prefs)

        service = Service("/opt/homebrew/bin/chromedriver")
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)

        # Wait for the history tab to be present, then click it
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.ID, 'ctl00_ContentPlaceHolder1_CompanyDetail1_lnkDividendTab'))
        )
        self.driver.find_element(By.ID, 'ctl00_ContentPlaceHolder1_CompanyDetail1_lnkDividendTab').click()
        self.driver.implicitly_wait(2)

    def df(self):
        max_attempts = 3  # Set a maximum retry count
        all_data = []

        while True:
            for attempt in range(max_attempts):
                try:
                    # Wait for the table to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDividendData"]/div[2]/table/tbody/tr'))
                    )

                    # Extract all rows from the table
                    rows = self.driver.find_elements(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDividendData"]/div[2]/table/tbody/tr')

                    # print(rows)

                    # Iterate over rows and extract the required data
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if len(cols) >= 1:  # Ensure there are enough columns
                            all_data.append({
                                'Fiscal Year': cols[1].text,
                                'Cash Dividend': cols[2].text,
                                'Bonus Share': cols[3].text,
                                'Right Share': cols[4].text
                            })
                    break  # Break out of retry loop if successful
                except StaleElementReferenceException:
                    print(f"Stale element encountered, retrying... (attempt {attempt})")
                    time.sleep(2)
                except Exception as e:
                    print(f"Failed to scrape table: {e}")
                    return pd.DataFrame(all_data)  # Return what we have so far

            # Try to click the "Next" button if it exists and is enabled
            try:
                next_btn = self.driver.find_element(
                    By.XPATH,
                    '//div[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDividendData"]//a[contains(text(),"Next")]'
                )
                # Check if the button is disabled
                if 'disabled' in next_btn.get_attribute('class').lower() or not next_btn.is_enabled():
                    break  # No more pages
                else:
                    next_btn.click()
                    time.sleep(1)  # Wait for the next page to load
            except Exception as e:
                print(f"No next button found or error: {e}")
                break  # No next button found, exit loop

        return pd.DataFrame(all_data)


    def datas(self):
        print("Started scraping for:", self.url)
        data = self.df()  # Only scrape the first page
        return data
    
    def close(self):
        self.driver.quit()

def save_datas(symbol, out_dir):
    url = f'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}'
    scrapper = Scrapper(url)
    try:
        data = scrapper.datas()
        if data.empty:
            print(f"No data found for {symbol}, skipping save.")
            return
        filename = f"{symbol}.csv"
        path = os.path.join(out_dir, filename)
        data.to_csv(path, index=False)
        print(f"Data saved for {symbol} in {path}")
    finally:
        scrapper.close()


if __name__ == '__main__':
    
    with open('company_list.json', 'r') as f:
        company_list = json.load(f)
    stock_symbols = [c['d'] for c in company_list]

    # Output directory for CSVs
    data_dir = os.path.join(os.getcwd(), 'dividend_data')
    os.makedirs(data_dir, exist_ok=True)

    max_workers = 5  # Adjust as needed

    # Use ProcessPoolExecutor for parallel scraping
    from functools import partial
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(partial(save_datas, out_dir=data_dir), stock_symbols)