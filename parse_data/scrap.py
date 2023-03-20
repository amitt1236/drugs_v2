# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

def get_summary(file):
    fail = 0
    lst = pd.read_csv(file) 

    for i in  tqdm(range(len(lst))):
        # Set the URL you want to webscrape from
        url = 'https://go.drugbank.com/drugs/' + lst.iloc[i]['drugbank_id'] 

        # Connect to the URL
        response = requests.get(url)

        if response == '<Response [200]>':
            continue
        # Parse HTML and save to BeautifulSoup object¶
        soup = BeautifulSoup(response.text, "html.parser")
        
        sum = soup.find("dt", {"id": "summary"})

        if sum is None:
            fail = fail + 1
            continue

        lst.loc[i, 'indication'] = sum.next.next.p.text
    
    lst.to_csv('sum.csv')
    print(fail)
if __name__ == "__main__":
    get_summary('./parse_data/drugs_all.csv')
