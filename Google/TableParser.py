import os.path
import sys
# sys.path.append('/var/AAA/lib')

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import io
import requests
import json
import typing as tp
import inspect
from copy import deepcopy
from tqdm.notebook import tqdm as tqdm_notebook
import gspread
import pathlib
import pandas as pd
import csv
import shutil



class GoogleSheetsParser():
    
    def __init__(self, token_path: str = '/var/AAA/google_secrets/token.json', 
                 credentials_path: str = '/var/AAA/google_secrets/credentials.json'):
        SCOPES = [
            'https://www.googleapis.com/auth/drive', 
            'https://www.googleapis.com/auth/spreadsheets',
        ]
        creds = None
        # Define local port for OAuth server
        local_port = 0  # This will use a random available port
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=local_port)

            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        self.sheets_service = build('sheets', 'v4', credentials=creds)
        
    
    def read_sheet(self, sheet_code, sheet_list_name, skip_rows=0):
        rows = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=sheet_code,
            range=sheet_list_name
        ).execute()
        data = rows.get('values')
        columns = data[skip_rows]
        col_len = len(columns)
        rows = [row[:col_len] for row in data[1 + skip_rows:]]
        df = pd.DataFrame(rows, columns=columns)
        return df