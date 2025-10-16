import os.path
import sys
sys.path.append('../../Lib')
from Google.common_funcs import *

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
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
from gspread_dataframe import set_with_dataframe

class File():
    def __init__(self, 
                 file_id: str, 
                 file_path_by_ids: tp.Tuple[str], 
                 file_name: str, 
                 file_path_by_names: tp.Tuple[str], 
                 file_type: str,
                 parent_file
                ):
        args = inspect.getfullargspec(getattr(self, '__init__')).args[1:]        
        for key in args:
            setattr(self, key, locals()[key])
        
        path = "/".join(inspect.getfile(self.__class__).split('/')[:-1])
        path = path + '/file_str_repr.txt'
        with open(path, 'r') as print_file:
            self.str_body = "".join(list(print_file.readlines()))


    def __str__(self) -> str:
        ret = self.str_body.format(
            file_name=self.file_name,
            file_id=self.file_id,
            file_type=self.file_type,
            file_path_by_names=str(self.file_path_by_names),
            parent_name= self.parent_file.file_name if self.parent_file is not None else '-'
        )
        return ret
    

class Googlya():
    
    def __init__(self, 
                 main_path,
                 main_folder_id,
                 token_path: str = '', 
                 credentials_path: tp.Optional[str] = '', 
                 local_port: int = 8008,
                ):
        """
            Инициализация. 
            Поиск credentials:
            https://console.cloud.google.com/apis/credentials?project=balmy-ocean-325310&folder=&organizationId=
            
            token_path - путь до токена, сгенерированного после первого успешного запуска.
            Для первого запуска нужен credentials.json и нужен проброс порта с виртуалки на локалку local_port.
        """
        
        SCOPES = ['https://www.googleapis.com/auth/drive', 
                  'https://www.googleapis.com/auth/spreadsheets',
                 ]
        creds = None
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
        
        self.client = gspread.authorize(credentials=creds)
        self.sheets_service = build('sheets', 'v4', credentials=creds)
        self.service_v3 = build('drive', 'v3', credentials=creds)
        self.service_v2 = build('drive', 'v2', credentials=creds)
        self.main_folder_id = main_folder_id
        self.main_folder_name = self.get_file_by_id(main_folder_id)['title']
        self.create_full_tree(main_path)

         
    def get_file_by_id(self, file_id: str):
        return self.service_v2.files().get(
            fileId=file_id, 
            supportsAllDrives=True
        ).execute()
        
    
    def get_type(self, name: str, mimeType: str):
        if 'google' in mimeType:
            return 'google/' + mimeType.split('.')[-1]
        return 'file/' + name.split('.')[-1]
    

    def create_full_tree(self, main_path: tp.List[str]):
        self.files_dict = {
            self.main_folder_id: File(
                    self.main_folder_id,
                    tuple([self.main_folder_id]), 
                    self.main_folder_name,
                    tuple([self.main_folder_name]),
                    'root google folder',
                    None
                )
        }
        self.get_all_file_tree(self.files_dict[self.main_folder_id], main_path)
        
    
    def check_path(self, curr_path: tp.Tuple[str], main_path: tp.List[str]):
        min_len = min(len(main_path), len(curr_path))

        if list(curr_path[:min_len]) == list(main_path[:min_len]):
            return True
        return False


    def get_all_file_tree(self, curr_file, main_path: tp.List[str]):
        page_token = None
        while True:
            try:
                param = {}
                if page_token:
                    param['pageToken'] = page_token
                children = self.service_v2.children().list(
                    folderId=curr_file.file_id, **param
                ).execute()
                page_token = children.get('nextPageToken')
                for child in children.get('items', []):
                    file_info = self.get_file_by_id(child['id'])
                    name = file_info['title']
                    if file_info['labels']['trashed']:
                        continue
                    
                    curr_path = tuple(curr_file.file_path_by_names + tuple([name]))
                    if not self.check_path(curr_path, main_path):
                        continue
                    child_file = File(
                        child['id'],
                        tuple(curr_file.file_path_by_ids + tuple([child['id']])),
                        name,
                        tuple(curr_file.file_path_by_names + tuple([name])),
                        self.get_type(name, file_info['mimeType']),
                        curr_file
                    )
                    self.files_dict[child['id']] = child_file
#                     print(child_file)
                    self.get_all_file_tree(child_file, main_path)
                    
                if not page_token:
                    break

            except Exception as err:
                print(f'An error occurred:\n{str(err)}')
                break
    
    
    def get_file_by_tags(self, tags: tp.List[str]):
        answer_files = []
        for file_id, file in self.files_dict.items():
            common_tags = set(tags) & set(file.file_path_by_names)
            if common_tags == set(tags) and file.file_name in tags:
                answer_files.append(file)
                
        assert len(answer_files) <= 1, f"По {tags} найдено больше одного файла: {len(answer_files)}. Проверьте корзину."
        if len(answer_files) == 1:
            return answer_files[0]
        return None
    
    
    def create_googlesheets(self, tags_path, name):
        spreadsheet = {
            'properties': {
                'title': name
            }
        }
        try:
            already_exists_id = self.get_file_by_tags(tags_path + [name]).file_id
            if already_exists_id is not None:
                return already_exists_id
        except Exception:
            pass
        
        sheetsfolder = self.get_file_by_tags(tags_path)
        spreadsheet = self.sheets_service.spreadsheets().create(body=spreadsheet,
                                                                    fields='spreadsheetId').execute()
        spreadsheet_id = spreadsheet.get('spreadsheetId')
        file = self.service_v3.files().update(fileId=spreadsheet_id,
                                                addParents=sheetsfolder.file_id,
                                                fields='id, parents', supportsAllDrives=True).execute()
        file_info = self.get_file_by_id(spreadsheet_id)
        self.files_dict[spreadsheet_id] = File(
                        spreadsheet_id,
                        tuple(sheetsfolder.file_path_by_ids + tuple([spreadsheet_id])),
                        name,
                        tuple(sheetsfolder.file_path_by_names + tuple([name])),
                        self.get_type(name, file_info['mimeType']),
                        sheetsfolder
                    )
        return spreadsheet_id


    def delete_googlesheets_file(self, tags_path, list_name):
        sheetsfile = self.get_file_by_tags(tags_path)
        sh = self.client.open_by_key(sheetsfile.file_id)
        try:
            sheet = sh.worksheet(list_name)
        except Exception:
            print('file does not exist!')
            return
        body=delete_list_json(sheet.id)
        self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=sheetsfile.file_id,  body=body).execute()
        

    def get_id_by_path(self, tuple_path: tp.Tuple[str]):
        for file_id, file in self.files_dict.items():
            if file.file_path_by_names == tuple_path:
                return file.file_id
        return None

    
    def create_folder(self, folder_name, parent_folder):
        body = {
            'name': folder_name,
            'mimeType': "application/vnd.google-apps.folder",
            'parents': [parent_folder.file_id]
        }
        folder = self.service_v3.files().create(body = body, supportsAllDrives=True).execute()
        curr_file = File(
            folder['id'],
            tuple(parent_folder.file_path_by_ids + tuple([folder['id']])),
            folder_name,
            tuple(parent_folder.file_path_by_names + tuple([folder_name])),
            self.get_type(folder_name, folder['mimeType']),
            parent_folder
        )
        self.files_dict[folder['id']] = curr_file
        return folder['id'] 


    def save_file(self, file_name: str, path_to_file_on_vm, parent_folder = None):
        
        file_metadata = {
            'name': file_name,
            'mimeType': '*/*',
            'parents': [parent_folder.file_id]
        }
        media = MediaFileUpload(path_to_file_on_vm,
                                mimetype='*/*',
                                resumable=True)
        file = self.service_v3.files().create(body=file_metadata, media_body=media, fields='id', 
                                              supportsAllDrives=True).execute()
        file_info = self.get_file_by_id(file['id'])
        curr_file = File(
            file['id'],
            tuple(parent_folder.file_path_by_ids + tuple([file['id']])),
            file_name,
            tuple(parent_folder.file_path_by_names + tuple([file_name])),
            self.get_type(file_name, file_info['mimeType']),
            parent_folder
        )
        self.files_dict[file['id']] = curr_file
        return file['id']


    def create_path(self, file_path_to_save: tp.Tuple[str]):
        current_path = []
        for folder in file_path_to_save:
            current_path.append(folder)
            curr_parent_id = self.get_id_by_path(tuple(current_path))
            if curr_parent_id is None:
                curr_parent_id = self.create_folder(folder, self.files_dict[parent_id])
            parent_id = curr_parent_id
        return parent_id


    def create_path_and_save_file(self, file_path_to_save: tp.Tuple[str], path_to_file_on_vm: str):
        parent_id = None
        file_name = file_path_to_save[-1]
        if self.get_id_by_path(file_path_to_save) is not None:
            return

        parent_id = self.create_path(file_path_to_save[:-1])
        return self.save_file(file_name, path_to_file_on_vm, self.files_dict[parent_id])
        
    
    def make_raw_res(self, df, tags_path, sheet_name):
        sheetsfile = self.get_file_by_tags(tags_path)
        sh = self.client.open_by_key(sheetsfile.file_id)
        try:
            sheet = sh.worksheet(sheet_name)
        except Exception:
            sheet = sh.add_worksheet(sheet_name, rows=1, cols=1)
        set_with_dataframe(sheet, df, include_index=False, resize=True)
        return sheetsfile, sheet
    
    
    def make_or_get_sheet(self, tags_path, sheet_name):
        sheetsfile = self.get_file_by_tags(tags_path)
#         print(sheetsfile)
        sh = self.client.open_by_key(sheetsfile.file_id)
        try:
            sheet = sh.worksheet(sheet_name)
        except Exception:
            sheet = sh.add_worksheet(sheet_name, rows=100, cols=20)
        return sheetsfile, sheet
    
    

    def make_table_with_logcat_as_row(self, df, tags_path, save_name):
        sheetsfile = self.get_file_by_tags(tags_path)
        sh = self.client.open_by_key(sheetsfile.file_id)
        try:
            sheet = sh.worksheet(save_name)
        except Exception:
            sheet = sh.add_worksheet(save_name, rows=1, cols=1)
        set_with_dataframe(sheet, df, include_index=True, resize=True) 
        sheet_id = sheet.id
        DATA = {"requests": 
                [
                    make_text_montserrat(sheet_id),
                    make_header(sheet_id),
                    make_left_columns(sheet_id),
                    make_numbers_with_size(sheet_id, startRowIndex=1),
                ] + 
                [
                    {
                      "autoResizeDimensions": {
                        "dimensions": {
                          "sheetId": sheet_id,
                          "dimension": "COLUMNS",
                          "startIndex": 0,
                        }
                      }
                    }
                ]
        }
        self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=sheetsfile.file_id, body=DATA).execute()
        sheet.format("A:Z", {"wrapStrategy": "WRAP"})

        return


    def get_raw_sheet_data(self, file_id, sheet_name):
        rows = self.sheets_service.spreadsheets().values().get(
            spreadsheetId=file_id,
            range=sheet_name).execute()
        raw_table = rows.get('values')
        
        columns = raw_table[0]
        try:
            columns = columns[:columns.index('')]
        except Exception:
            pass
        table_len = len(columns)
        df = pd.DataFrame([row[:table_len] + [None for _ in range(table_len - len(row[:table_len]))] for row in raw_table[1:]], columns=columns)
        df.fillna('', inplace=True)
        return df