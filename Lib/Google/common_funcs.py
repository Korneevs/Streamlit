white_color = {
    "red": 1,
    "green": 1,
    "blue": 1
}

yellow_color = {
    "red": 255/255,
    "green": 241/255,
    "blue": 204/255
}

orange_color = {
    "red": 252/255,
    "green": 229/255,
    "blue": 205/255
}


grey_color = {
    "red": 204/255,
    "green": 204/255,
    "blue": 204/255
}

green_color = {
    "red": 217/255,
    "green": 234/255,
    "blue": 211/255
}

red_color = {
    "red": 238/255,
    "green": 205/255,
    "blue": 205/255
}

blue_color = {
    "red": 207/255,
    "green": 226/255,
    "blue": 243/255
}


dark_green_1_color = {
    "red": 106/255,
    "green": 168/255,
    "blue": 79/255
}

light_magenta_3_color = {
    "red": 234/255,
    "green": 209/255,
    "blue": 220/255
}

light_blue_3_color = {
    "red": 207/255,
    "green": 226/255,
    "blue": 243/255
}

light_cyan_3_color = {
    "red": 208/255,
    "green": 224/255,
    "blue": 227/255
}

light_yellow_3_color = {
    "red": 252/255,
    "green": 242/255,
    "blue": 204/255
}

light_red_3_color = {
    "red": 244/255,
    "green": 204/255,
    "blue": 204/255
}


def delete_list_json(sheet_id):
    delete_body={
      "requests": [
        {
            "deleteSheet": {
                "sheetId": sheet_id
            }
        }
      ]
    }
    return delete_body


def get_text_format(is_bold=False, fontSize=10, 
                    fontFamily='Montserrat',
                    is_centered=False, color_dict=None, is_top=False, 
                    use_number_format=False, use_money_format = ""):
    json = {
        "textFormat": {
            "fontSize": fontSize,
            'fontFamily': fontFamily,
            "bold": is_bold
        },
    }
    if use_number_format:
        json["numberFormat"] = {
                "type": "NUMBER",
                'pattern': "0.#%",
        }
    if use_money_format == 'K':
        json["numberFormat"] = {
                "type": "NUMBER",
                'pattern': '0.0,"K"',
        }
    if use_money_format == 'M':
        json["numberFormat"] = {
                "type": "NUMBER",
                'pattern': '0.0,,"M"',
        }
    if color_dict is not None:
        json["backgroundColor"] = color_dict
    if is_centered:
        json["horizontalAlignment"] = "CENTER"
        json["verticalAlignment"] = "MIDDLE"
    if is_top:
        json["verticalAlignment"] = "TOP"
    return json


def update_value_in_cell(sheet_id,
                         startColumnIndex, 
                         startRowIndex,
                         text,
                         cell_type="text",
                        ):
    text_dict = {
       "stringValue": text
    }
    if cell_type == 'formula':
        text_dict = {
           "formulaValue": text
        }
    if cell_type == 'number':
        text_dict = {
           "numberValue": text
        }
    body = {"updateCells": {
        "range": {
            "sheetId": sheet_id,
            "startColumnIndex": startColumnIndex - 1,
            "endColumnIndex": startColumnIndex,
            "startRowIndex": startRowIndex - 1,
            "endRowIndex": startRowIndex
        },
        "fields": "*",
        "rows": [{
            "values": [
                {
                  "userEnteredValue": text_dict,
                }
            ]
        }]
        }
    }
    return body


def format_cells_body(sheet_id, 
                      startColumnIndex, endColumnIndex, 
                      startRowIndex, endRowIndex,
                      is_bold, fontSize, is_centered, color_dict,
                      fontFamily='Montserrat', is_top=False, use_number_format="",
                      use_money_format=False
                     ):
    body = {"repeatCell": {
        "range": {
            "sheetId": sheet_id,
            "startColumnIndex": startColumnIndex - 1,
            "endColumnIndex": endColumnIndex - 1,
            "startRowIndex": startRowIndex - 1,
            "endRowIndex": endRowIndex - 1

        },
        "cell": {
            "userEnteredFormat": get_text_format(is_bold=is_bold, 
                                                 fontSize=fontSize,
                                                 is_centered=is_centered,
                                                 fontFamily=fontFamily,
                                                 color_dict=color_dict,
                                                 is_top=is_top,
                                                 use_number_format=use_number_format,
                                                 use_money_format=use_money_format
                                                ),
        },
        "fields": "userEnteredFormat(backgroundColor, textFormat, horizontalAlignment, verticalAlignment, numberFormat)"
    }}
    return body


def merge_cells_body(sheet_id, 
                       startColumnIndex, endColumnIndex, 
                       startRowIndex, endRowIndex):
    body = {'mergeCells': {
                'mergeType': 'MERGE_ALL',
                 'range': {
                     'sheetId': sheet_id,
                     'startColumnIndex': startColumnIndex - 1,
                     'endColumnIndex': endColumnIndex - 1,
                     'startRowIndex': startRowIndex - 1,
                     'endRowIndex': endRowIndex - 1,
                 }
            }}
    return body


def set_width(sheet_id, 
              startColumnIndex, endColumnIndex, pixel_size):
    body = {
    "updateDimensionProperties": {
        "range": {
          "sheetId": sheet_id,
          "dimension": "COLUMNS",
          "startIndex": startColumnIndex - 1,
          "endIndex": endColumnIndex - 1
        },
        "properties": {
          "pixelSize": pixel_size
        },
        "fields": "pixelSize"
        }
    }
    return body


def set_height(sheet_id, 
              startRowIndex, endRowIndex, pixel_size):
    body = {
    "updateDimensionProperties": {
        "range": {
          "sheetId": sheet_id,
          "dimension": "ROWS",
          "startIndex": startRowIndex - 1,
          "endIndex": endRowIndex - 1
        },
        "properties": {
          "pixelSize": int(pixel_size)
        },
        "fields": "pixelSize"
        }
    }
    return body


def make_cond_formatting(sheet_id, startRowIndex, startColumnIndex, formula, color):
    body = {
      "addConditionalFormatRule": {
        "rule": {
          "ranges": [
            {
              "sheetId": sheet_id,
              "startColumnIndex": startColumnIndex - 1,
              "endColumnIndex": startColumnIndex,
              "startRowIndex": startRowIndex - 1,
              "endRowIndex": startRowIndex
            }
          ],
          "booleanRule": {
            "condition": {
              "type": "CUSTOM_FORMULA",
              "values": [
                {
                  "userEnteredValue": formula
                }
              ]
            },
            "format": {
              "backgroundColor": color
            }
          }
        },
        "index": 0
      }
    }
    return body


def make_gradient_coloring(sheet_id, startColumnIndex, min_color, max_color, mid_color=None):
    if mid_color == None:
        grad_rule = {
            "minpoint": {
              "color": min_color,
              "type": "MIN"
            },
            "maxpoint": {
              "color": max_color,
              "type": "MAX"
            },
          }
    else:
        grad_rule = {
            "minpoint": {
              "color": min_color,
              "type": "MIN"
            },
            "maxpoint": {
              "color": max_color,
              "type": "MAX"
            },
            "midpoint": {
              "color": mid_color,
              "type": "PERCENTILE",
              "value": '50'
            },
          }
    body = {
        "addConditionalFormatRule": {
        "rule": {
          "ranges": [
            {
              "sheetId": sheet_id,
              "startColumnIndex": startColumnIndex - 1,
              "endColumnIndex": startColumnIndex,
              "startRowIndex":  1,
            }
          ],
          "gradientRule": grad_rule
        },
        "index": 0
      }
    }
    return body


def make_drop_down_list_body(sheet_id, values, startRowIndex, startColumnIndex):
    body = {"setDataValidation": {
        "range": {
          "sheetId": sheet_id,
          "startRowIndex": startRowIndex - 1,
          "endRowIndex": startRowIndex,
          "startColumnIndex": startColumnIndex - 1,
          "endColumnIndex": startColumnIndex
        },
        "rule": {
          "condition": {
            "type": 'ONE_OF_LIST',
            "values": [
              {
                  "userEnteredValue": v,
              } for v in values
            ],
          },
          "showCustomUi": True,
          "strict": True
        }
      }
    }
    return body



def hide_columns(sheet_id, startColumnIndex, endColumnIndex):
    body = {
      'updateDimensionProperties': {
        "range": {
          "sheetId": sheet_id,
          "dimension": 'COLUMNS',
          "startIndex": startColumnIndex - 1,
          "endIndex": endColumnIndex - 1,
        },
        "properties": {
          "hiddenByUser": True,
        },
        "fields": 'hiddenByUser',
    }}
    return body



def common_coloring_part(my_range, cond, value, color):
        return {'rule': {
                'ranges': [my_range],
                'booleanRule': {
                    "condition": {
                        'type': cond,
                        'values': [{
                            'userEnteredValue':value
                        }]
                    },
                    "format": {
                      "backgroundColor": color 
                    }
                }
            }
        }
    
    
def add_rows(sheet_id, startIndex, endIndex):
    return {
      "insertDimension": {
        "range": {
          "sheetId": sheet_id,
          "dimension": "ROWS",
          "startIndex": startIndex - 1,
          "endIndex": endIndex - 1
        },
        "inheritFromBefore": False
      }
    },



def merge_cells_body(sheet_id, 
                       startColumnIndex, endColumnIndex, 
                       startRowIndex, endRowIndex):
    body = {'mergeCells': {
                'mergeType': 'MERGE_ALL',
                 'range': {
                     'sheetId': sheet_id,
                     'startColumnIndex': startColumnIndex - 1,
                     'endColumnIndex': endColumnIndex - 1,
                     'startRowIndex': startRowIndex - 1,
                     'endRowIndex': endRowIndex - 1,
                 }
            }}
    return body
