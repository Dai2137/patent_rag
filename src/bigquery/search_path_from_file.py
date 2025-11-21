import pandas as pd
from pathlib import Path
import re
import numpy as np

PATH_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'path'
NUM_PATH = PATH_DATA_DIR / "patent_path_numpy.npy"

# NUM_PATH存在確認、なければメッセージを表示して終了
# NUM_PATH_ARRAYの変数に、NUM_PATHを読み込んだ結果を格納
if not NUM_PATH.exists():
    raise FileNotFoundError(f"Required file not found: {NUM_PATH}")

NUM_PATH_ARRAY = np.load(NUM_PATH)

# document_typeを整数に変換するマッピング（numpy_file.pyと同じ）
TYPE_MAPPING = {'A': 0, '1': 1, 'B': 2, 'U': 3}
# 整数からdocument_typeへの逆マッピング
TYPE_REVERSE_MAPPING = {0: 'A', 1: '1', 2: 'B', 3: 'U'}

# # A_path_01 - 08
# A_PATH_FILE = 'A_path_'
# A_PATH_FILES = []
# for i in range(1, 9):
#     path_file = PATH_DATA_DIR / f'{A_PATH_FILE}{str(i).zfill(2)}.csv'
#     A_PATH_FILES.append(path_file)
#     if not path_file.exists():
#         raise FileNotFoundError(f"Required file not found: {path_file}")

def search_path(top_k_df: pd.DataFrame, top_k=None) -> pd.DataFrame:
    # top_k_dfにpathカラムを追加、NaNで初期化
    top_k_df['table_name'] = None
    top_k_df['name'] = None
    top_k_number = None

    if top_k is None:
        top_k = len(top_k_df)

    top_k_counter = 0
    for idx, row in top_k_df.iterrows():
        if top_k_counter >= top_k:
            break
        publication_number = row['publication_number']
        # remove - from publication_number
        publication_number = publication_number.replace('-', '')
        # publication_number 'JP6343124B2'から"JP", "B2"を取り除く
        # JPの後ろの数字の後、アルファベットがあれば、そこから最後まで文字列をdocument_type変数として取得し、
        # このときもしアルファベットがなければ、document_typeは空文字列とする
        # これによって、publication_numberには数字のみが残る
        #　この数字を整数に変換し、int_publication_number変数に格納する

        # 正規表現で高速に抽出: 最初の非数字部分をスキップし、数字部分と残りを取得
        match = re.match(r'^\D*(\d+)(.*)$', publication_number)
        if match:
            publication_number_str = match.group(1)
            int_publication_number = int(publication_number_str)
            #最初の４文字は年号なので、year_4_digit変数に格納し、
            year_4_digit = str(int_publication_number)[:4]
            # int_publication_number = int(str(int_publication_number)[4:])   
            
            document_type = match.group(2)
            # document typeが"A1"や"B2"のように数字を含む場合、最初の文字だけを取得
            document_type = re.match(r'^[A-Z]', document_type).group(0) if re.match(r'^[A-Z]', document_type) else ''
        else:
            int_publication_number = 0
            document_type = ''

        # NUM_PATH_ARRAYをここで使用
        # numpy_file.pyを参照し、NUM_PATH_ARRAYから
        # doc_number列がint_publication_numberと等しく、type列がdocument_type
        # に対応する整数と等しい行を検索する

        # document_typeを整数に変換
        document_type_int = TYPE_MAPPING.get(document_type, -1)

        # document_typeが不明な場合はスキップ
        # -1: typeなし、0:A、1:1、2:B、3:U
        if document_type != "A":
            continue

        # NUM_PATH_ARRAYから検索
        # 配列の列構造: [doc_number (0), table_name (1), type (2)]
        mask = (NUM_PATH_ARRAY[:, 0] == int_publication_number) & (NUM_PATH_ARRAY[:, 2] == document_type_int)
        matched_rows = NUM_PATH_ARRAY[mask]

        # マッチする行が見つからなければスキップ
        if len(matched_rows) == 0:
            continue

        # 最初の一致行からtable_nameを取得
        table_name = matched_rows[0, 1]
        top_k_df.loc[idx, 'table_name'] = str(table_name)
        top_k_df.loc[idx, 'name'] = publication_number
        top_k_df.loc[idx, 'number'] = publication_number_str

        top_k_counter += 1
    # top_k_dfから、table_nameがNoneでない行だけを抽出して返す
    top_k_df = top_k_df[top_k_df['table_name'].notna()].reset_index(drop=True)

    return top_k_df