"""BigQueryでpatent_lookupテーブルを作成"""

from google.cloud import bigquery
import copy

PROJECT_ID = "llmatch-471107"
DATASET_ID = "dataset_lookup"
TABLE_ID = "patent_lookup"
# TABLE_ID = "patent_lookup_application"
SOURCE_DATASET = "dataset03"


def create_patent_lookup_table():
    """patent_lookupテーブルを作成"""
    client = bigquery.Client(project=PROJECT_ID, location="us-central1")

    query = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    CLUSTER BY doc_number
    OPTIONS(
        description="Application doc_number to result_X table mapping",
        labels=[("purpose", "lookup_index"), ("key_type", "application_number")]
    )
    AS
    SELECT
        _TABLE_SUFFIX AS result_table,
        application.doc_number, -- ここを変更しました
        path
    FROM `{PROJECT_ID}.{SOURCE_DATASET}.result_*`
    WHERE _TABLE_SUFFIX BETWEEN '1' AND '18'
    """

    # query = f"""
    # CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    # CLUSTER BY doc_number
    # OPTIONS(
    #     description="Patent doc_number to result_X table mapping",
    #     labels=[("purpose", "lookup_index")]
    # )
    # AS
    # SELECT
    #     _TABLE_SUFFIX AS result_table,
    #     publication.doc_number,
    #     path
    # FROM `{PROJECT_ID}.{SOURCE_DATASET}.result_*`
    # WHERE _TABLE_SUFFIX BETWEEN '1' AND '18'
    # """

    print(f"テーブル作成中: {DATASET_ID}.{TABLE_ID}")
    query_job = client.query(query)
    query_job.result()

    table = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
    print(f"完了: {table.num_rows:,} 件, {table.num_bytes / 1024**2:.2f} MB")


def find_documents_batch(publication_numbers):
    client = bigquery.Client(project=PROJECT_ID)
    
    # 入力リストをTOP_Kで切り取る（必要であれば）
    # publication_numbers = publication_numbers[:TOP_K]

    # SQL: UNNESTを使って配列を展開し、LIKE検索でJOINする
    # DISTINCTをつけることで、複数の検索値に同じドキュメントがヒットした場合の重複を除去します
    query = f"""
        SELECT DISTINCT
            t.result_table, 
            t.doc_number,
            t.path 
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS t
        INNER JOIN UNNEST(@pub_nums_array) AS input_num
            ON t.doc_number LIKE CONCAT('%', input_num, '%')
    """

    # リストをARRAYパラメータとして渡す設定
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("pub_nums_array", "STRING", publication_numbers)
        ]
    )

    # 1回だけクエリを実行
    query_job = client.query(query, job_config=job_config)
    results = list(query_job.result())

    # 結果を辞書リストで返す
    result_dicts = [dict(row) for row in results]
    return result_dicts

DEBUG = False

def get_abstract_claims_by_query(top_k_df):
    
    client = bigquery.Client(project=PROJECT_ID)

    name_table_dict = {}
    abstraccts_claims_list = []

    for _, row in top_k_df.iterrows():
        table_name = row['table_name']
        publication_number = row['number']
        if table_name is None:
            continue
        # table_nameをキーにして、publication_numberをリストでまとめる
        if table_name not in name_table_dict:
            name_table_dict[table_name] = []
        name_table_dict[table_name].append(publication_number)


    for table_name, name_list in name_table_dict.items():
        # doc_infosからpathを取得し、クエリ対象の文献番号リストを作成
        # '/tmp/tmpn5es9j7o/result_16/3/JP2025021568A/text.txt'
        # JP2025021568Aを取得
        table_name = f"result_{table_name}"

        query = f"""
            SELECT 
                publication.doc_number,
                abstract,
                claims
            FROM `{PROJECT_ID}.{SOURCE_DATASET}.{table_name}`
            WHERE publication.doc_number IN UNNEST(@doc_numbers_array)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("doc_numbers_array", "STRING", name_list)
            ]
        )

        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        for row in results:
            row_dict = {row["doc_number"]: (row["abstract"], row["claims"])}
            # find n-th row in top_k_df where publication_number == row["doc_number"]
            # get index of that row
            n_th_row_index = top_k_df.index[top_k_df['number'] == row["doc_number"]].tolist()[0]
            # pandas series to dict
            row_dict = dict(row)
            row_dict["top_k"] = n_th_row_index + 1
            abstraccts_claims_list.append(copy.deepcopy(row_dict))

        if DEBUG:# デバッグモード注意
            print("DEBUG: get_abstract_claims_by_query ")
            return abstraccts_claims_list
    
    return abstraccts_claims_list



