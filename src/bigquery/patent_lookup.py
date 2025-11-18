"""BigQueryでpatent_lookupテーブルを作成"""

from google.cloud import bigquery

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





def find_patent_document(publication_number):
    # SELECT doc_number FROM `llmatch-471107.dataset_lookup.patent_lookup` LIMIT 1000
    """publication_numberに対応するpatent_lookupエントリを検索"""
    client = bigquery.Client(project=PROJECT_ID)

# ---------------------------------------------------------
# Lookupテーブルのみを検索 (データ量が少ないので安くて速い)
# ---------------------------------------------------------
    query = f"""
        SELECT 
            result_table, 
            doc_number,
            path 
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE doc_number LIKE CONCAT('%', @publication_number, '%')
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("publication_number", "STRING", publication_number)
        ]
    )

    query_job = client.query(query, job_config=job_config)
    results = list(query_job.result())

    # resultを辞書形式で返す
    result_dicts = [dict(row) for row in results]
    return result_dicts

if __name__ == "__main__":
    pass
    # create_patent_lookup_table()