import os
import csv
from collections import defaultdict
from typing import Tuple, Generator, Dict
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
import queue

# --- 設定変数 ---
# データを配置するルートディレクトリ
BASE_DIR = "/mnt/eightthdd/raw_data"
# CSVの出力先ディレクトリ
OUTPUT_DIR = "/home/sonozuka/staging/patent_rag/data/path"
# 1ファイルあたりの最大行数
MAX_ROWS_PER_FILE = 500000
# CSVファイル書き込みバッファサイズ（メモリ効率のため）
BUFFER_SIZE = 10000
# マルチプロセスのワーカー数（Noneの場合はCPUコア数を使用）
NUM_WORKERS = None
# プログレスバーの予想総件数（プログレスバー表示用）
EXPECTED_TOTAL_ITEMS = 2000000

# --- 抽出関数 ---

def extract_info(full_path: str) -> Tuple[str, str, str]:
    """
    パスからdoc_numberとdoc_idを抽出し、CSV行のタプルを返します。
    """
    doc_id_with_prefix = os.path.basename(full_path)
    doc_id = doc_id_with_prefix

    doc_number = "ERROR"
    try:
        # JP2010000001A -> 0000001 の抽出ロジック
        number_part_raw = doc_id[:-1][6:]
        doc_number = number_part_raw.lstrip('0')
        doc_number = doc_number.zfill(7)
    except (IndexError, ValueError):
        pass  # doc_number は "ERROR" のまま

    return doc_number, doc_id, full_path

# --- マルチプロセス用関数 ---

def process_chunk_directory(chunk_dir_path: Path, result_queue: Queue):
    """
    チャンクディレクトリを処理し、結果をキューに送信します。

    Args:
        chunk_dir_path: チャンクディレクトリのパス
        result_queue: 結果を送信するキュー
    """
    try:
        count = 0
        for doc_dir in chunk_dir_path.iterdir():
            if not doc_dir.is_dir():
                continue

            # 最終文字を取得 (例: 'A')
            doc_dir_name = doc_dir.name
            suffix = doc_dir_name[-1].upper()

            # 情報を抽出
            doc_number, doc_id, path = extract_info(str(doc_dir))

            # 結果をキューに送信
            result_queue.put(('data', suffix, doc_number, doc_id, path))
            count += 1

        # 処理完了を通知（カウント情報付き）
        result_queue.put(('progress', count))

    except PermissionError as e:
        result_queue.put(('error', f"警告: {chunk_dir_path} へのアクセスが拒否されました: {e}"))
    except Exception as e:
        result_queue.put(('error', f"エラー: {chunk_dir_path} の処理中にエラーが発生: {e}"))


def writer_process(result_queue: Queue, output_dir: str, max_rows_per_file: int, total_chunks: int):
    """
    Queueからデータを受け取り、CSVファイルに書き込む専用プロセス。

    【同時書き込み回避の仕組み】
    - このプロセスは**1つだけ**起動され、すべてのCSV書き込みを担当
    - 複数のワーカープロセスからのデータは result_queue で集約される
    - StreamingCSVWriter のインスタンスも**1つだけ**作成される
    - すべてのファイル書き込みは**順次的**に行われるため、同時書き込みは発生しない

    Args:
        result_queue: データを受け取るキュー
        output_dir: 出力ディレクトリ
        max_rows_per_file: 1ファイルあたりの最大行数
        total_chunks: 総チャンク数（プログレスバー用）
    """
    # 重要: StreamingCSVWriterのインスタンスは1つのみ
    # すべてのsuffixのファイルを一元管理し、同時書き込みを回避
    writer = StreamingCSVWriter(output_dir, max_rows_per_file)

    # マルチプロセス対応のプログレスバー
    # チャンクの処理進捗を表示
    chunk_pbar = tqdm(total=total_chunks, desc="チャンク処理", unit="チャンク", position=0, dynamic_ncols=True)

    try:
        while True:
            try:
                # キューからデータを取得（タイムアウト付き）
                item = result_queue.get(timeout=1)

                if item is None:  # 終了シグナル
                    break

                msg_type = item[0]

                if msg_type == 'data':
                    _, suffix, doc_number, doc_id, path = item
                    writer.write_row(suffix, doc_number, doc_id, path)
                elif msg_type == 'error':
                    _, error_msg = item
                    tqdm.write(error_msg)
                elif msg_type == 'progress':
                    # ワーカーがチャンクの処理を完了したときに送信される
                    # プログレスバーを1チャンク分進める
                    chunk_pbar.update(1)

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\nWriterプロセスが中断されました")
    finally:
        chunk_pbar.close()
        writer.close()


# --- ジェネレータ関数 (ストリーミング処理) ---

def iter_directories(base_dir: str) -> Generator[Tuple[str, str, str, str], None, None]:
    """
    ベースディレクトリ配下の全ディレクトリを1つずつyieldします。
    メモリに全データを読み込まずに、順次処理できます。

    Yields:
        (suffix, doc_number, doc_id, path) のタプル
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"エラー: ベースディレクトリ {base_dir} が見つかりません。")
        return

    # プログレスバーを作成（総数は不明なのでtotal=Noneで動的に更新）
    pbar = tqdm(desc="ディレクトリスキャン", unit="件", dynamic_ncols=True)

    # 中間ディレクトリを順次処理
    for intermediate_dir in sorted(base_path.iterdir()):
        if not intermediate_dir.is_dir():
            continue

        # チャンクディレクトリ（例: '0', '1', '2', ...）を順次処理
        for chunk_dir in sorted(intermediate_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue

            pbar.set_postfix_str(f"処理中: {intermediate_dir.name}/{chunk_dir.name}")

            # チャンク内のドキュメントディレクトリを順次処理
            try:
                for doc_dir in chunk_dir.iterdir():
                    if not doc_dir.is_dir():
                        continue

                    # 最終文字を取得 (例: 'A')
                    doc_dir_name = doc_dir.name
                    suffix = doc_dir_name[-1].upper()

                    # 情報を抽出
                    doc_number, doc_id, path = extract_info(str(doc_dir))

                    # 1件ずつyield（メモリに溜め込まない）
                    yield suffix, doc_number, doc_id, path

                    # プログレスバーを更新
                    pbar.update(1)

            except PermissionError as e:
                tqdm.write(f"警告: {chunk_dir} へのアクセスが拒否されました: {e}")
                continue
            except Exception as e:
                tqdm.write(f"エラー: {chunk_dir} の処理中にエラーが発生: {e}")
                continue

    pbar.close()
    print(f"\n合計 {pbar.n:,} 件のディレクトリを処理しました")

# --- ストリーミングCSV書き込み ---

class StreamingCSVWriter:
    """
    suffix（末尾文字）ごとにCSVファイルをストリーミング書き込みするクラス。
    バッファリングを使用してメモリ効率を最適化します。
    """

    def __init__(self, output_dir: str, max_rows_per_file: int = MAX_ROWS_PER_FILE):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rows_per_file = max_rows_per_file

        # suffix ごとの状態を管理
        self.file_handles: Dict[str, any] = {}
        self.csv_writers: Dict[str, csv.writer] = {}
        self.row_counts: Dict[str, int] = defaultdict(int)
        self.file_numbers: Dict[str, int] = defaultdict(int)
        self.buffers: Dict[str, list] = defaultdict(list)

        self.header = ["doc_number", "doc_id", "path"]
        self.total_written = 0

        # プログレスバーを作成（position=1でチャンクプログレスバーの下に表示）
        # 総数を200万件として表示
        self.pbar = tqdm(total=EXPECTED_TOTAL_ITEMS, desc="CSV書き込み", unit="件", position=1, dynamic_ncols=True, leave=True)

    def _get_file_path(self, suffix: str, file_number: int) -> Path:
        """CSVファイルのパスを生成"""
        file_name = f"{suffix}_path_{file_number:02d}.csv"
        return self.output_dir / file_name

    def _open_new_file(self, suffix: str):
        """新しいCSVファイルを開く"""
        # 既存のファイルを閉じる
        if suffix in self.file_handles:
            self._flush_buffer(suffix)
            self.file_handles[suffix].close()

        # 新しいファイルを開く
        self.file_numbers[suffix] += 1
        file_path = self._get_file_path(suffix, self.file_numbers[suffix])

        f = open(file_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(self.header)

        self.file_handles[suffix] = f
        self.csv_writers[suffix] = writer
        self.row_counts[suffix] = 0

        tqdm.write(f"  新しいファイルを作成: {file_path.name}")

    def _flush_buffer(self, suffix: str):
        """バッファの内容をファイルに書き込む"""
        if suffix in self.buffers and self.buffers[suffix]:
            self.csv_writers[suffix].writerows(self.buffers[suffix])
            self.buffers[suffix].clear()

    def write_row(self, suffix: str, doc_number: str, doc_id: str, path: str):
        """1行をCSVファイルに書き込む（バッファリング付き）"""
        # ファイルが開かれていない、または最大行数に達した場合は新しいファイルを開く
        if suffix not in self.file_handles or self.row_counts[suffix] >= self.max_rows_per_file:
            if suffix in self.file_handles:
                self._flush_buffer(suffix)
            self._open_new_file(suffix)

        # バッファに追加
        self.buffers[suffix].append((doc_number, doc_id, path))
        self.row_counts[suffix] += 1
        self.total_written += 1

        # プログレスバーを更新
        self.pbar.update(1)
        self.pbar.set_postfix_str(f"suffix: {suffix}")

        # バッファサイズに達したらフラッシュ
        if len(self.buffers[suffix]) >= BUFFER_SIZE:
            self._flush_buffer(suffix)

    def close(self):
        """全てのファイルを閉じる"""
        for suffix in list(self.file_handles.keys()):
            self._flush_buffer(suffix)
            self.file_handles[suffix].close()

        # プログレスバーを閉じる
        self.pbar.close()

        print(f"\n合計 {self.total_written:,} 件のデータをCSVファイルに書き込みました")

        # 作成されたファイルのサマリーを表示
        print("\n作成されたファイル:")
        for suffix in sorted(self.file_numbers.keys()):
            num_files = self.file_numbers[suffix]
            print(f"  {suffix}_path_*.csv: {num_files} ファイル")

# --- メイン処理 ---

def worker_process(task_queue: Queue, result_queue: Queue):
    """
    タスクキューからチャンクディレクトリを取得し、処理するワーカープロセス。

    Args:
        task_queue: 処理するチャンクディレクトリのパスを受け取るキュー
        result_queue: 処理結果を送信するキュー
    """
    try:
        while True:
            try:
                chunk_dir_path = task_queue.get(timeout=1)

                if chunk_dir_path is None:  # 終了シグナル
                    break

                # チャンクディレクトリを処理
                process_chunk_directory(chunk_dir_path, result_queue)

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        pass


def main_multiprocess():
    """
    マルチプロセス版のメイン処理。
    チャンクディレクトリを並列処理し、専用Writerプロセスに結果を送信します。

    【アーキテクチャ - 同時書き込み回避の設計】

    メインプロセス
      │
      ├─ タスクキュー (task_queue)
      │   └→ チャンクディレクトリのパスを配布
      │
      ├─ ワーカープロセス × N個 (並列実行)
      │   ├→ ワーカー1: タスクキューからチャンクを取得 → ディレクトリスキャン → result_queueに送信
      │   ├→ ワーカー2: タスクキューからチャンクを取得 → ディレクトリスキャン → result_queueに送信
      │   └→ ...
      │
      ├─ 結果キュー (result_queue)
      │   └→ すべてのワーカーからのデータを集約
      │
      └─ Writerプロセス × 1個のみ (単一実行) ★重要★
          └→ result_queueからデータを取得 → CSV書き込み（順次処理）

    このアーキテクチャにより:
    - 複数のワーカーが並列でディレクトリをスキャン（高速化）
    - 単一のWriterがすべての書き込みを担当（同時書き込み回避）
    - Queueによる安全なプロセス間通信
    """
    print("=" * 60)
    print("ディレクトリスキャン & CSV生成（マルチプロセス版）")
    print("=" * 60)
    print(f"入力ディレクトリ: {BASE_DIR}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"1ファイルあたりの最大行数: {MAX_ROWS_PER_FILE:,}")
    print(f"バッファサイズ: {BUFFER_SIZE:,}")

    num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count()
    print(f"ワーカープロセス数: {num_workers}")
    print("=" * 60)

    base_path = Path(BASE_DIR)

    if not base_path.exists():
        print(f"エラー: ベースディレクトリ {BASE_DIR} が見つかりません。")
        return

    # チャンクディレクトリのリストを収集
    print("\nチャンクディレクトリを収集中...")
    chunk_dirs = []
    for intermediate_dir in sorted(base_path.iterdir()):
        if not intermediate_dir.is_dir():
            continue
        for chunk_dir in sorted(intermediate_dir.iterdir()):
            if chunk_dir.is_dir():
                chunk_dirs.append(chunk_dir)

    print(f"合計 {len(chunk_dirs)} 個のチャンクディレクトリを発見しました")

    # タスクキューと結果キューを作成
    task_queue = Queue()
    result_queue = Queue(maxsize=num_workers * 1000)

    # === 同時書き込み回避: Writerプロセスは1つのみ起動 ===
    # すべてのワーカーからのデータは、この単一のWriterプロセスに集約され、
    # 順次的にCSVファイルに書き込まれます。
    print("\nWriterプロセスを起動中...")
    writer_proc = Process(target=writer_process, args=(result_queue, OUTPUT_DIR, MAX_ROWS_PER_FILE, len(chunk_dirs)))
    writer_proc.start()

    # ワーカープロセスを起動（複数）
    # これらのプロセスはファイルに直接書き込まず、result_queueにデータを送信するのみ
    print(f"{num_workers} 個のワーカープロセスを起動中...")
    workers = []
    for i in range(num_workers):
        worker = Process(target=worker_process, args=(task_queue, result_queue))
        worker.start()
        workers.append(worker)

    try:
        # タスクキューにチャンクディレクトリを投入
        print("\nタスクを投入中...")
        for chunk_dir in chunk_dirs:
            task_queue.put(chunk_dir)

        # ワーカープロセスに終了シグナルを送信
        for _ in range(num_workers):
            task_queue.put(None)

        # ワーカープロセスの終了を待つ
        # プログレスバーはWriterプロセスで表示される
        print("\n処理を開始します...")
        for worker in workers:
            worker.join()

        print("\nすべてのワーカープロセスが完了しました")

        # Writerプロセスに終了シグナルを送信
        result_queue.put(None)

    except KeyboardInterrupt:
        print("\n\n処理が中断されました")
        # ワーカープロセスを強制終了
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Writerプロセスの終了を待つ
        print("\nWriterプロセスの終了を待機中...")
        writer_proc.join()

    print("=" * 60)
    print("処理が完了しました")
    print(f"CSVファイルは {OUTPUT_DIR} に保存されました")
    print("=" * 60)


def main():
    """
    ストリーミング処理で200万件のディレクトリを読み込み、
    CSVファイルに書き込みます。
    """
    print("=" * 60)
    print("ディレクトリスキャン & CSV生成（ストリーミング版）")
    print("=" * 60)
    print(f"入力ディレクトリ: {BASE_DIR}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"1ファイルあたりの最大行数: {MAX_ROWS_PER_FILE:,}")
    print(f"バッファサイズ: {BUFFER_SIZE:,}")
    print("=" * 60)

    # CSVライターを初期化
    writer = StreamingCSVWriter(OUTPUT_DIR, MAX_ROWS_PER_FILE)

    try:
        # ディレクトリを順次読み込んでCSVに書き込む
        print("\nディレクトリのスキャンを開始...")
        for suffix, doc_number, doc_id, path in iter_directories(BASE_DIR):
            writer.write_row(suffix, doc_number, doc_id, path)

        print("\nすべてのディレクトリのスキャンが完了しました")

    except KeyboardInterrupt:
        print("\n\n処理が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 必ずファイルを閉じる
        print("\nCSVファイルを保存中...")
        writer.close()

    print("=" * 60)
    print("処理が完了しました")
    print(f"CSVファイルは {OUTPUT_DIR} に保存されました")
    print("=" * 60)

if __name__ == "__main__":
    # マルチプロセス版を使用
    main_multiprocess()

    # シングルプロセス版を使用する場合は以下のコメントを解除
    # main()
